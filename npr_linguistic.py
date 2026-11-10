import argparse
import os
import random
import time
import urllib.request
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from npr_core import (BASE_NAMES, PrimitiveLibrary, ProgramMemory, RuleSynthesizer, candidate_logits,
                      compress_library, diversity_loss, execute_program, make_schedule, novelty_loss, pick_device,
                      save_json, set_seed, steps_from_memory)

DEVICE = pick_device()
ANALOGY_URL = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
ANALOGY_FILE = "questions-words.txt"

CONFIG = {
    "proj_dim": 256,
    "max_program_steps": 6,
    "min_program_steps": 1,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "num_examples": 5,
    "memory_capacity": 500,
    "memory_threshold": 0.9,
    "compression_threshold": 8,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [6000, 4000, 2000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "compositional_ratio": 0.3,
    "test_fraction": 0.2,
    "max_relations": 12,
    "tau": 0.05,
    "latent_weight": 0.5,
    "diversity_weight": 0.05,
    "novelty_weight": 0.1,
    "ponder_weight": 0.01,
    "decoder_bypass": False,
    "use_program": True,
    "eval_seed": 1234,
    "gpt2_eval_limit": 50,
    "gpt2_batch": 128,
}

ABLATIONS = {
    "none": {},
    "decoder_bypass": {"decoder_bypass": True},
    "no_program": {"use_program": False},
}


def load_relations(max_relations):
    if not os.path.exists(ANALOGY_FILE):
        print("downloading Google analogy dataset")
        urllib.request.urlretrieve(ANALOGY_URL, ANALOGY_FILE)
    relations, current = defaultdict(set), None
    with open(ANALOGY_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current = line[1:].strip().lower()
                continue
            parts = line.lower().split()
            if current and len(parts) == 4:
                relations[current].add((parts[0], parts[1]))
                relations[current].add((parts[2], parts[3]))
    rels = {k: sorted(v) for k, v in relations.items() if len(v) >= 10}
    keep = sorted(rels, key=lambda k: -len(rels[k]))[:max_relations]
    return {k: rels[k] for k in sorted(keep)}


def split_relations(rels, frac, seed):
    rng = random.Random(seed)
    train, test = {}, {}
    for name, pairs in rels.items():
        sources = sorted({a for a, _ in pairs})
        rng.shuffle(sources)
        held = set(sources[:max(1, int(len(sources) * frac))])
        train[name] = [p for p in pairs if p[0] not in held]
        test[name] = [p for p in pairs if p[0] in held]
    return train, test


def find_chains(rels, max_depth=3, min_count=5):
    by_source = defaultdict(list)
    for r, pairs in rels.items():
        for a, b in pairs:
            by_source[a].append((r, b))
    chains = defaultdict(list)
    for r1, pairs in rels.items():
        for a, b in pairs:
            frontier = [([a, b], [r1])]
            for _ in range(max_depth - 1):
                nxt = []
                for words, names in frontier:
                    for r, c in by_source.get(words[-1], []):
                        if r != names[-1] and c not in words:
                            item = (words + [c], names + [r])
                            chains["+".join(item[1])].append(item)
                            nxt.append(item)
                frontier = nxt
    return {k: v for k, v in chains.items() if len(v) >= min_count}


def chain_pairs(chain):
    words, names = chain
    return [(names[i], words[i], words[i + 1]) for i in range(len(names))]


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="ignore")


def load_perceiver(kind, seed):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    if kind == "gpt2":
        return GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval(), GPT2Tokenizer.from_pretrained("gpt2")
    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=256, n_positions=256, bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(config).to(DEVICE).eval(), ByteTokenizer()


class WordEncoder:
    def __init__(self, lm, tokenizer, words, train_words, layer):
        self.lm, self.tokenizer, self.layer = lm, tokenizer, layer
        raw = {}
        with torch.no_grad():
            for w in sorted(words):
                ids = self.tokenizer.encode(" " + w)
                out = self.lm(torch.tensor([ids], device=DEVICE), output_hidden_states=True)
                raw[w] = out.hidden_states[layer][0].mean(0)
        train = torch.stack([raw[w] for w in sorted(train_words)])
        center = train.mean(0)
        scale = (train - center).norm(dim=-1).mean().clamp_min(1e-6)
        self.vec = {w: (v - center) / scale for w, v in raw.items()}
        self.words = sorted(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.matrix = torch.stack([self.vec[w] for w in self.words])
        self.dim = self.matrix.shape[1]


class NPR(nn.Module):
    def __init__(self, sd, cfg):
        super().__init__()
        self.cfg = cfg
        pd = cfg["proj_dim"]
        self.lib = PrimitiveLibrary(sd, max_macros=cfg["max_new_primitives"] * max(cfg["num_cycles"] - 1, 1))
        self.syn = RuleSynthesizer(sd * 3, sd, pd, self.lib.capacity, cfg["max_program_steps"],
                                   cfg["min_program_steps"])
        self.adapt = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd), nn.GELU(), nn.Linear(sd, sd))
        self.fallback = None if cfg["use_program"] else nn.Sequential(
            nn.Linear(sd * 2, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))
        head_in = sd + (sd + pd if cfg["decoder_bypass"] else 0)
        self.head = nn.Sequential(nn.Linear(head_in, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))
        self.mem = ProgramMemory(cfg["memory_capacity"], cfg["memory_threshold"])

    def forward(self, enc, task, temp=0.8, use_mem=True, override=None):
        ex = torch.stack([torch.cat([enc.vec[a], enc.vec[b], enc.vec[b] - enc.vec[a]]) for a, b in task["examples"]])
        sig = self.syn.signature(ex)
        code = self.syn.code(sig)
        x = enc.vec[task["word"]]
        s = self.adapt(x)
        info = {"sig": sig, "steps": [], "length": torch.zeros((), device=s.device), "prog": []}
        if not self.cfg["use_program"]:
            out = s + self.fallback(torch.cat([s, code]))
        else:
            if override is not None:
                prog, halts = steps_from_memory(override, self.lib.capacity, self.lib.n, s.device), []
            else:
                remembered = self.mem.lookup(sig) if (use_mem and not self.training) else None
                if remembered is not None:
                    prog, halts = steps_from_memory(remembered, self.lib.capacity, self.lib.n, s.device), []
                else:
                    prog, halts = self.syn(sig, s, self.lib.n, temp)
            out, steps, length = execute_program(self.lib, prog, halts, s, code, self.cfg["min_program_steps"],
                                                 self.training)
            info.update({"steps": steps, "length": length, "prog": prog})
        parts = [out] + ([x, sig] if self.cfg["decoder_bypass"] else [])
        return self.head(torch.cat(parts)), info


def make_task(rel, pairs_train, test_pair, n, rng, depth=1):
    pool = [p for p in pairs_train if p[0] != test_pair[0] and p[1] != test_pair[1]]
    examples = rng.sample(pool, min(n, len(pool)))
    return {"relation": rel, "examples": examples, "word": test_pair[0], "expected": test_pair[1], "depth": depth}


def chain_task(name, chains, test_chain, n, rng):
    pool = [c for c in chains if c[0][0] != test_chain[0][0] and c[0][-1] != test_chain[0][-1]]
    picked = rng.sample(pool, min(n, len(pool)))
    return {"relation": name, "examples": [(c[0][0], c[0][-1]) for c in picked], "word": test_chain[0][0],
            "expected": test_chain[0][-1], "depth": len(test_chain[1])}


def split_chains(chains, test_pairs):
    held = {(r, a, b) for r, pairs in test_pairs.items() for a, b in pairs}
    train, test = {}, {}
    for name, items in chains.items():
        tr = [c for c in items if not any(p in held for p in chain_pairs(c))]
        te = [c for c in items if chain_pairs(c)[0] in held]
        if len(tr) >= 4:
            train[name] = tr
            if te:
                test[name] = te
    return train, test


def masked_logits(pred, enc, task, tau):
    logits = candidate_logits(pred, enc.matrix, tau)
    logits[enc.index[task["word"]]] = float("-inf")
    return logits


def train_cycle(model, enc, train, chains_train, cfg, n_iters, lr, cycle, rng):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sch = make_schedule(opt, max(n_iters // cfg["grad_accumulation"], 1), warmup=20)
    model.train()
    ok = tot = 0
    t0 = time.time()
    rel_names = sorted(train)
    chain_names = sorted(chains_train)
    for it in range(n_iters):
        if chain_names and rng.random() < cfg["compositional_ratio"]:
            name = rng.choice(chain_names)
            task = chain_task(name, chains_train[name], rng.choice(chains_train[name]), cfg["num_examples"], rng)
        else:
            rel = rng.choice(rel_names)
            task = make_task(rel, train[rel], rng.choice(train[rel]), cfg["num_examples"], rng)
        pred, info = model(enc, task, cfg["temperature"], use_mem=False)
        logits = masked_logits(pred, enc, task, cfg["tau"])
        tgt = torch.tensor([enc.index[task["expected"]]], device=pred.device)
        ce = F.cross_entropy(logits.unsqueeze(0), tgt)
        latent = 1.0 - F.cosine_similarity(pred.unsqueeze(0), enc.vec[task["expected"]].unsqueeze(0)).squeeze()
        loss = ce + cfg["latent_weight"] * latent
        if info["prog"]:
            loss = loss + cfg["diversity_weight"] * diversity_loss(info["prog"])
            loss = loss + cfg["novelty_weight"] * novelty_loss(info["prog"], len(BASE_NAMES))
            loss = loss + cfg["ponder_weight"] * info["length"] / cfg["max_program_steps"]
        (loss / cfg["grad_accumulation"]).backward()
        correct = int(logits.argmax().item()) == enc.index[task["expected"]]
        ok += correct
        tot += 1
        if info["steps"]:
            model.mem.store(info["sig"], info["steps"], task["relation"], correct)
        if (it + 1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            sch.step()
        if it % 500 == 0:
            speed = (it + 1) / max(time.time() - t0, 1e-6)
            print(f"  [C{cycle}] {it:6d}/{n_iters} ce={ce.item():.3f} train_acc={100 * ok / max(tot, 1):5.1f}% "
                  f"{speed:.1f}it/s")
            ok = tot = 0


def offset_baseline(enc, task, tau):
    delta = torch.stack([enc.vec[b] - enc.vec[a] for a, b in task["examples"]]).mean(0)
    logits = masked_logits(enc.vec[task["word"]] + delta, enc, task, tau)
    return enc.words[int(logits.argmax().item())]


class GPT2Scorer:
    def __init__(self, lm, tokenizer, words, batch):
        self.lm, self.tokenizer, self.batch = lm, tokenizer, batch
        self.words = sorted(words)
        self.ids = [tokenizer.encode(" " + w) for w in self.words]

    def __call__(self, task):
        prompt = "\n".join(f"{a} means {b}" for a, b in task["examples"]) + f"\n{task['word']} means"
        p = self.tokenizer.encode(prompt)
        scores = []
        with torch.no_grad():
            for i in range(0, len(self.ids), self.batch):
                chunk = self.ids[i:i + self.batch]
                length = len(p) + max(len(c) for c in chunk)
                inp = torch.zeros(len(chunk), length, dtype=torch.long, device=DEVICE)
                mask = torch.zeros(len(chunk), length, dtype=torch.long, device=DEVICE)
                for j, c in enumerate(chunk):
                    seq = p + c
                    inp[j, :len(seq)] = torch.tensor(seq, device=DEVICE)
                    mask[j, :len(seq)] = 1
                logp = F.log_softmax(self.lm(input_ids=inp, attention_mask=mask).logits, dim=-1)
                for j, c in enumerate(chunk):
                    pos = torch.arange(len(p) - 1, len(p) - 1 + len(c), device=DEVICE)
                    scores.append(logp[j, pos, torch.tensor(c, device=DEVICE)].sum().item())
        order = sorted(range(len(self.words)), key=lambda k: -scores[k])
        for k in order:
            if self.words[k] != task["word"]:
                return self.words[k]
        return None


def evaluate(model, enc, train, test, chains_train, chains_test, cfg, gpt2_scorer=None):
    model.eval()
    rng = random.Random(cfg["eval_seed"])
    report = {"relations": {}, "compositional": {}}
    usage = defaultdict(Counter)
    with torch.no_grad():
        for rel in sorted(test):
            hits = defaultdict(list)
            for i, pair in enumerate(test[rel]):
                task = make_task(rel, train[rel], pair, cfg["num_examples"], rng)
                pred, info = model(enc, task, 0.1)
                out = enc.words[int(masked_logits(pred, enc, task, cfg["tau"]).argmax().item())]
                hits["npr"].append(out == task["expected"])
                hits["vector_offset"].append(offset_baseline(enc, task, cfg["tau"]) == task["expected"])
                if gpt2_scorer is not None and i < cfg["gpt2_eval_limit"]:
                    hits["gpt2_fewshot"].append(gpt2_scorer(task) == task["expected"])
                usage[rel][" -> ".join(model.lib.names[s] for s in info["steps"])] += 1
            report["relations"][rel] = {"n": len(test[rel]),
                                        **{k: round(100.0 * sum(v) / max(len(v), 1), 2) for k, v in hits.items()}}
        for name in sorted(chains_test):
            hits = defaultdict(list)
            for c in chains_test[name][:30]:
                task = chain_task(name, chains_train[name], c, 3, rng)
                pred, _ = model(enc, task, 0.1)
                out = enc.words[int(masked_logits(pred, enc, task, cfg["tau"]).argmax().item())]
                hits["npr"].append(out == task["expected"])
                hits["vector_offset"].append(offset_baseline(enc, task, cfg["tau"]) == task["expected"])
            report["compositional"][name] = {"n": len(hits["npr"]),
                                             **{k: round(100.0 * sum(v) / max(len(v), 1), 2)
                                                for k, v in hits.items()}}
        vecs = enc.matrix[:64]
        report["negate_involution_error"] = model.lib.involution_error(vecs)
    report["programs"] = {rel: dict(c.most_common(3)) for rel, c in sorted(usage.items())}
    tops = {rel: c.most_common(1)[0][0] for rel, c in usage.items() if c}
    report["distinct_programs_across_relations"] = len(set(tops.values()))
    report["library"] = list(model.lib.names)
    return report


def macro_average(block, key):
    vals = [v[key] for v in block.values() if key in v]
    return round(sum(vals) / max(len(vals), 1), 2)


def print_report(report):
    print("\n  held-out pairs, accuracy (%)")
    for rel, row in report["relations"].items():
        cells = " ".join(f"{k}={v:5.1f}" for k, v in row.items() if k != "n")
        print(f"    {rel:28s} n={row['n']:4d} {cells}")
    for key in ["npr", "vector_offset", "gpt2_fewshot"]:
        print(f"    macro {key}: {macro_average(report['relations'], key)}")
    print("\n  held-out compositional chains, accuracy (%)")
    for name, row in report["compositional"].items():
        cells = " ".join(f"{k}={v:5.1f}" for k, v in row.items() if k != "n")
        print(f"    {name[:40]:40s} n={row['n']:3d} {cells}")
    print(f"\n  distinct top programs across relations: {report['distinct_programs_across_relations']}")
    print(f"  NEGATE involution error: {report['negate_involution_error']:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="none")
    parser.add_argument("--perceiver", choices=["gpt2", "random"], default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-gpt2-baseline", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = dict(CONFIG)
    cfg.update(ABLATIONS[args.ablation])
    cfg["iters_per_cycle"] = [max(int(n * args.scale), cfg["grad_accumulation"]) for n in cfg["iters_per_cycle"]]
    rels = load_relations(cfg["max_relations"])
    train, test = split_relations(rels, cfg["test_fraction"], args.seed)
    chains = find_chains(rels)
    chains_train, chains_test = split_chains(chains, test)
    words = {w for pairs in rels.values() for p in pairs for w in p}
    train_words = {w for pairs in train.values() for p in pairs for w in p}
    print(f"device={DEVICE} seed={args.seed} ablation={args.ablation} perceiver={args.perceiver}")
    print(f"relations={len(rels)} words={len(words)} train_pairs={sum(map(len, train.values()))} "
          f"test_pairs={sum(map(len, test.values()))} chain_types={len(chains_train)}")
    lm, tokenizer = load_perceiver(args.perceiver, args.seed)
    enc = WordEncoder(lm, tokenizer, words, train_words, args.layer)
    model = NPR(enc.dim, cfg).to(DEVICE)
    rng = random.Random(args.seed)
    history = []
    for c in range(cfg["num_cycles"]):
        print(f"\n{'=' * 70}\nCYCLE {c + 1}/{cfg['num_cycles']} library={model.lib.names}\n{'=' * 70}")
        train_cycle(model, enc, train, chains_train, cfg, cfg["iters_per_cycle"][c], cfg["lr_per_cycle"][c], c + 1,
                    rng)
        if c < cfg["num_cycles"] - 1:
            created = compress_library(model.lib, model.mem, cfg["compression_threshold"], cfg["max_new_primitives"])
            print(f"  compression: {created if created else 'nothing'}")
            history.append({"cycle": c + 1, "created": created})
    scorer = None
    if args.perceiver == "gpt2" and not args.no_gpt2_baseline:
        scorer = GPT2Scorer(lm, tokenizer, words, cfg["gpt2_batch"])
    report = evaluate(model, enc, train, test, chains_train, chains_test, cfg, scorer)
    report["compression"] = history
    report["config"] = cfg
    report["args"] = vars(args)
    print_report(report)
    out = args.out or f"results_linguistic_{args.ablation}_{args.perceiver}_seed{args.seed}.json"
    save_json(report, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
