import math
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from npr_core import (BASE_NAMES, PrimitiveLibrary, ProgramMemory, RuleSynthesizer, candidate_logits,
                      compress_library, diversity_loss, execute_program, make_schedule, novelty_loss,
                      steps_from_memory)
from npr_world import BASELINES, POLICIES

WORLD_CONFIG = {
    "proj_dim": 256,
    "num_slots": 2,
    "max_program_steps": 6,
    "min_program_steps": 1,
    "temperature": 0.8,
    "grad_accumulation": 16,
    "memory_capacity": 300,
    "memory_threshold": 0.9,
    "compression_threshold": 6,
    "max_new_primitives": 2,
    "num_cycles": 3,
    "iters_per_cycle": [12000, 8000, 4000],
    "lr_per_cycle": [5e-4, 3e-4, 1e-4],
    "num_examples": 5,
    "multistep_ratio": 0.25,
    "policy_weights": [0.4, 0.3, 0.3],
    "tau": 0.05,
    "latent_weight": 0.5,
    "diversity_weight": 0.05,
    "novelty_weight": 0.1,
    "ponder_weight": 0.01,
    "condition": "program",
    "decoder_bypass": False,
    "use_program": True,
    "beam_width": 3,
    "eval_seed": 1234,
}

ABLATIONS = {
    "none": {},
    "decoder_bypass": {"decoder_bypass": True},
    "film": {"condition": "film"},
    "no_program": {"use_program": False},
}


class SlotAttention(nn.Module):
    def __init__(self, sd, num_slots=2, n_iters=2):
        super().__init__()
        self.num_slots, self.n_iters, self.sd = num_slots, n_iters, sd
        self.slot_mu = nn.Parameter(torch.randn(num_slots, sd) * 0.02)
        self.k_proj, self.v_proj, self.q_proj = nn.Linear(sd, sd), nn.Linear(sd, sd), nn.Linear(sd, sd)
        self.gru = nn.GRUCell(sd, sd)
        self.mlp = nn.Sequential(nn.Linear(sd, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))
        self.norm_in, self.norm1, self.norm2 = nn.LayerNorm(sd), nn.LayerNorm(sd), nn.LayerNorm(sd)
        self.scale = math.sqrt(sd)

    def forward(self, tokens):
        x = self.norm_in(tokens)
        k, v = self.k_proj(x), self.v_proj(x)
        slots = self.slot_mu.clone()
        attn = None
        for _ in range(self.n_iters):
            prev = slots
            q = self.q_proj(self.norm1(slots))
            attn = F.softmax(q @ k.T / self.scale, dim=0)
            weights = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            slots = self.gru(weights @ v, prev)
            slots = slots + self.mlp(self.norm2(slots))
        return slots, attn


class ObjectExtractor(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.query = nn.Parameter(torch.randn(sd) * 0.02)
        self.norm = nn.LayerNorm(sd)
        self.k, self.v = nn.Linear(sd, sd), nn.Linear(sd, sd)
        self.out = nn.Sequential(nn.Linear(sd, sd), nn.LayerNorm(sd))
        self.scale = math.sqrt(sd)

    def forward(self, tokens):
        x = self.norm(tokens)
        a = F.softmax(self.k(x) @ self.query / self.scale, dim=0)
        return self.out(a @ self.v(x)), a


class WorldModel(nn.Module):
    def __init__(self, sd, cfg):
        super().__init__()
        self.cfg, self.sd = cfg, sd
        pd = cfg["proj_dim"]
        self.obj_ext = ObjectExtractor(sd)
        self.slot_attn = SlotAttention(sd, cfg["num_slots"])
        self.slot_select = nn.Linear(pd, cfg["num_slots"])
        self.lib = PrimitiveLibrary(sd, max_macros=cfg["max_new_primitives"] * max(cfg["num_cycles"] - 1, 1))
        self.syn = RuleSynthesizer(sd * 4, sd, pd, self.lib.capacity, cfg["max_program_steps"],
                                   cfg["min_program_steps"])
        self.film = nn.Linear(sd, sd * 2) if cfg["condition"] == "film" else None
        self.fallback = None if cfg["use_program"] else nn.Sequential(
            nn.Linear(sd * 2, sd * 2), nn.GELU(), nn.Linear(sd * 2, sd))
        dec_in = sd * 2 + (sd + pd if cfg["decoder_bypass"] else 0)
        self.decoder = nn.Sequential(nn.Linear(dec_in, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(),
                                     nn.Linear(sd * 2, sd * 2), nn.LayerNorm(sd * 2), nn.GELU(),
                                     nn.Linear(sd * 2, sd))
        self.mem = ProgramMemory(cfg["memory_capacity"], cfg["memory_threshold"])

    def encode_examples(self, enc, examples, anonymous):
        rows = []
        for e in examples:
            s, r = enc.example(e.state), enc.example(e.result)
            a = torch.zeros_like(s) if anonymous else enc.action(e.action)
            rows.append(torch.cat([s, a, r, r - s]))
        return torch.stack(rows)

    def perceive(self, enc, state_text=None, image=None):
        vec, tokens = enc.encode_input(state_text) if image is None else enc.encode_image(image)
        obj, obj_attn = self.obj_ext(tokens)
        slots, slot_attn = self.slot_attn(tokens)
        return vec, obj, slots, obj_attn, slot_attn

    def rule(self, enc, examples, anonymous):
        ex = self.encode_examples(enc, examples, anonymous)
        sig = self.syn.signature(ex)
        return sig, self.syn.code(sig)

    def transform(self, p, sig, code, temp, use_mem, override=None):
        cfg = self.cfg
        if not cfg["use_program"]:
            return p + self.fallback(torch.cat([p, code])), [], torch.zeros((), device=p.device), []
        if self.film is not None:
            scale, shift = self.film(code).chunk(2)
            p = p * (1 + torch.tanh(scale)) + torch.tanh(shift)
        if override is not None:
            prog = steps_from_memory(override, self.lib.capacity, self.lib.n, p.device)
            halts = []
        else:
            remembered = self.mem.lookup(sig) if (use_mem and not self.training) else None
            if remembered is not None:
                prog = steps_from_memory(remembered, self.lib.capacity, self.lib.n, p.device)
                halts = []
            else:
                prog, halts = self.syn(sig, p, self.lib.n, temp)
        out, steps, length = execute_program(self.lib, prog, halts, p, code, cfg["min_program_steps"],
                                             self.training)
        return out, steps, length, prog

    def decode(self, obj, p, enc, action, sig):
        parts = [obj, p]
        if self.cfg["decoder_bypass"]:
            parts += [enc.action(action), sig]
        return self.decoder(torch.cat(parts))

    def forward(self, enc, ep, temp=0.8, use_mem=True, overrides=None, image=None):
        _, obj, slots, _, _ = self.perceive(enc, ep["state"], image)
        p, info = None, {"steps": [], "sigs": [], "length": [], "progs": [], "codes": []}
        sig = None
        for k, examples in enumerate(ep["per_step"]):
            sig, code = self.rule(enc, examples, ep["anonymous"])
            if p is None:
                sw = F.softmax(self.slot_select(sig), dim=-1)
                p = (sw.unsqueeze(1) * slots).sum(0)
            override = overrides[k] if overrides is not None else None
            if override is not None and isinstance(override, tuple):
                code = override[1]
                override = override[0]
            p, steps, length, prog = self.transform(p, sig, code, temp, use_mem, override)
            info["steps"].append(steps)
            info["sigs"].append(sig)
            info["codes"].append(code)
            info["length"].append(length)
            info["progs"].append(prog)
        pred = self.decode(obj, p, enc, ep["actions"][-1], sig)
        return pred, info

    def remember(self, info, label, ok):
        for sig, steps in zip(info["sigs"], info["steps"]):
            if steps:
                self.mem.store(sig, steps, label, ok)

    def compress(self):
        created = compress_library(self.lib, self.mem, self.cfg["compression_threshold"],
                                   self.cfg["max_new_primitives"])
        return created


def train_cycle(model, enc, world, cfg, n_iters, lr, cycle, rng, log_every=500):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": lr}]
    enc_params = enc.trainable_parameters()
    if enc_params:
        groups.append({"params": enc_params, "lr": lr * 0.1})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sch = make_schedule(opt, max(n_iters // cfg["grad_accumulation"], 1), warmup=20)
    model.train()
    enc.train()
    cand, index = enc.candidates(world.train_states)
    ok = tot = 0
    t0 = time.time()
    for it in range(n_iters):
        ep = world.sample_training_episode(rng, cfg)
        pred, info = model(enc, ep, cfg["temperature"], use_mem=False)
        target = enc.target(ep["expected"])
        logits = candidate_logits(pred, cand, cfg["tau"])
        tgt = torch.tensor([index[ep["expected"]]], device=pred.device)
        ce = F.cross_entropy(logits.unsqueeze(0), tgt)
        latent = 1.0 - F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).squeeze()
        loss = ce + cfg["latent_weight"] * latent
        progs = [w for prog in info["progs"] for w in prog]
        if progs:
            loss = loss + cfg["diversity_weight"] * diversity_loss(progs)
            loss = loss + cfg["novelty_weight"] * novelty_loss(progs, len(BASE_NAMES))
            loss = loss + cfg["ponder_weight"] * sum(info["length"]) / cfg["max_program_steps"]
        (loss / cfg["grad_accumulation"]).backward()
        correct = int(logits.argmax().item()) == index[ep["expected"]]
        ok += correct
        tot += 1
        with torch.no_grad():
            model.remember(info, ep["actions"][0], correct)
        if (it + 1) % cfg["grad_accumulation"] == 0:
            nn.utils.clip_grad_norm_(params + enc_params, 1.0)
            opt.step()
            opt.zero_grad()
            sch.step()
            if enc.after_step():
                cand, index = enc.candidates(world.train_states)
        if it % log_every == 0:
            speed = (it + 1) / max(time.time() - t0, 1e-6)
            print(f"  [C{cycle}] {it:6d}/{n_iters} ce={ce.item():.3f} lat={latent.item():.3f} "
                  f"train_acc={100 * ok / max(tot, 1):5.1f}% {speed:.1f}it/s")
            ok = tot = 0
    enc.refresh()


def predict(model, enc, ep, cand, states, overrides=None, image=None, use_mem=True):
    pred, info = model(enc, ep, 0.1, use_mem=use_mem, overrides=overrides, image=image)
    logits = candidate_logits(pred, cand, model.cfg["tau"])
    return states[int(logits.argmax().item())], pred, info


def accuracy_table(rows):
    out = {}
    for key, values in rows.items():
        out[key] = round(100.0 * sum(values) / max(len(values), 1), 2)
    return out


def evaluate_single_step(model, enc, world, cfg, cand, states):
    report = {}
    for split in ["train", "heldout"]:
        for policy in POLICIES:
            eps = world.eval_episodes(split, policy, cfg["num_examples"], cfg["eval_seed"])
            overall = defaultdict(list)
            per_action = defaultdict(lambda: defaultdict(list))
            for ep in eps:
                pred, _, _ = predict(model, enc, ep, cand, states)
                overall["npr"].append(pred == ep["expected"])
                per_action[ep["actions"][0]]["npr"].append(pred == ep["expected"])
                for name, fn in BASELINES.items():
                    hit = fn(world, ep) == ep["expected"]
                    overall[name].append(hit)
                    per_action[ep["actions"][0]][name].append(hit)
            report[f"{split}/{policy}"] = {
                "n": len(eps),
                "overall": accuracy_table(overall),
                "per_action": {a: accuracy_table(v) for a, v in sorted(per_action.items())},
            }
    return report


def evaluate_multistep(model, enc, world, cfg, cand, states):
    rng = random.Random(cfg["eval_seed"])
    report = {}
    for depth in [2, 3]:
        chains = world.all_chains(depth, world.held_out, rng)
        for policy in ["same_transition", "same_action"]:
            hits = defaultdict(list)
            for chain in chains:
                ep = world.episode(chain, policy, cfg["num_examples"], rng)
                pred, _, _ = predict(model, enc, ep, cand, states)
                hits["npr"].append(pred == ep["expected"])
                for name, fn in BASELINES.items():
                    hits[name].append(fn(world, ep) == ep["expected"])
            report[f"depth{depth}/{policy}"] = {"n": len(chains), "overall": accuracy_table(hits)}
    return report


def evaluate_temporal(model, enc, world, cfg, cand, states):
    rng = random.Random(cfg["eval_seed"])
    report = {}
    for policy in POLICIES:
        by_transition = defaultdict(list)
        for t in world.test:
            if t.action != "wait":
                continue
            ep = world.episode([t], policy, cfg["num_examples"], rng)
            pred, _, _ = predict(model, enc, ep, cand, states)
            by_transition[f"{t.prop}:{t.src}->{t.dst}"].append(pred == ep["expected"])
        report[f"wait/{policy}"] = accuracy_table(by_transition)
    rollout = defaultdict(list)
    stepwise = defaultdict(list)
    for kind, chain in world.temporal_chains(world.held_out):
        ep = world.episode(chain, "same_transition", cfg["num_examples"], rng)
        pred, _, _ = predict(model, enc, ep, cand, states)
        rollout[kind].append(pred == ep["expected"])
        ok = True
        for t in chain:
            sub = world.episode([t], "same_transition", cfg["num_examples"], rng)
            p1, _, _ = predict(model, enc, sub, cand, states)
            ok = ok and p1 == sub["expected"]
        stepwise[kind].append(ok)
    report["chains/rollout"] = accuracy_table(rollout)
    report["chains/stepwise"] = accuracy_table(stepwise)
    return report


def action_examples(world, action, n, rng):
    pool = world.train_by_action[action]
    by_src = defaultdict(list)
    for t in pool:
        by_src[(t.prop, t.src)].append(t)
    picked = [rng.choice(v) for _, v in sorted(by_src.items())]
    rng.shuffle(picked)
    if len(picked) < n:
        rest = [t for t in pool if t not in picked]
        picked += rng.sample(rest, min(n - len(picked), len(rest)))
    return picked[:max(n, len(by_src))]


def plan(model, enc, world, cfg, start, goal, max_depth, cand, states, rng):
    goal_vec = enc.target(goal)
    rules = {a: model.rule(enc, action_examples(world, a, cfg["num_examples"], rng), False) for a in world.actions}
    _, obj, slots, _, _ = model.perceive(enc, start)
    beams = [(0.0, [], None)]
    for _ in range(max_depth):
        candidates = []
        for _, acts, p in beams:
            for a in world.actions:
                sig, code = rules[a]
                p0 = p
                if p0 is None:
                    sw = F.softmax(model.slot_select(sig), dim=-1)
                    p0 = (sw.unsqueeze(1) * slots).sum(0)
                p1, _, _, _ = model.transform(p0, sig, code, 0.1, True)
                pred = model.decode(obj, p1, enc, a, sig)
                score = F.cosine_similarity(pred.unsqueeze(0), goal_vec.unsqueeze(0)).item()
                decoded = states[int(candidate_logits(pred, cand, cfg["tau"]).argmax().item())]
                candidates.append((score, acts + [a], p1, decoded))
        candidates.sort(key=lambda c: -c[0])
        for score, acts, _, decoded in candidates:
            if decoded == goal:
                return acts
        beams = [(s, a, p) for s, a, p, _ in candidates[:cfg["beam_width"]]]
    return beams[0][1]


def evaluate_planning(model, enc, world, cfg, cand, states):
    rng = random.Random(cfg["eval_seed"])
    report = {}
    for depth in [1, 2, 3]:
        if depth == 1:
            chains = [[t] for t in world.test]
        else:
            chains = world.all_chains(depth, world.held_out, rng)
        success, exact, chance = [], [], []
        for chain in chains:
            start, goal = chain[0].state, chain[-1].result
            acts = plan(model, enc, world, cfg, start, goal, depth, cand, states, rng)
            success.append(world.execute(start, acts) == goal)
            exact.append(acts == [t.action for t in chain])
            trials = [world.execute(start, [rng.choice(world.actions) for _ in range(depth)]) == goal
                      for _ in range(20)]
            chance.append(sum(trials) / len(trials))
        n = max(len(chains), 1)
        report[f"depth{depth}"] = {"n": len(chains), "success": round(100.0 * sum(success) / n, 2),
                                   "same_as_reference": round(100.0 * sum(exact) / n, 2),
                                   "random_plan_success": round(100.0 * sum(chance) / n, 2)}
    return report


def evaluate_programs(model, enc, world, cfg, cand, states):
    rng = random.Random(cfg["eval_seed"])
    report = {}
    eps = world.eval_episodes("heldout", "same_transition", cfg["num_examples"], cfg["eval_seed"])
    if not cfg["use_program"]:
        return {"note": "program path disabled"}
    by_action = defaultdict(Counter)
    records = []
    for ep in eps:
        pred, _, info = predict(model, enc, ep, cand, states, use_mem=False)
        steps = tuple(info["steps"][0])
        by_action[ep["actions"][0]][steps] += 1
        records.append((ep, steps, info["codes"][0], pred == ep["expected"]))
    names = model.lib.names
    report["programs"] = {a: {" -> ".join(names[i] for i in s): n for s, n in c.most_common(3)}
                          for a, c in sorted(by_action.items())}
    top = {a: c.most_common(1)[0][0] for a, c in by_action.items()}
    report["distinct_programs_across_actions"] = len(set(top.values()))
    report["num_actions"] = len(top)
    by_donor_action = defaultdict(list)
    for r in records:
        by_donor_action[r[0]["actions"][0]].append(r)
    swapped, followed, kept = 0, 0, 0
    for ep, steps, code, _ in records:
        t = ep["transitions"][0]
        for alt in world.transitions_from(t.obj, t.prop, t.src):
            if alt.action == t.action or alt.result == t.result:
                continue
            donors = [r for r in by_donor_action[alt.action] if r[0]["obj"] != ep["obj"]]
            if not donors:
                continue
            donor = rng.choice(donors)
            pred, _, _ = predict(model, enc, ep, cand, states, overrides=[(list(donor[1]), donor[2])],
                                 use_mem=False)
            swapped += 1
            followed += pred == alt.result
            kept += pred == ep["expected"]
    report["intervention"] = {"n": swapped,
                              "follows_donor_program": round(100.0 * followed / max(swapped, 1), 2),
                              "keeps_original_answer": round(100.0 * kept / max(swapped, 1), 2)}
    vecs = torch.stack([enc.example(s) for s in world.all_states[:64]])
    report["negate_involution_error"] = model.lib.involution_error(vecs)
    return report


def evaluate_all(model, enc, world, cfg, extra=None):
    model.eval()
    enc.eval()
    with torch.no_grad():
        cand, _ = enc.candidates(world.all_states)
        states = world.all_states
        report = {
            "single_step": evaluate_single_step(model, enc, world, cfg, cand, states),
            "multistep": evaluate_multistep(model, enc, world, cfg, cand, states),
            "temporal": evaluate_temporal(model, enc, world, cfg, cand, states),
            "planning": evaluate_planning(model, enc, world, cfg, cand, states),
            "programs": evaluate_programs(model, enc, world, cfg, cand, states),
            "library": list(model.lib.names),
            "memory_entries": len(model.mem),
        }
        if extra is not None:
            report.update(extra(model, enc, world, cfg, cand, states))
    return report


def print_report(report):
    print("\n  single-step accuracy (%)")
    for key, block in report["single_step"].items():
        cells = " ".join(f"{k}={v:5.1f}" for k, v in block["overall"].items())
        print(f"    {key:28s} n={block['n']:4d} {cells}")
    print("\n  multi-step rollouts on held-out objects (%)")
    for key, block in report["multistep"].items():
        cells = " ".join(f"{k}={v:5.1f}" for k, v in block["overall"].items())
        print(f"    {key:28s} n={block['n']:4d} {cells}")
    print("\n  temporal (%)")
    for key, block in report["temporal"].items():
        print(f"    {key:28s} {block}")
    print("\n  planning on held-out objects (%)")
    for key, block in report["planning"].items():
        print(f"    {key:28s} {block}")
    prog = report["programs"]
    if "programs" in prog:
        print(f"\n  distinct top programs: {prog['distinct_programs_across_actions']}/{prog['num_actions']} actions")
        print(f"  intervention: {prog['intervention']}")
        print(f"  NEGATE involution error: {prog['negate_involution_error']:.2e}")
    for key in report:
        if key not in {"single_step", "multistep", "temporal", "planning", "programs", "library", "memory_entries"}:
            print(f"\n  {key}: {report[key]}")


def run(model, enc, world, cfg, seed, extra=None):
    rng = random.Random(seed)
    history = []
    for c in range(cfg["num_cycles"]):
        print(f"\n{'=' * 70}\nCYCLE {c + 1}/{cfg['num_cycles']} library={model.lib.names}\n{'=' * 70}")
        train_cycle(model, enc, world, cfg, cfg["iters_per_cycle"][c], cfg["lr_per_cycle"][c], c + 1, rng)
        if c < cfg["num_cycles"] - 1:
            created = model.compress()
            print(f"  compression: {created if created else 'nothing'}")
            history.append({"cycle": c + 1, "created": created})
    report = evaluate_all(model, enc, world, cfg, extra)
    report["compression"] = history
    print_report(report)
    return report


def apply_overrides(cfg, ablation, scale):
    out = dict(cfg)
    out.update(ABLATIONS[ablation])
    out["iters_per_cycle"] = [max(int(n * scale), out["grad_accumulation"]) for n in out["iters_per_cycle"]]
    out["ablation"] = ablation
    return out
