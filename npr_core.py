import json
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_NAMES = ["IDENTITY", "NEGATE", "MORPH", "ASSOCIATE", "LOOKUP", "BLEND"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


class Householder(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.v = nn.Parameter(torch.randn(sd) / math.sqrt(sd))

    def forward(self, x):
        v = self.v
        coeff = 2.0 * (x @ v) / (v @ v).clamp_min(1e-12)
        return x - coeff[..., None] * v


class SubspaceAttention(nn.Module):
    def __init__(self, sd, n_tokens=12, n_heads=4):
        super().__init__()
        if sd % n_tokens != 0:
            raise ValueError("state dim must be divisible by n_tokens")
        self.t = n_tokens
        self.d = sd // n_tokens
        self.pos = nn.Parameter(torch.randn(n_tokens, self.d) * 0.02)
        self.norm = nn.LayerNorm(self.d)
        self.attn = nn.MultiheadAttention(self.d, n_heads, batch_first=True)

    def forward(self, x):
        shape = x.shape
        z = self.norm(x.reshape(-1, self.t, self.d) + self.pos)
        a, _ = self.attn(z, z, z)
        return a.reshape(shape)


class KeyValueMemory(nn.Module):
    def __init__(self, sd, n_slots=128):
        super().__init__()
        self.q = nn.Linear(sd, sd)
        self.keys = nn.Parameter(torch.randn(n_slots, sd) / math.sqrt(sd))
        self.values = nn.Parameter(torch.randn(n_slots, sd) * 0.02)

    def forward(self, x):
        w = F.softmax(self.q(x) @ self.keys.T / math.sqrt(x.shape[-1]), dim=-1)
        return w @ self.values


def _logit(p):
    return math.log(p / (1.0 - p))


class PrimitiveLibrary(nn.Module):
    def __init__(self, sd, max_macros=4, n_tokens=12, kv_slots=128):
        super().__init__()
        self.sd = sd
        self.max_macros = max_macros
        self.names = list(BASE_NAMES)
        self.negate = Householder(sd)
        self.morph = nn.Sequential(nn.LayerNorm(sd), nn.Linear(sd, sd), nn.GELU(), nn.Linear(sd, sd))
        self.associate = SubspaceAttention(sd, n_tokens)
        self.lookup = KeyValueMemory(sd, kv_slots)
        self.blend = nn.Sequential(nn.Linear(sd * 2, sd), nn.GELU(), nn.Linear(sd, sd))
        self.gates = nn.Parameter(torch.tensor([_logit(0.3), _logit(0.4), _logit(0.5), _logit(0.3)]))
        self.macros = []
        self.macro_fix = nn.ModuleList()
        self.macro_gates = nn.ParameterList()

    @property
    def n(self):
        return len(BASE_NAMES) + len(self.macros)

    @property
    def capacity(self):
        return len(BASE_NAMES) + self.max_macros

    def _base(self, i, s, ctx):
        if i == 0:
            return s
        if i == 1:
            return self.negate(s)
        if i == 2:
            return s + torch.sigmoid(self.gates[0]) * self.morph(s)
        if i == 3:
            return s + torch.sigmoid(self.gates[1]) * self.associate(s)
        if i == 4:
            return s + torch.sigmoid(self.gates[2]) * self.lookup(s)
        return s + torch.sigmoid(self.gates[3]) * self.blend(torch.cat([s, ctx], dim=-1))

    def apply(self, i, s, ctx):
        if i < len(BASE_NAMES):
            return self._base(i, s, ctx)
        j = i - len(BASE_NAMES)
        a, b = self.macros[j]
        y = self.apply(b, self.apply(a, s, ctx), ctx)
        return y + torch.sigmoid(self.macro_gates[j]) * self.macro_fix[j](y)

    def apply_soft(self, w, s, ctx):
        if not self.training:
            return self.apply(int(w.argmax().item()), s, ctx)
        return sum(w[i] * self.apply(i, s, ctx) for i in range(self.n))

    def add_macro(self, a, b):
        if len(self.macros) >= self.max_macros:
            return None
        dev = self.negate.v.device
        fix = nn.Sequential(nn.Linear(self.sd, self.sd), nn.GELU(), nn.Linear(self.sd, self.sd)).to(dev)
        with torch.no_grad():
            fix[2].weight.zero_()
            fix[2].bias.zero_()
        self.macro_fix.append(fix)
        self.macro_gates.append(nn.Parameter(torch.tensor(0.0, device=dev)))
        self.macros.append((a, b))
        self.names.append(f"{self.names[a]}_{self.names[b]}")
        return self.n - 1

    def involution_error(self, x):
        with torch.no_grad():
            y = self.negate(self.negate(x))
            return ((y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).mean().item()


class RuleSynthesizer(nn.Module):
    def __init__(self, ex_dim, sd, pd, capacity, max_steps, min_steps):
        super().__init__()
        self.capacity = capacity
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.proj = nn.Sequential(nn.Linear(ex_dim, pd), nn.LayerNorm(pd), nn.GELU())
        layer = nn.TransformerEncoderLayer(pd, 4, pd * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.state_proj = nn.Linear(sd, pd)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(pd * 2, 128), nn.GELU(), nn.Linear(128, capacity)) for _ in range(max_steps)])
        self.step_emb = nn.Embedding(max_steps, pd)
        self.halt = nn.Sequential(nn.Linear(pd * 3, 128), nn.GELU(), nn.Linear(128, 1))
        self.code = nn.Sequential(nn.Linear(pd, sd), nn.LayerNorm(sd))

    def signature(self, ex):
        x = self.encoder(self.proj(ex).unsqueeze(0))
        return x.mean(1).squeeze(0)

    def forward(self, sig, state, n_active, temp=0.8):
        ctx = torch.cat([sig, self.state_proj(state)])
        mask = torch.full((self.capacity,), float("-inf"), device=sig.device)
        mask[:n_active] = 0.0
        prog, halts = [], []
        for i, h in enumerate(self.heads):
            lg = h(ctx) + mask
            if self.training:
                w = F.gumbel_softmax(lg, tau=temp, hard=True)
            else:
                w = F.one_hot(lg.argmax(), self.capacity).float()
            prog.append(w[:n_active])
            if i + 1 >= self.min_steps and i + 1 < self.max_steps:
                step = self.step_emb(torch.tensor(i, device=sig.device))
                halts.append(torch.sigmoid(self.halt(torch.cat([ctx, step]))).squeeze())
        return prog, halts


def execute_program(lib, prog, halts, s, ctx, min_steps, training):
    if not training:
        steps = []
        for i, w in enumerate(prog):
            s = lib.apply_soft(w, s, ctx)
            steps.append(int(w.argmax().item()))
            k = i + 1 - min_steps
            if 0 <= k < len(halts) and halts[k].item() > 0.5:
                break
        return s, steps, torch.tensor(float(len(steps)), device=s.device)
    outs = []
    for w in prog:
        s = lib.apply_soft(w, s, ctx)
        outs.append(s)
    weights = []
    remaining = torch.ones((), device=s.device)
    for t in range(len(prog)):
        k = t + 1 - min_steps
        if k < 0:
            weights.append(torch.zeros((), device=s.device))
        elif k < len(halts):
            weights.append(remaining * halts[k])
            remaining = remaining * (1.0 - halts[k])
        else:
            weights.append(remaining)
    out = sum(w * o for w, o in zip(weights, outs))
    expected_len = sum(w * (t + 1) for t, w in enumerate(weights))
    steps = [int(w.argmax().item()) for w in prog[:max(min_steps, int(round(expected_len.item())))]]
    return out, steps, expected_len


class ProgramMemory:
    def __init__(self, capacity=300, threshold=0.9):
        self.capacity = capacity
        self.threshold = threshold
        self.entries = []

    def store(self, sig, steps, label, ok):
        if not ok:
            return
        self.entries.append({"sig": sig.detach().clone(), "steps": list(steps), "label": label, "count": 1})
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda e: -e["count"])
            self.entries = self.entries[:self.capacity]

    def lookup(self, sig):
        if not self.entries:
            return None
        sigs = torch.stack([e["sig"] for e in self.entries])
        sims = F.cosine_similarity(sigs, sig.detach().unsqueeze(0), dim=-1)
        best = int(sims.argmax().item())
        if sims[best].item() < self.threshold:
            return None
        self.entries[best]["count"] += 1
        return self.entries[best]["steps"]

    def frequent_pairs(self, min_count):
        c = Counter()
        for e in self.entries:
            for a, b in zip(e["steps"], e["steps"][1:]):
                if a != 0 and b != 0:
                    c[(a, b)] += 1
        return [(p, n) for p, n in c.most_common() if n >= min_count]

    def clear(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)


def compress_library(lib, memory, min_count, max_new):
    created = []
    for (a, b), n in memory.frequent_pairs(min_count):
        if len(created) >= max_new:
            break
        name = f"{lib.names[a]}_{lib.names[b]}"
        if name in lib.names:
            continue
        idx = lib.add_macro(a, b)
        if idx is None:
            break
        created.append((name, n))
    if created:
        memory.clear()
    return created


def steps_from_memory(steps, capacity, n_active, device):
    return [F.one_hot(torch.tensor(i, device=device), capacity).float()[:n_active] for i in steps]


def diversity_loss(prog):
    if len(prog) < 2:
        return prog[0].sum() * 0.0
    p = torch.stack(prog)
    sim = p @ p.T
    n = p.shape[0]
    off = sim.sum() - sim.diag().sum()
    return off / (n * (n - 1))


def novelty_loss(prog, n_base):
    p = torch.stack(prog)
    if p.shape[1] <= n_base:
        return torch.zeros((), device=p.device)
    return F.relu(0.15 - p.mean(0)[n_base:].sum())


def candidate_logits(pred, cand, tau):
    return F.cosine_similarity(pred.unsqueeze(0), cand, dim=-1) / tau


def make_schedule(opt, n_iters, warmup=200):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(n_iters - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)
