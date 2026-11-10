import argparse

import torch

from npr_core import pick_device, save_json, set_seed
from npr_world import World
from npr_world_model import ABLATIONS, WORLD_CONFIG, WorldModel, apply_overrides, run

DEVICE = pick_device()


class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))


def load_perceiver(kind, seed):
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
    if kind == "gpt2":
        return GPT2Model.from_pretrained("gpt2").to(DEVICE).eval(), GPT2Tokenizer.from_pretrained("gpt2")
    torch.manual_seed(seed)
    model = GPT2Model(GPT2Config(vocab_size=256, n_positions=128, bos_token_id=0, eos_token_id=0)).to(DEVICE).eval()
    return model, ByteTokenizer()


class TextEncoder:
    def __init__(self, gpt2, tokenizer, world, layer):
        self.gpt2, self.tokenizer, self.layer = gpt2, tokenizer, layer
        self.last, self.tokens = {}, {}
        phrases = set(world.all_states) | {f"__action__{a}" for a in world.actions}
        with torch.no_grad():
            for p in sorted(phrases):
                self._encode(p)
        train = torch.stack([self.last[s] for s in world.train_states])
        self.center = train.mean(0)
        self.scale = (train - self.center).norm(dim=-1).mean().clamp_min(1e-6)
        acts = torch.stack([self.last[f"__action__{a}"] for a in world.actions])
        self.action_center = acts.mean(0)
        self.action_scale = (acts - self.action_center).norm(dim=-1).mean().clamp_min(1e-6)
        self.dim = train.shape[1]

    def _encode(self, phrase):
        if phrase in self.last:
            return
        text = " " + phrase.replace("__action__", "")
        ids = self.tokenizer.encode(text)
        out = self.gpt2(torch.tensor([ids], device=DEVICE), output_hidden_states=True)
        h = out.hidden_states[self.layer][0]
        self.last[phrase] = h[-1].clone()
        self.tokens[phrase] = h.clone()

    def encode_input(self, text):
        self._encode(text)
        return self.last[text], self.tokens[text]

    def encode_image(self, image):
        raise NotImplementedError

    def example(self, text):
        self._encode(text)
        return (self.last[text] - self.center) / self.scale

    def target(self, text):
        return self.example(text)

    def action(self, name):
        key = f"__action__{name}"
        self._encode(key)
        return (self.last[key] - self.action_center) / self.action_scale

    def candidates(self, states):
        return torch.stack([self.target(s) for s in states]), {s: i for i, s in enumerate(states)}

    def trainable_parameters(self):
        return []

    def after_step(self):
        return False

    def refresh(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="none")
    parser.add_argument("--perceiver", choices=["gpt2", "random"], default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = apply_overrides(WORLD_CONFIG, args.ablation, args.scale)
    world = World()
    print(f"device={DEVICE} seed={args.seed} ablation={args.ablation} perceiver={args.perceiver}")
    print(world.summary())
    gpt2, tokenizer = load_perceiver(args.perceiver, args.seed)
    enc = TextEncoder(gpt2, tokenizer, world, args.layer)
    model = WorldModel(enc.dim, cfg).to(DEVICE)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    report = run(model, enc, world, cfg, args.seed)
    report["config"] = {k: v for k, v in cfg.items()}
    report["args"] = vars(args)
    out = args.out or f"results_text_{args.ablation}_{args.perceiver}_seed{args.seed}.json"
    save_json(report, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
