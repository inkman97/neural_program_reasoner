import argparse
import copy
import math
import random
from collections import defaultdict

import torch
from PIL import Image, ImageDraw

from npr_core import pick_device, save_json, set_seed
from npr_world import OBJECTS, World, display_name, render, step
from npr_world_model import ABLATIONS, WORLD_CONFIG, WorldModel, apply_overrides, candidate_logits, run

DEVICE = pick_device()
BACKGROUND = (245, 245, 245)
FLOOR = (200, 190, 170)


def _hash(name):
    return sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 1000


def _seed_for(*parts):
    return sum(_hash(str(p)) * (i + 7) for i, p in enumerate(parts))


def _draw_container(draw, cx, cy, sz, name):
    w, h = sz, int(sz * 1.2)
    draw.polygon([(cx - w // 2, cy - h // 2), (cx + w // 2, cy - h // 2), (cx + w // 2 + 8, cy + h // 2),
                  (cx - w // 2 - 8, cy + h // 2)], fill=(180, 140, 100), outline=(80, 60, 40), width=2)
    if name in ["cup", "mug", "jug"]:
        draw.arc([cx + w // 2, cy - h // 4, cx + w // 2 + 20, cy + h // 4], 270, 90, fill=(80, 60, 40), width=3)


def _draw_door(draw, cx, cy, sz, is_open):
    w, h = int(sz * 0.8), int(sz * 1.5)
    if is_open:
        draw.polygon([(cx - w // 2, cy - h // 2), (cx, cy - h // 2 + 10), (cx, cy + h // 2 - 10),
                      (cx - w // 2, cy + h // 2)], fill=(160, 120, 80), outline=(80, 60, 40), width=2)
        draw.rectangle([cx + 2, cy - h // 2 + 12, cx + w // 2, cy + h // 2 - 12], fill=(100, 140, 100))
    else:
        draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(160, 120, 80),
                       outline=(80, 60, 40), width=2)
        draw.ellipse([cx + w // 4, cy - 4, cx + w // 4 + 8, cy + 4], fill=(200, 180, 50))


def _draw_box(draw, cx, cy, sz, is_open, h):
    w_, h_ = sz, int(sz * 0.8)
    body = ((h * 31) % 100 + 120, (h * 61) % 80 + 100, (h * 97) % 60 + 80)
    draw.rectangle([cx - w_ // 2, cy - h_ // 4, cx + w_ // 2, cy + h_ // 2], fill=body, outline=(100, 80, 60), width=2)
    lid = (body[0] - 20, body[1] - 20, body[2] - 20)
    if is_open:
        draw.polygon([(cx - w_ // 2, cy - h_ // 4), (cx - w_ // 2 + 10, cy - h_ // 4 - 20),
                      (cx + w_ // 2 + 10, cy - h_ // 4 - 20), (cx + w_ // 2, cy - h_ // 4)], fill=lid,
                     outline=(100, 80, 60), width=2)
    else:
        draw.rectangle([cx - w_ // 2 - 3, cy - h_ // 4 - 5, cx + w_ // 2 + 3, cy - h_ // 4], fill=lid,
                       outline=(100, 80, 60), width=2)


def _draw_device(draw, cx, cy, sz, is_on):
    w, h = sz, int(sz * 0.8)
    draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(60, 60, 70), outline=(30, 30, 40),
                   width=2)
    if is_on:
        draw.rectangle([cx - w // 2 + 6, cy - h // 2 + 6, cx + w // 2 - 6, cy + h // 2 - 6], fill=(100, 200, 255))
        for angle in range(0, 360, 45):
            r = math.radians(angle)
            draw.line([(cx + int((w // 2 + 3) * math.cos(r)), cy + int((h // 2 + 3) * math.sin(r))),
                       (cx + int((w // 2 + 15) * math.cos(r)), cy + int((h // 2 + 15) * math.sin(r)))],
                      fill=(255, 255, 100), width=1)
    else:
        draw.rectangle([cx - w // 2 + 6, cy - h // 2 + 6, cx + w // 2 - 6, cy + h // 2 - 6], fill=(30, 30, 35))


def _draw_table(draw, cx, cy, sz):
    w = sz + 40
    sy = cy + sz // 2 + 5
    draw.rectangle([cx - w // 2, sy, cx + w // 2, sy + 8], fill=(139, 90, 43), outline=(80, 50, 20))
    draw.rectangle([cx - w // 2 + 5, sy + 8, cx - w // 2 + 10, sy + 35], fill=(120, 80, 40))
    draw.rectangle([cx + w // 2 - 10, sy + 8, cx + w // 2 - 5, sy + 35], fill=(120, 80, 40))


def _draw_cracks(draw, cx, cy, sz, rnd):
    pts = [(cx - sz // 2, cy)]
    for i in range(6):
        pts.append((cx - sz // 2 + (i + 1) * sz // 6, cy + rnd.randint(-sz // 4, sz // 4)))
    for p1, p2 in zip(pts, pts[1:]):
        draw.line([p1, p2], fill=(0, 0, 0), width=3)
    for _ in range(4):
        px = cx + rnd.randint(-sz // 2 - 10, sz // 2 + 10)
        py = cy + rnd.randint(sz // 4, sz // 2 + 15)
        ps = rnd.randint(3, 8)
        draw.polygon([(px, py - ps), (px + ps, py), (px, py + ps), (px - ps, py)], fill=(180, 150, 120),
                     outline=(100, 80, 60))


LIQUID = {"cold": (100, 150, 255), "warm": (255, 200, 100), "hot": (255, 80, 50), "boiling": (200, 50, 200)}


def _draw_temperature_effects(draw, cx, cy, sz, value, rnd):
    if value == "cold":
        for _ in range(6):
            sx = cx + rnd.randint(-sz // 2 - 10, sz // 2 + 10)
            sy = cy + rnd.randint(-sz // 2 - 15, sz // 2 + 15)
            for angle in range(0, 360, 60):
                r = math.radians(angle)
                draw.line([(sx, sy), (sx + int(6 * math.cos(r)), sy + int(6 * math.sin(r)))], fill=(150, 200, 255),
                          width=1)
    elif value == "warm":
        for i in range(3):
            wy = cy - sz // 2 - 10 - i * 10
            pts = [(x, wy + int(3 * math.sin(x / 4))) for x in range(cx - 20, cx + 21, 2)]
            draw.line(pts, fill=(255, 180, 100), width=2)
    elif value == "hot":
        for i in range(4):
            sx = cx - 15 + i * 10
            for j in range(3):
                sy = cy - sz // 2 - 8 - j * 12
                draw.line([(sx, sy), (sx + 3, sy - 8)], fill=(255, 100, 50), width=2)
                draw.line([(sx + 3, sy - 8), (sx - 2, sy - 14)], fill=(255, 80, 30), width=2)
    elif value == "boiling":
        for _ in range(10):
            bx = cx + rnd.randint(-sz // 3, sz // 3)
            by = cy + rnd.randint(-sz // 3, sz // 4)
            br = rnd.randint(4, 10)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 255, 255), outline=(200, 0, 200), width=2)
        for i in range(5):
            sx = cx - 20 + i * 10
            for j in range(4):
                sy = cy - sz // 2 - 5 - j * 10
                draw.line([(sx, sy), (sx + rnd.randint(-4, 4), sy - 8)], fill=(200, 0, 200), width=2)


def _draw_temperature(draw, cx, cy, sz, name, h, liquid, effects, rnd):
    _draw_container(draw, cx, cy, sz, name)
    _draw_temperature_effects(draw, cx, cy, sz, effects, rnd)
    lw, lh = sz - 10, sz // 2
    draw.rectangle([cx - lw // 2, cy - lh // 4, cx + lw // 2, cy + lh], fill=LIQUID[liquid])
    label = ((h * 47) % 150 + 50, (h * 83) % 150 + 50, (h * 127) % 150 + 50)
    draw.rectangle([cx - lw // 2, cy + lh - 6, cx + lw // 2, cy + lh], fill=label)


def _draw_battery(draw, cx, cy, sz, level):
    w, h = int(sz * 0.6), int(sz * 1.0)
    draw.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=(50, 50, 60), outline=(30, 30, 40),
                   width=2)
    draw.rectangle([cx - w // 2 + 4, cy - h // 2 + 4, cx + w // 2 - 4, cy + h // 2 - 15], fill=(40, 40, 50))
    bar_w, bar_h, bar_y = w - 12, 8, cy + h // 2 - 12
    draw.rectangle([cx - bar_w // 2, bar_y, cx + bar_w // 2, bar_y + bar_h], fill=(60, 60, 60),
                   outline=(100, 100, 100))
    levels = {"empty": (0.0, (200, 50, 50)), "charging_1": (0.33, (255, 200, 50)),
              "charging_2": (0.66, (200, 220, 50)), "half": (0.5, (255, 180, 50)), "full": (1.0, (50, 200, 50))}
    pct, color = levels.get(level, (0.0, (100, 100, 100)))
    if pct > 0:
        draw.rectangle([cx - bar_w // 2, bar_y, cx - bar_w // 2 + int(bar_w * pct), bar_y + bar_h], fill=color)
    if "charging" in level:
        bx, by = cx, cy - 5
        draw.polygon([(bx - 3, by - 10), (bx + 5, by - 10), (bx + 1, by - 2), (bx + 6, by - 2), (bx - 2, by + 10),
                      (bx, by + 2), (bx - 5, by + 2)], fill=(255, 220, 50))


def _draw_food(draw, cx, cy, sz, state, rnd):
    w = int(sz * 0.8)
    colors = {"raw": ((240, 210, 180), (200, 170, 130)), "cooking_1": ((240, 190, 140), (200, 150, 100)),
              "cooking_2": ((220, 160, 100), (180, 120, 70)), "cooked": ((180, 130, 80), (140, 100, 60)),
              "burnt": ((60, 40, 30), (30, 20, 15))}
    fill, outline = colors.get(state, ((200, 200, 200), (150, 150, 150)))
    draw.rounded_rectangle([cx - w // 2, cy - w // 3, cx + w // 2, cy + w // 3], radius=8, fill=fill,
                           outline=outline, width=2)
    if state == "cooking_1":
        for i in range(2):
            sx = cx - 5 + i * 10
            draw.line([(sx, cy - w // 3 - 5), (sx + 2, cy - w // 3 - 11)], fill=(255, 150, 50), width=2)
    if state == "cooking_2":
        for i in range(4):
            sx = cx - 15 + i * 10
            for j in range(3):
                sy = cy - w // 3 - 5 - j * 8
                draw.line([(sx, sy), (sx + 2, sy - 6)], fill=(255, 120, 40), width=2)
    if state == "burnt":
        for i in range(4):
            sx = cx - 12 + i * 8
            sy = cy - w // 3 - 5 - rnd.randint(0, 8)
            draw.line([(sx, sy), (sx + 1, sy - 5)], fill=(100, 100, 100), width=1)


def _draw_plant(draw, cx, cy, sz, state):
    pw, ph = int(sz * 0.6), int(sz * 0.3)
    draw.polygon([(cx - pw // 2, cy + 5), (cx + pw // 2, cy + 5), (cx + pw // 2 - 5, cy + 5 + ph),
                  (cx - pw // 2 + 5, cy + 5 + ph)], fill=(180, 100, 60), outline=(120, 70, 40), width=2)
    draw.ellipse([cx - pw // 2 + 3, cy, cx + pw // 2 - 3, cy + 10], fill=(100, 70, 40))
    if state == "seed":
        draw.ellipse([cx - 3, cy + 2, cx + 3, cy + 8], fill=(140, 100, 50))
    elif state == "growing_1":
        draw.line([(cx, cy + 3), (cx, cy - 10)], fill=(80, 160, 80), width=2)
        draw.ellipse([cx - 4, cy - 14, cx + 4, cy - 8], fill=(100, 180, 100))
    elif state == "growing_2":
        draw.line([(cx, cy + 3), (cx, cy - 20)], fill=(60, 140, 60), width=3)
        draw.ellipse([cx - 8, cy - 24, cx, cy - 16], fill=(80, 170, 80))
        draw.ellipse([cx, cy - 20, cx + 8, cy - 12], fill=(80, 170, 80))
    elif state == "grown":
        draw.line([(cx, cy + 3), (cx, cy - 28)], fill=(50, 120, 50), width=3)
        for i in range(3):
            lx, ly = cx + (-1) ** i * 10, cy - 10 - i * 8
            draw.ellipse([lx - 6, ly - 4, lx + 6, ly + 4], fill=(60, 150, 60))
        draw.ellipse([cx - 5, cy - 32, cx + 5, cy - 24], fill=(220, 50, 50))


def _draw_flame(draw, cx, cy, sz, state, rnd):
    bw, bh = int(sz * 0.3), int(sz * 0.5)
    draw.rectangle([cx - bw // 2, cy, cx + bw // 2, cy + bh], fill=(200, 180, 140), outline=(150, 130, 100), width=2)
    draw.line([(cx, cy), (cx, cy - 5)], fill=(50, 50, 50), width=2)
    if state == "burning":
        draw.polygon([(cx, cy - 25), (cx - 10, cy - 5), (cx + 10, cy - 5)], fill=(255, 150, 30))
        draw.polygon([(cx, cy - 20), (cx - 6, cy - 5), (cx + 6, cy - 5)], fill=(255, 220, 80))
        draw.polygon([(cx, cy - 15), (cx - 3, cy - 5), (cx + 3, cy - 5)], fill=(255, 255, 180))
    elif state == "dim":
        draw.polygon([(cx, cy - 12), (cx - 4, cy - 5), (cx + 4, cy - 5)], fill=(255, 180, 80))
        draw.polygon([(cx, cy - 10), (cx - 2, cy - 5), (cx + 2, cy - 5)], fill=(255, 220, 150))
    elif state == "extinguished":
        for i in range(3):
            sx, sy = cx + rnd.randint(-5, 5), cy - 8 - i * 6
            draw.line([(sx, sy), (sx + 2, sy - 5)], fill=(150, 150, 150), width=1)


def _draw_shape(draw, cx, cy, r, shape, color):
    if shape == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2)
    elif shape == 1:
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2)
    elif shape == 2:
        draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=color, outline=(0, 0, 0), width=2)
    else:
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color, outline=(0, 0, 0),
                     width=2)


def _draw_identity_marker(draw, key):
    h = _hash(key)
    color = ((h * 37) % 200 + 55, (h * 73) % 200 + 55, (h * 113) % 200 + 55)
    _draw_shape(draw, 15, 15, 8, h % 4, color)
    dots = ((h * 53) % 200 + 55, (h * 97) % 200 + 55, (h * 137) % 200 + 55)
    for i in range(h % 3 + 1):
        draw.ellipse([35 + i * 10 - 3, 12, 35 + i * 10 + 3, 18], fill=dots)


def render_state(key, prop, value, size=224, liquid=None, effects=None):
    rnd = random.Random(_seed_for(key, prop, value, liquid, effects))
    img = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(img)
    floor_y = int(size * 0.75)
    draw.rectangle([0, floor_y, size, size], fill=FLOOR)
    draw.line([(0, floor_y), (size, floor_y)], fill=(150, 140, 120), width=2)
    name, spec, h = display_name(key), OBJECTS[key], _hash(key)
    cx, cy, sz = size // 2, floor_y - 45, 50
    if prop == "temperature":
        _draw_temperature(draw, cx, cy, sz, name, h, liquid or value, effects or value, rnd)
    elif prop == "openness":
        if name in ["door", "gate", "cabinet"]:
            _draw_door(draw, cx, cy, sz, value == "open")
        else:
            _draw_box(draw, cx, cy, sz, value == "open", h)
    elif prop == "power":
        _draw_device(draw, cx, cy, sz, value == "on")
    elif prop == "fullness":
        _draw_container(draw, cx, cy, sz, name)
        if value == "full":
            lw = sz - 10
            draw.rectangle([cx - lw // 2, cy - sz // 4, cx + lw // 2, cy + sz // 2], fill=(80, 150, 230))
    elif prop == "integrity":
        color = ((h * 41) % 120 + 100, (h * 73) % 100 + 80, (h * 109) % 80 + 80)
        draw.ellipse([cx - sz // 2, cy - sz // 4, cx + sz // 2, cy + sz // 4], fill=color,
                     outline=tuple(max(0, c - 50) for c in color), width=2)
        if value == "broken":
            _draw_cracks(draw, cx, cy, sz, rnd)
    elif prop == "position":
        color = ((h * 37) % 150 + 80, (h * 67) % 150 + 80, (h * 103) % 150 + 80)
        y = floor_y - 15 if value == "floor" else floor_y - 65
        _draw_shape(draw, cx, y, sz // 3, h % 4, color)
        if value == "surface":
            _draw_table(draw, cx, y - 10, sz)
    elif prop == "containment":
        color = ((h * 43) % 150 + 80, (h * 79) % 150 + 80, (h * 113) % 150 + 80)
        ch = _hash(spec["container"])
        box = ((ch * 31) % 100 + 120, (ch * 67) % 80 + 100, (ch * 97) % 60 + 90)
        if value == "outside":
            _draw_shape(draw, cx - 50, cy, 14, h % 4, color)
            draw.rectangle([cx + 10, cy - sz // 2, cx + 80, cy + sz // 2], fill=box, outline=(80, 60, 40), width=2)
        else:
            draw.rectangle([cx - 35, cy - sz // 2, cx + 35, cy + sz // 2], fill=box, outline=(80, 60, 40), width=2)
            _draw_shape(draw, cx, cy, 11, h % 4, color)
    elif prop == "battery":
        _draw_battery(draw, cx, cy, sz, value)
    elif prop == "cooking_state":
        _draw_food(draw, cx, cy, sz, value, rnd)
    elif prop == "growth":
        _draw_plant(draw, cx, cy, sz, value)
    elif prop == "burn_state":
        _draw_flame(draw, cx, cy, sz, value, rnd)
    _draw_identity_marker(draw, key)
    return img


def render_action(action, size=224):
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    h = _hash(action)
    color = ((h * 37) % 200 + 55, (h * 73) % 200 + 55, (h * 113) % 200 + 55)
    cx, cy = size // 2, size // 2
    draw.polygon([(cx - 50, cy - 20), (cx + 30, cy - 20), (cx + 30, cy - 40), (cx + 70, cy), (cx + 30, cy + 40),
                  (cx + 30, cy + 20), (cx - 50, cy + 20)], fill=color, outline=(0, 0, 0), width=2)
    _draw_shape(draw, cx - 20, cy + 60, 8, h % 4, color)
    return img


def load_perceiver(kind, seed):
    from transformers import ViTConfig, ViTImageProcessor, ViTModel
    if kind == "vit":
        name = "google/vit-base-patch16-224"
        return ViTModel.from_pretrained(name).to(DEVICE), ViTImageProcessor.from_pretrained(name)
    torch.manual_seed(seed)
    return ViTModel(ViTConfig()).to(DEVICE), ViTImageProcessor()


class ImageEncoder:
    def __init__(self, vit, processor, world, trainable_layers, ema_decay, refresh_every):
        self.world, self.processor = world, processor
        self.online = vit
        for p in self.online.parameters():
            p.requires_grad = False
        n_layers = len(self.online.encoder.layer)
        keep = {f"encoder.layer.{i}." for i in range(n_layers - trainable_layers, n_layers)}
        for name, p in self.online.named_parameters():
            if any(name.startswith(k) for k in keep) or name.startswith("layernorm"):
                p.requires_grad = True
        self.target_net = copy.deepcopy(self.online).eval()
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.decay, self.refresh_every, self.steps = ema_decay, refresh_every, 0
        self.pixels = {}
        for s in world.all_states:
            self.pixels[s] = self._pixels(render_state(*world.state_info[s]))
        for a in world.actions:
            self.pixels[f"__action__{a}"] = self._pixels(render_action(a))
        self.dim = self.online.config.hidden_size
        self.refresh()

    def _pixels(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)

    def _forward(self, net, pixels):
        h = net(pixel_values=pixels).last_hidden_state[0]
        return h[0], h[1:]

    def refresh(self):
        with torch.no_grad():
            raw = {k: self._forward(self.target_net, v)[0] for k, v in self.pixels.items()}
        train = torch.stack([raw[s] for s in self.world.train_states])
        self.center = train.mean(0)
        self.scale = (train - self.center).norm(dim=-1).mean().clamp_min(1e-6)
        acts = torch.stack([raw[f"__action__{a}"] for a in self.world.actions])
        a_center = acts.mean(0)
        a_scale = (acts - a_center).norm(dim=-1).mean().clamp_min(1e-6)
        self.cache = {s: (raw[s] - self.center) / self.scale for s in self.world.all_states}
        self.action_cache = {a: (raw[f"__action__{a}"] - a_center) / a_scale for a in self.world.actions}

    def encode_input(self, text):
        return self._forward(self.online, self.pixels[text])

    def encode_image(self, image):
        return self._forward(self.online, self._pixels(image))

    def example(self, text):
        return self.cache[text]

    def target(self, text):
        return self.cache[text]

    def action(self, name):
        return self.action_cache[name]

    def candidates(self, states):
        return torch.stack([self.cache[s] for s in states]), {s: i for i, s in enumerate(states)}

    def trainable_parameters(self):
        return [p for p in self.online.parameters() if p.requires_grad]

    def after_step(self):
        with torch.no_grad():
            for o, t in zip(self.online.parameters(), self.target_net.parameters()):
                t.mul_(self.decay).add_(o.detach(), alpha=1 - self.decay)
        self.steps += 1
        if self.steps % self.refresh_every == 0:
            self.refresh()
            return True
        return False

    def train(self):
        self.online.train()

    def eval(self):
        self.online.eval()


def cue_conflict(model, enc, world, cfg, cand, states):
    rng = random.Random(cfg["eval_seed"])
    values = ["cold", "warm", "hot", "boiling"]
    counts = defaultdict(lambda: defaultdict(int))
    for key in sorted(world.held_out | {t.obj for t in world.train}):
        if not OBJECTS[key].get("temperature"):
            continue
        split = "heldout" if key in world.held_out else "train"
        for liquid in values:
            for effects in values:
                if liquid == effects:
                    continue
                for action in ["heat", "cool"]:
                    by_liquid = step(key, "temperature", liquid, action)
                    by_effects = step(key, "temperature", effects, action)
                    if by_liquid is None or by_effects is None or by_liquid == by_effects:
                        continue
                    t = next(x for x in world.transitions_from(key, "temperature", liquid) if x.action == action)
                    ep = world.episode([t], "same_action", cfg["num_examples"], rng)
                    img = render_state(key, "temperature", liquid, liquid=liquid, effects=effects)
                    pred, _ = model(enc, ep, 0.1, use_mem=False, image=img)
                    out = states[int(candidate_logits(pred, cand, cfg["tau"]).argmax().item())]
                    c = counts[split]
                    c["n"] += 1
                    c["follows_liquid"] += out == world.execute(t.state, [action])
                    c["follows_effects"] += out == world.execute(render(key, "temperature", effects), [action])
    report = {}
    for split, c in counts.items():
        n = max(c["n"], 1)
        report[split] = {"n": c["n"], "follows_liquid": round(100.0 * c["follows_liquid"] / n, 2),
                         "follows_effects": round(100.0 * c["follows_effects"] / n, 2)}
    return {"cue_conflict": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="none")
    parser.add_argument("--perceiver", choices=["vit", "random"], default="vit")
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--refresh-every", type=int, default=125)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--save-images", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = apply_overrides(WORLD_CONFIG, args.ablation, args.scale)
    world = World()
    print(f"device={DEVICE} seed={args.seed} ablation={args.ablation} perceiver={args.perceiver}")
    print(world.summary())
    if args.save_images:
        import os
        os.makedirs(args.save_images, exist_ok=True)
        for s in world.all_states:
            render_state(*world.state_info[s]).save(os.path.join(args.save_images, s.replace(" ", "_") + ".png"))
    vit, processor = load_perceiver(args.perceiver, args.seed)
    enc = ImageEncoder(vit, processor, world, args.trainable_layers, args.ema_decay, args.refresh_every)
    model = WorldModel(enc.dim, cfg).to(DEVICE)
    print(f"trainable params: world model {sum(p.numel() for p in model.parameters() if p.requires_grad):,}, "
          f"encoder {sum(p.numel() for p in enc.trainable_parameters()):,}")
    report = run(model, enc, world, cfg, args.seed, extra=cue_conflict)
    report["config"] = {k: v for k, v in cfg.items()}
    report["args"] = vars(args)
    out = args.out or f"results_visual_{args.ablation}_{args.perceiver}_seed{args.seed}.json"
    save_json(report, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
