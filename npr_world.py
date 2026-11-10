import random
from collections import Counter, defaultdict

PROPERTY_VALUES = {
    "temperature": ["cold", "warm", "hot", "boiling"],
    "openness": ["closed", "open"],
    "power": ["off", "on"],
    "fullness": ["empty", "full"],
    "integrity": ["intact", "broken"],
    "position": ["surface", "floor"],
    "containment": ["outside", "inside"],
    "battery": ["empty", "charging_1", "charging_2", "full", "half"],
    "cooking_state": ["raw", "cooking_1", "cooking_2", "cooked", "burnt"],
    "growth": ["seed", "growing_1", "growing_2", "grown"],
    "burn_state": ["unlit", "burning", "dim", "extinguished"],
}

TEMPORAL_PROPERTIES = {"battery", "cooking_state", "growth", "burn_state"}

OBJECTS = {
    "water": {"temperature": True}, "soup": {"temperature": True},
    "coffee": {"temperature": True}, "tea": {"temperature": True},
    "milk": {"temperature": True}, "food": {"temperature": True},
    "oil": {"temperature": True}, "pan": {"temperature": True},
    "iron": {"temperature": True}, "metal": {"temperature": True},
    "door": {"openness": True}, "window": {"openness": True},
    "box": {"openness": True}, "jar": {"openness": True},
    "gate": {"openness": True}, "drawer": {"openness": True},
    "lid": {"openness": True}, "bag": {"openness": True},
    "bottle": {"openness": True}, "cabinet": {"openness": True},
    "lamp": {"power": True, "surface": "table", "fragile": True, "integrity": True},
    "light": {"power": True}, "screen": {"power": True},
    "tv": {"power": True}, "radio": {"power": True},
    "computer": {"power": True}, "fan": {"power": True},
    "heater": {"power": True}, "oven": {"power": True},
    "speaker": {"power": True},
    "cup": {"fullness": True, "surface": "shelf", "fragile": True, "integrity": True},
    "glass": {"fullness": True, "surface": "table", "fragile": True, "integrity": True},
    "bowl": {"fullness": True, "fragile": True, "integrity": True},
    "pot": {"fullness": True}, "tub": {"fullness": True},
    "bucket": {"fullness": True}, "jug": {"fullness": True},
    "tank": {"fullness": True}, "pool": {"fullness": True},
    "plate": {"integrity": True, "surface": "counter", "fragile": True},
    "vase": {"integrity": True, "fragile": True, "surface": "shelf"},
    "mirror": {"integrity": True},
    "ball": {"surface": "table"}, "book": {"surface": "desk"},
    "phone": {"surface": "bed"}, "toy": {"surface": "shelf"},
    "pen": {"surface": "desk"}, "remote": {"surface": "couch"},
    "key": {"container": "pocket"}, "coin": {"container": "jar"},
    "letter": {"container": "envelope"}, "shirt": {"container": "closet"},
    "tool": {"container": "shed"}, "food_c": {"container": "fridge"},
    "ball_c": {"container": "box"}, "toy_c": {"container": "bag"},
    "pen_c": {"container": "case"}, "book_c": {"container": "drawer"},
    "phone_t": {"battery": True}, "laptop": {"battery": True}, "tablet": {"battery": True},
    "bread": {"cooking_state": True}, "cake": {"cooking_state": True}, "chicken": {"cooking_state": True},
    "tomato": {"growth": True}, "flower": {"growth": True}, "herb": {"growth": True},
    "candle": {"burn_state": True}, "fire": {"burn_state": True}, "torch": {"burn_state": True},
}

ACTIONS = {
    "heat": {"temperature": {"cold": "warm", "warm": "hot", "hot": "boiling"}},
    "cool": {"temperature": {"boiling": "hot", "hot": "warm", "warm": "cold"}},
    "boil": {"temperature": {"cold": "boiling", "warm": "boiling", "hot": "boiling"}},
    "chill": {"temperature": {"warm": "cold", "hot": "cold", "boiling": "cold"}},
    "open": {"openness": {"closed": "open"}},
    "close": {"openness": {"open": "closed"}},
    "switch on": {"power": {"off": "on"}},
    "switch off": {"power": {"on": "off"}},
    "toggle": {"power": {"off": "on", "on": "off"}, "openness": {"closed": "open", "open": "closed"}},
    "fill": {"fullness": {"empty": "full"}},
    "empty out": {"fullness": {"full": "empty"}},
    "drop": {"integrity": {"intact": "broken"}},
    "push": {"position": {"surface": "floor"}},
    "put in": {"containment": {"outside": "inside"}},
    "take out": {"containment": {"inside": "outside"}},
    "charge": {"battery": {"empty": "charging_1", "half": "charging_1"}},
    "put in oven": {"cooking_state": {"raw": "cooking_1"}},
    "plant": {"growth": {"seed": "growing_1"}},
    "light": {"burn_state": {"unlit": "burning"}},
    "wait": {
        "battery": {"charging_1": "charging_2", "charging_2": "full", "full": "half", "half": "empty"},
        "cooking_state": {"cooking_1": "cooking_2", "cooking_2": "cooked", "cooked": "burnt"},
        "growth": {"growing_1": "growing_2", "growing_2": "grown"},
        "burn_state": {"burning": "dim", "dim": "extinguished"},
    },
}

STATE_DEPENDENT_ACTIONS = {"heat", "cool", "toggle", "wait", "push"}

HELD_OUT_OBJECTS = {
    "temperature": ["iron", "metal"], "openness": ["bottle", "cabinet"],
    "power": ["oven", "speaker"], "fullness": ["jug", "tank"],
    "integrity": ["mirror", "vase"], "position": ["pen", "remote"],
    "containment": ["tool", "book_c"], "battery": ["tablet"],
    "cooking_state": ["chicken"], "growth": ["herb"], "burn_state": ["torch"],
}

POLICIES = ["same_transition", "same_action", "anonymous"]


def display_name(key):
    return key.split("_")[0]


def object_properties(key):
    spec = OBJECTS[key]
    props = [p for p in PROPERTY_VALUES if spec.get(p) is True]
    if "surface" in spec:
        props.append("position")
    if "container" in spec:
        props.append("containment")
    return props


def render(key, prop, value):
    spec, name = OBJECTS[key], display_name(key)
    if prop == "position":
        return f"the {name} is on the {spec['surface']}" if value == "surface" else f"the {name} is on the floor"
    if prop == "containment":
        return f"the {name} is {value} the {spec['container']}"
    return f"the {name} is {value}"


def step(key, prop, value, action):
    table = ACTIONS[action].get(prop)
    if not table or value not in table:
        return None
    if action == "push" and OBJECTS[key].get("fragile") and OBJECTS[key].get("integrity"):
        return "integrity", "broken"
    return prop, table[value]


class Transition:
    __slots__ = ("obj", "prop", "src", "action", "dst_prop", "dst", "state", "result")

    def __init__(self, obj, prop, src, action, dst_prop, dst):
        self.obj, self.prop, self.src, self.action = obj, prop, src, action
        self.dst_prop, self.dst = dst_prop, dst
        self.state = render(obj, prop, src)
        self.result = render(obj, dst_prop, dst)

    @property
    def key(self):
        return self.action, self.prop, self.src

    def __repr__(self):
        return f"{self.state} + {self.action} -> {self.result}"


class World:
    def __init__(self, held_out=None):
        held_out = HELD_OUT_OBJECTS if held_out is None else held_out
        self.held_out = {o for objs in held_out.values() for o in objs}
        self.transitions = []
        for key in OBJECTS:
            for prop in object_properties(key):
                for value in PROPERTY_VALUES[prop]:
                    for action in ACTIONS:
                        nxt = step(key, prop, value, action)
                        if nxt is not None:
                            self.transitions.append(Transition(key, prop, value, action, nxt[0], nxt[1]))
        self.actions = sorted({t.action for t in self.transitions})
        self.train = [t for t in self.transitions if t.obj not in self.held_out]
        self.test = [t for t in self.transitions if t.obj in self.held_out]
        self.train_by_action = defaultdict(list)
        self.train_by_key = defaultdict(list)
        for t in self.train:
            self.train_by_action[t.action].append(t)
            self.train_by_key[t.key].append(t)
        self.state_info = {}
        for key in OBJECTS:
            for prop in object_properties(key):
                for value in PROPERTY_VALUES[prop]:
                    text = render(key, prop, value)
                    if text in self.state_info and self.state_info[text] != (key, prop, value):
                        raise ValueError(f"ambiguous state string: {text}")
                    self.state_info[text] = (key, prop, value)
        self.all_states = sorted(self.state_info)
        self.train_states = sorted(s for s, (k, _, _) in self.state_info.items() if k not in self.held_out)
        self.label_table = defaultdict(Counter)
        for t in self.train:
            self.label_table[t.key][(t.dst_prop, t.dst)] += 1

    def summary(self):
        lines = [f"objects={len(OBJECTS)} held_out={len(self.held_out)} actions={len(self.actions)}",
                 f"transitions={len(self.transitions)} train={len(self.train)} test={len(self.test)}",
                 f"states={len(self.all_states)} train_states={len(self.train_states)}"]
        return "\n".join(lines)

    def examples(self, t, policy, n, rng):
        pool = []
        if policy == "same_transition":
            pool = [e for e in self.train_by_key[t.key] if e.obj != t.obj]
        if not pool:
            pool = [e for e in self.train_by_action[t.action] if e.obj != t.obj]
        return rng.sample(pool, min(n, len(pool)))

    def episode(self, transitions, policy, n_examples, rng):
        per_step = [self.examples(t, policy, n_examples, rng) for t in transitions]
        return {
            "state": transitions[0].state,
            "expected": transitions[-1].result,
            "actions": [t.action for t in transitions],
            "per_step": per_step,
            "anonymous": policy == "anonymous",
            "policy": policy,
            "transitions": transitions,
            "obj": transitions[0].obj,
            "depth": len(transitions),
        }

    def sample_training_episode(self, rng, cfg):
        policy = rng.choices(POLICIES, weights=cfg["policy_weights"])[0]
        if rng.random() < cfg["multistep_ratio"]:
            chain = self.random_chain(rng, rng.choice([2, 3]), self.train)
            if chain:
                return self.episode(chain, policy, cfg["num_examples"], rng)
        action = rng.choice(self.actions)
        t = rng.choice(self.train_by_action[action])
        return self.episode([t], policy, cfg["num_examples"], rng)

    def transitions_from(self, obj, prop, value):
        out = []
        for action in self.actions:
            nxt = step(obj, prop, value, action)
            if nxt is not None:
                out.append(Transition(obj, prop, value, action, nxt[0], nxt[1]))
        return out

    def random_chain(self, rng, depth, pool):
        t = rng.choice(pool)
        chain = [t]
        while len(chain) < depth:
            nxt = [x for x in self.transitions_from(t.obj, t.dst_prop, t.dst) if x.action != t.action]
            if not nxt:
                return None
            t = rng.choice(nxt)
            chain.append(t)
        return chain

    def all_chains(self, depth, objects, rng, limit_per_object=6):
        chains = []
        for obj in sorted(objects):
            found = []
            starts = [t for t in self.transitions if t.obj == obj]

            def grow(chain):
                if len(chain) == depth:
                    if chain[0].state != chain[-1].result:
                        found.append(list(chain))
                    return
                last = chain[-1]
                for x in self.transitions_from(obj, last.dst_prop, last.dst):
                    if x.action != last.action:
                        chain.append(x)
                        grow(chain)
                        chain.pop()

            for s in starts:
                grow([s])
            rng.shuffle(found)
            chains.extend(found[:limit_per_object])
        return chains

    def eval_episodes(self, split, policy, n_examples, seed):
        rng = random.Random(seed)
        pool = self.test if split == "heldout" else self.train
        return [self.episode([t], policy, n_examples, rng) for t in pool]

    def temporal_chains(self, objects):
        out, seen = [], set()
        for obj in sorted(objects):
            for prop in object_properties(obj):
                if prop not in TEMPORAL_PROPERTIES:
                    continue
                initiators = [a for a in self.actions if a != "wait" and prop in ACTIONS[a]]
                plans = [("initiated", initiators[:1] + ["wait"] * 3), ("decay", ["wait"] * 3)]
                for value in PROPERTY_VALUES[prop]:
                    for kind, actions in plans:
                        seq, cur = [], (prop, value)
                        for action in actions:
                            nxt = step(obj, cur[0], cur[1], action)
                            if nxt is None:
                                break
                            seq.append(Transition(obj, cur[0], cur[1], action, nxt[0], nxt[1]))
                            cur = nxt
                        sig = tuple((t.state, t.action) for t in seq)
                        if len(seq) >= 2 and sig not in seen:
                            seen.add(sig)
                            out.append((kind, seq))
        return out

    def execute(self, state_text, actions):
        key, prop, value = self.state_info[state_text]
        cur = (prop, value)
        for a in actions:
            nxt = step(key, cur[0], cur[1], a)
            if nxt is not None:
                cur = nxt
        return render(key, cur[0], cur[1])


def _majority(counter):
    if not counter:
        return None
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def baseline_label_table(world, ep):
    obj = ep["obj"]
    prop, value = ep["transitions"][0].prop, ep["transitions"][0].src
    for action in ep["actions"]:
        best = _majority(world.label_table.get((action, prop, value), Counter()))
        if best is None:
            return render(obj, prop, value)
        prop, value = best
    return render(obj, prop, value)


def baseline_example_copy(world, ep):
    obj = ep["obj"]
    prop, value = ep["transitions"][0].prop, ep["transitions"][0].src
    for examples in ep["per_step"]:
        exact = Counter((e.dst_prop, e.dst) for e in examples if e.prop == prop and e.src == value)
        same_prop = Counter((e.dst_prop, e.dst) for e in examples if e.prop == prop)
        anyp = Counter((e.dst_prop, e.dst) for e in examples)
        best = _majority(exact) or _majority(same_prop) or _majority(anyp)
        if best is None:
            break
        prop, value = best
    try:
        return render(obj, prop, value)
    except KeyError:
        return None


def baseline_majority(world, ep):
    obj = ep["obj"]
    prop, value = ep["transitions"][0].prop, ep["transitions"][0].src
    for examples in ep["per_step"]:
        best = _majority(Counter((e.dst_prop, e.dst) for e in examples))
        if best is None:
            break
        prop, value = best
    try:
        return render(obj, prop, value)
    except KeyError:
        return None


BASELINES = {
    "label_table": baseline_label_table,
    "example_copy": baseline_example_copy,
    "majority": baseline_majority,
}
