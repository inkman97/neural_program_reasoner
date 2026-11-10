# Neural Program Reasoner (NPR)

**Learning rule programs from a few examples by composing neural primitives**

*Alessandro Schino — Independent Research, 2026*

---

## Status

This repository contains the architecture, the simulated environments and the full evaluation protocol. **No results are reported yet.** The experiments listed below are designed so that a positive result would be informative and a negative one would be detected. Numbers should only be added once they have been produced by the scripts in this repository, on the splits defined here, with several seeds.

## What NPR is

NPR is a few-shot reasoner. Given a handful of input → output examples of an unknown transformation, it:

1. encodes the examples with a frozen (or partly trainable) perceiver,
2. **synthesizes a short program** over a library of neural primitives,
3. **executes** that program on a new input, with a learned halting rule,
4. decodes the result by retrieval among candidate states or words.

Frequently used pairs of primitives can be **compressed** into new macro-primitives between training cycles.

The design goal is that the program is not decorative: the information about *which* rule applies has to pass through the program, and this is tested directly (see *Program intervention* below).

## Repository layout

| File | Content |
|---|---|
| `npr_core.py` | Primitive library, rule synthesizer, program execution with halting, program memory, compression |
| `npr_world.py` | Simulated physical world, object-level train/test split, episode sampling, symbolic baselines |
| `npr_world_model.py` | World model (perceiver-agnostic), training loop, full evaluation suite |
| `npr_text_world_model.py` | World model with a text perceiver (GPT-2) |
| `npr_visual_world_model.py` | World model with a visual perceiver (ViT, online/EMA target), scene renderer, cue-conflict test |
| `npr_linguistic.py` | Word-relation reasoning on the Google analogy dataset, with fair baselines |
| `NPR_Paper.pdf` | Technical report: architecture and evaluation protocol |

## Architecture

### Primitive library

| Primitive | Definition | Structural property |
|---|---|---|
| IDENTITY | `x` | exact no-op |
| NEGATE | Householder reflection `x − 2 (v·x)/(v·v) v` | exactly involutive: NEGATE(NEGATE(x)) = x |
| MORPH | gated residual MLP | — |
| ASSOCIATE | self-attention over 12 sub-spaces of the vector | genuine attention, not a scalar gate |
| LOOKUP | differentiable key–value memory | content-addressed retrieval |
| BLEND | gated mix of the state with the **rule code** | the only primitive that reads the rule code |

A compressed macro `A_B` is initialised as the exact composition `B(A(x))` plus a zero-initialised correction, so compression never changes behaviour at the moment it happens.

### Rule synthesizer

A two-layer transformer encodes the examples (each example is `[input, action, output, output − input]`) into a signature. Per-step heads choose a primitive with straight-through Gumbel-Softmax; a halting head (ACT-style) decides the program length. At evaluation time programs are discrete and the halting decision is hard.

### Where the rule information can flow

In the default configuration the decoder sees **only the object vector and the transformed property vector**. It never sees the action label or the example signature. The rule reaches the output only through:

- the choice of primitives, and
- the rule code consumed by BLEND.

This is what makes the program testable. The ablations below re-open the other channels on purpose.

### World model

- **Perceiver**: GPT-2 hidden states (text) or ViT patch tokens (images). In the visual version the last ViT layers are trained and the targets come from an EMA copy.
- **Object extractor** and **slot attention** split the input into an object vector and property slots; the slot to edit is chosen from the example signature.
- **Executor** applies the synthesized program to the selected slot.
- **Decoder** maps (object, edited property) to a latent prediction, scored against all candidate states by cosine similarity. Unseen states can be produced, which is required for held-out objects.
- **Planner**: beam search that simulates every action through the world model and checks whether the decoded state is the goal. Success is verified by executing the plan in the symbolic world, so any valid plan counts, not only the reference one.

## The simulated world

70 objects, 20 actions, 11 property types. Actions belong to three classes, so that no single shortcut solves everything:

| Class | Examples | Why it is there |
|---|---|---|
| Fixed target | open, close, fill, boil, chill | Solvable from the action name alone; serves as a sanity check |
| State-dependent | heat, cool (one step on an ordinal scale), toggle, wait (11 transitions) | The output depends on the input state, not only on the action |
| Object-dependent | push (fragile objects break instead of falling) | The output depends on object identity |

Temporal effects (charging, cooking, growth, burning, natural decay) are encoded as ordinal states and advanced by `wait`.

## Splits

- **Objects**: 18 objects (at least one per property type, including temporal ones and one fragile object) are never seen during training. All reported test numbers are on these objects.
- **Examples** in every test episode come from training objects only.
- **Linguistic**: 20% of the source words of each relation are held out. Compositional chains are tested only when their first pair is held out, and chains that contain a held-out pair are excluded from training.

## Example policies

Each test episode is evaluated under three policies:

| Policy | Examples given | What it measures |
|---|---|---|
| `same_transition` | same action **and** same input value as the test | copying from the examples is enough |
| `same_action` | same action, any input values | the rule has to be conditioned on the input state |
| `anonymous` | as above, but the action identity is hidden | the rule has to be inferred from input→output pairs alone |

## Baselines

All baselines are evaluated on exactly the same episodes.

| Baseline | Information used |
|---|---|
| `label_table` | action name + parsed input value, lookup table built on training objects |
| `example_copy` | the examples only: copy the output of an example with the same input value |
| `majority` | the examples only: most frequent output value |
| `vector_offset` (linguistic) | mean example offset added to the input word |
| `gpt2_fewshot` (linguistic) | GPT-2 log-likelihood over **all** candidate words, multi-token words included |

The symbolic baselines read the structured state directly, so they are an upper bound on perception. They are strong on purpose: NPR is only interesting where it beats them, or matches them without being given the action name.

## Tests to run

1. **Single-step generalization** on held-out objects, per policy, per action class.
2. **Multi-step rollouts** (depth 2 and 3) executed in latent space, not step by step with gold inputs.
3. **Temporal**: `wait` accuracy per transition; temporal chains both as rollouts and step by step.
4. **Planning** on held-out objects, success measured by symbolic execution.
5. **Program intervention**: replace the program and rule code of an episode with those of a different action and check whether the prediction follows the donor rule. If the program carries the rule, the answer should follow the donor; if it keeps the original answer, the program is decorative.
6. **Program diversity**: number of distinct programs across actions.
7. **Structural checks**: NEGATE involution error (expected at floating-point precision).
8. **Ablations**: `decoder_bypass` (decoder sees action and signature), `film` (rule injected by FiLM), `no_program` (an MLP replaces the program).
9. **Perceiver control**: the same pipeline with a randomly initialised, frozen perceiver (`--perceiver random`).
10. **Cue conflict** (visual): temperature scenes whose liquid colour and surface effects disagree, to see which cue the model reads.
11. **Linguistic**: held-out pairs and held-out compositional chains against `vector_offset` and full-vocabulary GPT-2.

Every configuration should be run with at least 5 seeds; report mean and standard deviation.

## Running

```bash
pip install -r requirements.txt

python npr_text_world_model.py --seed 0
python npr_visual_world_model.py --seed 0
python npr_linguistic.py --seed 0

python npr_text_world_model.py --seed 0 --ablation decoder_bypass
python npr_text_world_model.py --seed 0 --ablation no_program
python npr_text_world_model.py --seed 0 --perceiver random
```

Useful options:

- `--scale 0.1` shortens training, for quick checks;
- `--out file.json` sets the results file (by default it is named after the configuration);
- `--save-images DIR` (visual) dumps every rendered scene for inspection.

Each run prints a summary and writes all metrics to a JSON file.

A GPU is recommended. The visual model trains part of a ViT and is the slowest.

## Known limitations of the setting

- The world is small and synthetic; states are discrete and fully described by one property at a time.
- Renderings are clean and noise-free. The background no longer encodes the property value, but other visual cues still correlate with it by construction.
- Only two slots are used.
- The perceivers are small by current standards.

## Citation

```bibtex
@misc{schino2026npr,
  title  = {Neural Program Reasoner: Learning Rule Programs from Few Examples by Composing Neural Primitives},
  author = {Schino, Alessandro},
  year   = {2026}
}
```
