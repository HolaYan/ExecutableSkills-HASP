# HASP — Harnessing LLM Agents with Skill Programs

> **Paper:** *Harnessing LLM Agents with Skill Programs*
> **Authors:** Hongjun Liu, Yifei Ming, Shafiq Joty, Chen Zhao
> **Year:** 2026

HASP turns reusable agent experience into **executable Program Functions (PFs)**: small state-to-intervention programs that watch an agent rollout, detect recurring failure patterns, and repair the next decision only when needed.

Instead of giving the agent another piece of textual advice, a PF becomes part of the policy loop itself:

```text
state -> policy proposal -> retrieve candidate PFs -> detect -> intervene / abstain -> execute -> observe
```

A Program Function has two jobs:

1. **Detect** — `should_activate(ctx, action, arg)` decides whether the current state and the proposed action match a known failure pattern.
2. **Repair** — `intervene(ctx, action, arg)` returns a typed intervention: `inject(...)`, `redirect(...)`, or `abstain()`.

Both halves take the proposed action as an argument rather than reading it off the state, because that is what a PF audits: not a rollout in the abstract, but one action about to be taken.

If no PF activates, the original policy action executes unchanged.

The same executable skill object supports three stages of agent improvement:

* **Act** — repair recurring failures online without updating model weights.
* **Teach** — turn each intervention into an explicit before/after supervision signal.
* **Evolve** — revisit residual failures, compile new candidate PFs, validate them, and grow the skill library.

---

## Why executable skills?

Agents often fail in repetitive ways even when their reasoning is fluent and their tools work: they answer before reading enough evidence, commit to an early mistake, or repeat an unproductive strategy.

Textual skills can describe these lessons, but the policy still has to decide **when** to apply them and **how** they should change the current action. HASP makes that decision explicit by compiling a reusable lesson into a small executable program.

The key distinction is:

> **The policy proposes. The skill decides whether that proposal needs repair.**

Because HASP records both the original proposal and the repaired action, interventions are explicit and auditable, and the same records can later be reused for post-training and skill evolution.

---

## Demo & Blog

Watch the 2.5-minute HASP overview for the full story — from recurring agent failures, to executable Program Functions, to intervention, training, and skill evolution.

<!-- A bare user-attachments URL on its own line is the only form GitHub
     renders as a player: <video> tags are stripped and external hosts are not
     loaded. Re-upload by dragging the .mp4 into a comment box to change it. -->

https://github.com/user-attachments/assets/6379a190-25f7-42b4-bbc4-ab3f8add17c2

For the full walkthrough, interactive figures, and results, see the **[HASP Blog](https://holayan.github.io/HASP_Blog)**.

---

## Setup

```bash
conda create -n hasp python=3.11 -y && conda activate hasp   # or: python3.11 -m venv .venv
pip install -r requirements.txt
```

**Let vLLM resolve torch.** `requirements.txt` pins `vllm==0.15.1`, which brings `torch 2.9.1+cu128`; pinning torch separately against a different CUDA build is the usual way to end up with a vLLM that imports but cannot see a GPU.

`flash-attn` is deliberately absent — it compiles against your exact torch/CUDA pair and fails the whole install when it does not match. Add it afterwards with `--no-build-isolation` if you want it.

Qwen3.5-class models need vLLM 0.20 and are not supported by the pinned stack. They appear in one optional place, as a cross-model arbiter, so keep them in a second environment rather than upgrading the default:

```bash
export HASP_CONDA_ENV=<newer-env>
```

Every SLURM script reads `HASP_CONDA_ENV`.

### Paths

Nothing in the repository hard-codes a filesystem layout. Point these at your own paths — in the shell, or once in `~/.hasp_site`, which every SLURM job sources:

```bash
export HASP_DATA=/path/to/corpora             # eval sets and mined rollouts (default: <repo>/data)
export HASP_CACHE_ROOT=/path/to/cache          # torch / vLLM compile caches (default: <repo>/.cache)
export HASP_AGENTIC_RL=/path/to/agentic_rl    # only to locate upstream rollouts
export HASP_SKILLS_AGENT=/path/to/agent       # only for web/code episode files
```

`hasp_paths.py` documents every variable and its default. A missing required one raises with the name to set, rather than failing later on a path that never existed.

### Check it

```bash
python -m pytest tests/ -q
python -m skills.show --list
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```

The first two need neither a GPU nor any external data, and are the fastest way to confirm the layout is right.

---

## The skill format: Detect + Repair

A skill is **Detect** and **Repair**. Detect asks whether the current state and proposed action match this failure; Repair decides what should happen instead.

```python
@pf_skill(
    "compute_observation_verify",
    domain="math",
    anchor=Anchor(
        level="step",
        evidence="deterministic",
        trigger="a compute[...] action whose Observation the model wrote itself",
    ),
    summary="Re-evaluate every self-written compute[...] Observation and give the true value.",
)
class ComputeObservationVerify:
    def should_activate(self, ctx, action, arg) -> bool:      # Detect
        return "compute[" in ctx.reasoning.lower()

    def intervene(self, ctx, action, arg) -> Intervention:    # Repair
        f = first_finding(ctx, check_compute_observation)
        if f:
            return verdict(ctx, f, redo=True)
        if stalled(ctx):
            return continuation(ctx)
        return abstain()
```

The `anchor` says **where** the skill attaches (`step` or `final`), **what** must be present for Detect to inspect the state, and **how** the verdict is produced (`deterministic`, `executed`, `helper`, or `reminder`).

Keep the code and each skill's `SKILL.md` card in sync with:

```bash
python -m skills_construct.sync_anchors --check
```

The three intervention types are:

* `inject(...)` — state the finding and let the model redo the work.
* `redirect(to, arg)` — replace the proposed action with a different one.
* `abstain()` — leave the policy proposal unchanged.

A skill that finds nothing should stay silent. Most of this library injects rather than redirects: a redirect is taken *instead of* what the policy proposed, so a wrong one has no fallback, while a wrong injection still leaves the model holding the pen.

---

## Browsing the skills in this release

```bash
python -m skills.show --list
python -m skills.show --list --domain math
python -m skills.show compute_observation_verify --demo
python -m skills.show arithmetic_slip --on my_rollout.txt
```

`--demo` runs the skill on a sample rollout and prints its anchor, both modules, and exactly what it produced — no GPU and no dataset required. `--on FILE` does the same with a rollout of your own.

---

## Measuring one skill on its own

Normally the model picks skills from a menu. To measure a single skill independently, force its id; the selection turn is skipped, so the run measures that skill and nothing else.

```python
run_inference_pf_select(
    ...,
    force_skill_ids=["compute_observation_verify"],
)
```

Through the evaluation entry point:

```bash
python pf_select/eval_models.py \
    --model <model> \
    --tag <run-name> \
    --skills compute_observation_verify \
    --n 64
```

Omit `--skills` for the full model-driven protocol. Results land in:

```text
$HASP_DATA/model_eval/<tag>/
```

---

## Building your own skill

A skill lives in one place:

```text
skills/executable/<domain>/skills.py
```

Add a declaration and it is registered, selectable, and demonstrable.

### 1. Write the two modules

```python
@pf_skill(
    "modulus_result_check",
    domain=D,
    anchor=Anchor(
        level="step",
        evidence="deterministic",
        trigger="a step asserting 'x mod n = y'",
    ),
    summary="Recompute any stated modular result against the value the step claims.",
)
class ModulusResultCheck:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(_MOD.search(ctx.reasoning))

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_modulus)
        if f:
            return verdict(ctx, f, redo=True)
        return abstain()
```

**Detect** should be cheap and narrow. It answers “could this failure be here?” from what the model itself wrote.

**Repair** should produce the strongest verdict it can and abstain when it has none.

### Evidence helpers

| Helper                            | What it looks at                                   |
| --------------------------------- | -------------------------------------------------- |
| `first_finding(ctx, checker)`     | Reasoning split into steps; first checker hit wins |
| `answer_finding(ctx, arg, check)` | The committed answer as a whole                    |
| `stalled(ctx)`                    | Whether the rollout never committed an answer      |

Each helper takes the prefix of `(ctx, action, arg)` it actually uses, in that order, so a call site mirrors the method it sits in.

### Intervention helpers

| Helper                              | What it produces                         |
| ----------------------------------- | ---------------------------------------- |
| `verdict(ctx, f, redo=True)`        | Finding attached to the relevant step    |
| `correction(ctx, arg, note, value)` | States the corrected value               |
| `helper_verdict(ctx, scope)`        | Verdict from the PF helper model         |
| `continuation(ctx)`                 | Prompt to continue and eventually finish |
| `reminder(ctx, text)`               | Family-level hint                        |
| `abstain()`                         | No intervention                          |

A Repair body should usually chain evidence from strongest to weakest:

```python
def intervene(self, ctx, action, arg):
    f = first_finding(ctx, check_compute_observation)
    if f:
        return verdict(ctx, f, redo=True)
    if stalled(ctx):
        return continuation(ctx)
    return reminder(ctx, self.HINT) or abstain()
```

Prefer `inject` when possible: stating the finding preserves fallback-to-original behavior, because the model still re-commits the next action.

### 2. Put the checker next to its peers

```python
def check_modulus(step_text, full_response, step_start):
    m = _MOD.search(step_text)
    if not m:
        return None

    a, b, c = map(int, m.groups())
    if a % b == c:
        return None

    return {
        "pf": "modulus_result_check",
        "verdict": f"{a} mod {b} = {a % b}, not {c}",
        "fix": str(a % b),
    }
```

Return `None` for “nothing wrong here.”

### 3. Write the card

Create:

```text
skills/textual/<domain>/<skill_id>/SKILL.md
```

Then synchronize the anchor:

```bash
python -m skills_construct.sync_anchors --write
```

### 4. Check it

```bash
python -m skills.show modulus_result_check --demo
python -m skills_construct.sync_anchors --check
python tests/pf_parity.py --check .baseline.json
```

### 5. Then measure it

**Firing is not the same as working.**

```bash
python pf_select/eval_models.py \
    --model <model> \
    --tag mycheck \
    --skills modulus_result_check \
    --n 64
```

A skill earns its place in the library by improving behavior, not merely by activating.

---

## Training with skills

Making skills executable also creates structured learning signals. Every activation records:

```text
original policy proposal -> PF intervention -> executed behavior -> outcome
```

In the paper, PF-corrected trajectories are used for SFT, rejection sampling, and on-policy distillation.

Each intervention is evaluated along four dimensions:

| Signal      | Question                              | Weight |
| ----------- | ------------------------------------- | -----: |
| Timing      | Did we intervene at the right moment? |   0.15 |
| Mode        | Did we intervene in the right way?    |   0.10 |
| Correctness | Was the repair itself valid?          |   0.25 |
| Outcome     | Did the intervention actually help?   |   0.50 |

A trajectory is therefore not selected only because it eventually reaches the correct answer; the intervention itself must occur at the right state, in the right mode, and produce a useful repair.

In the repository, skills enter training by pointing the config at a library:

```yaml
rollout:
  skill_library_dir: "./skills/math"
```

Example:

```bash
python -m training.rejection_sampling.train \
    --config configs/training/H1_math_rs.yaml
```

On SLURM:

```bash
CONFIG=configs/training/H1_math_rs.yaml \
  sbatch --partition=<p> --account=<a> scripts/slurm/train.sbatch
```

---

## Growing the library during training

Training does not eliminate every failure. HASP revisits residual failures under the updated policy, identifies recurring failure–repair patterns, compiles them into candidate PFs, validates them, and adds accepted skills back to the library.

Enable evolution with:

```yaml
evolve:
  enabled: true
  every_steps: 200
  eval_size: 48
  max_admit_per_cycle: 2
```

One cycle can also be run against a checkpoint:

```bash
python -m evolving.run --ckpt <path> --domain math --generation 1
python -m evolving.run --history --library <output_dir>/library
```

Two practical lessons from the paper are important:

1. **More skills are not automatically better skills.**
2. **Validation is part of evolution, not an afterthought.**

---

## Repository layout

```text
skills/
  pf_template.py            the two-module contract, Anchor, typed interventions
  executable/<domain>/      skills.py — every executable skill, declared once
  textual/<domain>/         SKILL.md cards

pf_select/                  runtime: selection, dispatch, evaluation
skills_construct/           mining and generation of new skills
evolving/                   in-training skill-library growth
self_improving/             propose, review, pseudo-gradient growth path
training/                   SFT / rejection sampling / policy training with skills
tests/                      behavioral parity and anchor-consistency guards
```

---

## Citation

If you use HASP in academic work, please cite:

```bibtex
@misc{liu2026harnessingllmagentsskill,
      title={Harnessing LLM Agents with Skill Programs}, 
      author={Hongjun Liu and Yifei Ming and Shafiq Joty and Chen Zhao},
      year={2026},
      eprint={2605.17734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.17734}, 
}
```

---

**HASP — Skills you can execute.**
