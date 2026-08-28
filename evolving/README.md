# evolving — PF evolution inside the training loop

Every N training steps, pause, evaluate the weights as they are right now,
and turn that checkpoint's own failures into new PFs.

```
   training ──── step ──── step ──── step ──── step ──── step ──── ▶
                            │                              │
                     every N steps                  every N steps
                            ▼                              ▼
                 ┌──────── evolve cycle ────────┐
                 │                              │
                 │  1. eval the LIVE weights    │   pass@1 on a held-out set
                 │        on a held-out set     │   → the progress curve
                 │              │               │
                 │              ▼               │
                 │  2. split by correctness     │   wrong  = what to mine
                 │              │               │   correct = the control set
                 │              ▼               │
                 │  3. cluster → propose → gate │   forge's cheap gates
                 │              │               │
                 │              ▼               │
                 │  4. append to the run's      │   with generation + step
                 │        library               │   stamped on every PF
                 └──────────────┬───────────────┘
                                ▼
                    next rollout phase uses the grown library
```

The point of doing this *inside* training rather than before it: after a few
hundred steps the model is no longer the model the library was mined from. Its
failures have moved. A library frozen at step 0 is aimed at a model that no
longer exists.

## Why the eval does double duty

The pause runs one eval and gets two things from it:

- **pass@1 on held-out questions** — a progress curve that is not training loss;
- **the screening corpus** — the same rollouts, split by correctness. Wrong
  ones are the material to mine; correct ones are the false-positive floor,
  which is the gate that killed most candidates in every offline round.

Both come from the *current* checkpoint, which is the whole point. No separate
mining pass, and no corpus staleness.

## What the gates are here, and what they are not

Mid-training only the cheap gates can run. The end-to-end probe needs its own
regeneration pass and the accuracy test needs n=64; neither belongs inside a
training step.

| gate | runs here? | why |
|---|---|---|
| structural (`skills_construct/forge/spec.py`) | yes | pure AST, microseconds |
| precision screen vs the correct set | yes | seconds of CPU |
| end-to-end probe (tier 1) | **no** | needs a regeneration pass |
| n=64 accuracy (tier 2) | **no** | hours of GPU |

**Everything this module admits is therefore provisional.** It has been shown
to fire on the current model's failures and to stay quiet on its successes. It
has *not* been shown to change an answer, let alone to change accuracy. Every
generated `SKILL.md` says so, and `on_train_end` logs it.

Paying that debt is a `forge` job after the run: point it at the evolved
library, probe the admitted PFs, and run the tier-2 accuracy test. Only what
survives that should be described as measured.

The proposer is also weaker here: the model being trained is the only one on
the GPU, so it writes its own checkers. That is a real limitation, not a design
flourish — it makes the gates matter more, not less.

## Reviewing what is already there — the four credit signals

Distilling new skills is only half a loop. A cycle that only adds grows a
library nobody can account for, so each cycle also scores the skills that
*fired* during its own evaluation. The rollouts already exist, so this costs
nothing beyond CPU.

One intervention is judged on four separate questions, and a skill can fail any
of them independently:

| family | asks | a skill fails it by |
|---|---|---|
| **timing** (S1) | did it fire on the steps that were actually risky? | firing everywhere, or never where it mattered |
| **modality** (S2) | did it intervene at the right point of the ReAct cycle? | speaking after the action it should have preceded |
| **correctness** (S3) | was what it said well-formed, on-topic, right for the domain? | producing a verdict the policy cannot act on |
| **outcome** (S4) | did the rollout end better, net of what the intervention cost? | helping locally and hurting downstream |

Each family is fifteen sub-signals aggregated
(`training/signals/s1..s4`), and the review keeps all four rather than summing
them, because the distinction is the point: *fires at the wrong time* and *fires
at the right time and says something useless* are different faults with
different fixes. Collapsing them to one number hides which one you have.

Every cycle logs the table, worst outcome first:

```
  skill                                timing  modality  correct  outcome  fires
  ------------------------------------------------------------------------------
  verification_missing                  +0.50     +0.35    +0.15    +0.00      1
  compute_observation_verify            +0.50     +0.35    +0.15    +0.20      2
```

and names any skill whose own interventions are not paying off:

```
[evolve] some_skill is not paying off: outcome -0.31 — the rollouts it touched
         ended no better, net of what it cost; timing -0.12 — it is firing on
         steps that were not the risky ones
```

**Flagged, never deleted.** Retiring a skill is a decision about the library;
the signals are evidence for it. A skill that fired fewer than
`review_min_fires` times is left alone — silence is the expected behaviour for
most skills on most rollouts, and scoring a skill on one or two fires says more
about the sample than about the skill.

The per-generation scores go into `evolution.jsonl` alongside what was admitted,
so a library's history records both halves: what it gained, and how what it
already had was doing at the time.

```yaml
evolve:
  review_enabled: true
  review_min_fires: 3    # below this, too few interventions to judge
```

## When a new PF actually takes effect

`SkillRolloutRunner.setup()` loads the library **once**. So:

- **RS** builds a fresh Rollouter per iteration → PFs admitted during
  iteration *k*'s SFT phase are live for iteration *k+1*'s rollouts;
- **online-rollout training** picks them up when its runner is next set up;
- **pure SFT with no rollouts** — the library grows and is recorded, but
  nothing in that run consumes it. That is fine (the library is the artifact),
  but do not expect the loss curve to react.

## The run-scoped library

Training never writes into `skills/`. Generation 0 copies the domain library to
`{output_dir}/{exp_id}/library/`, and `rollout.skill_library_dir` is redirected
there automatically (`_resolve_library_dir` in `rejection_sampling/train.py`).

```
{output_dir}/{exp_id}/library/
├── dynamic_program_functions.py   # chain-loads evidence_pfs.py, then evolved_pfs.py
├── evidence_pfs.py                # the hand-written library, copied at gen 0
├── evolved_pfs.py                 # grown here; each block stamped [generation N, step S]
├── <skill_id>/SKILL.md            # one per admitted PF, carrying its screening numbers
└── evolution.jsonl                # the generation ledger
```

Every PF block records the generation and step that produced it, so a result
can be attributed to the library that was actually live at the time, and any
generation can be rolled back.

## Turning it on

Add an `evolve:` block to a training config:

```yaml
evolve:
  enabled: true
  every_steps: 200        # optimizer steps between cycles
  skip_first: 200         # early checkpoints fail for training reasons, not skill reasons
  max_generations: 8
  eval_size: 48           # held-out questions per cycle — this is a probe, keep it small
  eval_dataset: "math500" # defaults per domain
  families_per_cycle: 3
  candidates_per_family: 2
  max_admit_per_cycle: 2  # cap library growth per generation
  review_enabled: true    # score what the library already has (see above)
  review_min_fires: 3
  min_fire_wrong: 0.05
  max_fire_correct: 0.05
  min_lift: 2.0
```

Defaults are conservative on purpose: a cycle that is too frequent, too large,
or too permissive does not produce a better library, it produces a run whose
results cannot be attributed to anything.

## Outside the training loop

```bash
python -m evolving.run --ckpt <path> --domain math --generation 1
python -m evolving.run --history --library training/outputs/H1_math_rs_evidence_pfs/library
```

## Failure behaviour

A cycle must never take a training run down — one may have been queued for a
day. Every cycle is wrapped; a failure is logged and training continues with
the library it already had. Generation also restores the model exactly as it
found it: eval mode and a KV cache are needed to generate, and neither may
leak into the next training step (`_inference_mode` in `eval_probe.py`).

---

## The other growth path

`self_improving/` + `training/closed_loop/` grow the library too, at epoch
granularity, and they read the same four signals — but as *credit* rather than
as a filter. A pseudo-gradient over a run's trajectories is what proposes a
skill there, and a five-dimension model review is what admits it, where this
loop proposes from the checkpoint's own failures and admits on a precision
screen against solutions that were already correct.

Neither subsumes the other. `training/README.md` compares them side by side.
