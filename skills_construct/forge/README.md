# forge — generating PF skills from real error cases, in a loop

This module turns the process we ran by hand into code: read real wrong
solutions, find what the model *wrote down* that can be re-verified, write a
deterministic checker for it, measure whether it actually helps, sharpen it if
it nearly did, admit it only if accuracy moved — and carry the verdict forward
so the next round starts somewhere new.

**generate → test → refine → admit → next round**

```
  ╔══════════════════════════════ round N ══════════════════════════════╗
  ║                                                                     ║
  ║   ── generate ──────────────   ── test ─────────   ── refine ──      ║
  ║                                                                     ║
  ║   cluster ────> propose ────>  screen ────> probe ────> refine       ║
  ║   which          candidate     offline      marginal    sharpen      ║
  ║   families       checkers      precision    rescue      the verdict  ║
  ║   have a free    (1 GPU)       (CPU)        / broke     / narrow     ║
  ║   claim surface      ▲                      (tier 1)    the trigger  ║
  ║                      │                          ▲           │       ║
  ║                      │                          └───────────┘       ║
  ║                      │                      re-screen + re-probe    ║
  ╚══════════════════════╪══════════════════════════╪══════════════════╝
                         │                          │ survivors
                         │                          ▼
                         │        ── admit ──────────────────────────────
                         │
                         │        tier 2: THE REAL TEST
                         │        n=64 pf_select, pass@1 before → after
                         │                          │
                         │             ┌────────────┴────────────┐
                         │      accuracy moved              it didn't
                         │             │                         │
                         │             ▼                         ▼
                         │      skills/<domain>/             retired
                         │      admitted library                 │
                         │             │                         │
                         │             └────────────┬────────────┘
                         │                          ▼
                         │              ledger: what was admitted,
                         └───────────── what failed, and why
                            falsified list                │
                            into the prompt               ▼
                                                     ── next round ──
```

Only the `admit` step touches the hand-written library, and it needs a measured
accuracy change to fire. Everything to its left is candidate work happening in
a throwaway probe library.

| stage | machine | what it decides |
|---|---|---|
| `cluster` | login | which error families have a free, checkable claim surface |
| `propose` | 1 GPU | candidate checkers, told what already failed |
| `screen` | CPU | offline precision — kills most candidates for free |
| `probe` | 1 GPU | tier 1: *marginal* rescue / broke end to end |
| `refine` | 1 GPU | one sharpening pass over the near-misses |
| tier 2 | 1 GPU | the real test: pass@1 before → after |
| `admit` | login | promote on measured accuracy, or retire into the ledger |

One round is one command:

```bash
DOMAIN=math sbatch scripts/slurm/forge_loop.sbatch
```

It prints its own tier-2 command and admit command when it finishes.

## Why the gate is the module

A candidate is judged on how much more often it fires on failures than on
solutions that were already correct. The thresholds in `screen.py` are
constants at the top of that file; set them from your own measurements before
trusting a verdict from this pipeline.
Those constants come from the table, not from taste. Passing the gate earns a
candidate an end-to-end run — **not** a place in training.

## The rules the proposer is held to

1. **A checker may only read what the model itself wrote.** `spec.py` rejects
   any checker whose source mentions `gold` / `label` / `ground_truth`, and
   `_screen_runner.py` never puts an answer key in `ctx`. A checker that needs
   the answer scores perfectly offline and is worthless at inference.
2. **Anchor on a checkable claim.** Every PF that worked re-verifies something
   the model asserted — its own `compute[...]` Observation, a uniqueness
   claim, a doctest. "The reasoning seems confused" is not checkable.
3. **Silence is the default.** The checker runs on correct solutions too.
4. **Deterministic or executed evidence only.** Helper-model evidence cannot be
   screened offline against the correct-set control, so it is not forgeable
   here — hand-write those.

## Stages

**cluster** — buckets wrong cases by error family and reports, per family, which
*claim surfaces* appear (self-computed observation, boxed commit, enumeration,
uniqueness claim, spec example…) and whether each is already taken by an
existing PF. A family with population but no free claim surface has nothing for
a deterministic checker to attach to; that is a result, not a failure.

```bash
python -m skills_construct.forge --stage cluster --domain math
```

**propose** — a local model (default Qwen3-8B, one L40S, no API) sees the
family, its claim surfaces, and 4 real failures in full, and returns candidate
`PFSpec`s with checker source. `spec.validate_spec` then applies the structural
gate — signature, gold-leak, forbidden imports — before anything is run.

```bash
MODEL=Qwen/Qwen3-8B DOMAIN=math sbatch scripts/slurm/forge.sbatch
```

**screen** — the gate. Each checker runs over the whole corpus in a contained
subprocess (CPU/AS rlimits, no network, per-case flush) using *production's own
step splitter*, so the screened fire rate is the rate it will have at inference.
A checker that hangs, crashes, or blows its CPU limit is rejected on that basis.

```bash
CMD="python -m skills_construct.forge --stage screen --domain math" \
  LOG=logs/forge_screen.log sbatch scripts/slurm/cpu.sbatch
```

**emit** — writes accepted PFs to `skills/<domain>/generated_pfs.py` plus a
`SKILL.md` per PF carrying its screening numbers. **Not loaded by default**:
`dynamic_program_functions.py` chain-loads only the hand-written
`evidence_pfs.py`. `--register` adds the chain-load explicitly.

```bash
python -m skills_construct.forge --stage emit --domain math          # write only
python -m skills_construct.forge --stage emit --domain math --register
```

## The loop, and why each gate is where it is

**Three gates, in increasing cost, and only the last one is accuracy.**

| gate | cost | what it measures | what it cannot tell you |
|---|---|---|---|
| screen | seconds, CPU | fires on wrong ≫ fires on correct | whether the agent acts on the verdict |
| probe (tier 1) | GPU | *marginal* rescue / broke on the curated corpus | accuracy — that corpus is not a natural distribution |
| tier 2 | GPU | pass@1 before → after, per dataset | — |

Attribution in the probe is by **difference**, not by tags: two dispatch arms
run (admitted library, admitted + candidate) and a case counts for the
candidate only if the candidate *created* the intervention. Rescues the
admitted library would have produced anyway are not credited to it.

`broke > 0` disqualifies a candidate outright. The fallback-to-original
behaviour is the reason pf_select is safe to run at all; a PF that trades it
away is not worth a rescue.

**Refinement targets the two failure modes with opposite fixes.** A candidate
that fires on correct solutions needs a *narrower trigger*; a candidate that
fires on wrong solutions and changes nothing needs a *sharper verdict* — the
gap between "check your arithmetic in this step" and "the Observation for
`compute[7*13]` is 81, but 7\*13 = 91". Everything else is retired: never
firing, or breaking a correct solution, is not a tuning problem.

## The ledger

`data/skills_construct/forge/ledger_<domain>.jsonl` — one record per candidate, with its
screening numbers, probe result, accuracy result, and status:

```
proposed ─screen─> screened_out
         └────────> probed ─probe─> probed_out | refine | measured
                                     measured ─tier 2─> admitted | probed_out
```

Its job is to stop the next round from re-deriving dead ideas. The six "more
skills" PFs, the answer-drift anchor and the web relation probe were each
plausible on their face and each cost a run; the falsified list goes into the
proposer's prompt verbatim. A family where enough candidates have died is
reported as exhausted and skipped by later rounds.

```bash
python -m skills_construct.forge --stage ledger --domain math
```

**`admitted` is the only status that may enter a training rollout**, and it
requires a measured accuracy change — never a screening or probe number. After
admitting, add the new ids to `measure.BASE_PFS` so the next round measures its
candidates against the improved baseline rather than the old one.

A round that admits nothing is not a wasted round. Rounds that falsify a family
are what make later rounds cheap.

## Corpora

| domain | source | size |
|---|---|---|
| math | `data/llm_anchor/cases.jsonl` | 321 wrong / 146 correct |
| code | `anchor/eval_code_polished.build_cases()` | failing / passing solutions |
| web | replayed episodes (live-search quota pending) | — |

Questions come from full-text sources. Never rebuild a corpus from
`*_results.json`: the upstream evaluation writer truncates `question[:100]`
when saving, which silently feeds the proposer half a problem.

## After the gate

Screening is necessary, not sufficient. Before a generated PF is used in
training, measure accuracy before → after end to end:

```bash
sbatch scripts/slurm/math_polished.sbatch     # anchor/eval_polished_pfs.py
```

Report the accuracy change only — never population-weighted "net rollouts".
