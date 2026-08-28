# Training × HASP PFs — integration notes

Goal: training-time rollouts use the SAME polished PF stack that evaluation
uses, and post-training evaluation is ALWAYS pf_select. Report results as
accuracy before → after only.

## How PFs enter training (no code duplication)

`training/common/skill_rollout.py::SkillRolloutRunner` already routes every
RS / distill rollout through the production `SkillAgentRunner`, loading the
skill library from the config's `rollout.skill_library_dir` and exec-importing
`{dir}/dynamic_program_functions.py`. HASP's domain libraries chain-load
`evidence_pfs.py` from there, so **pointing the config at `./skills/<domain>`
is the whole integration** for the PF side:

```yaml
rollout:
  skill_library_dir: "./skills/math"   # or ./skills/code, ./skills/web
                                       # resolved to skills/{textual,executable}/<domain>
```

Verified compatibility:
- `execute_program_functions` copies `step_context` and sets
  `step_context["thought"] = reasoning` → the math evidence PFs (which read
  `raw_reasoning` / `thought`) see the full reasoning during training rollouts.
- The runner's code-domain step_context already carries `question`,
  `entry_point`, `public_test_code`, `public_tests` → `spec_example_check`
  works as-is; web carries `all_read_contents` / `last_search_results_text` /
  `action_history` → `answer_grounding_check` works as-is.
- `backend="vllm"` (student rollouts) passes `teacher=None` → evidence PFs run
  their deterministic path; `backend="api"` (distill teacher) passes the API
  wrapper, which the PF helper hook can use.

## PATCHED: Case C at FINAL (`skill_agent_runner.py::_pre_dispatch_intervention`)

`pf_context_injections` are only consumed by the NEXT observation
(`_format_observation`). At FINAL there is no next step, so an evidence
injection at FINAL used to be silently dropped and the episode ended with the
original answer — training rollouts diverged from pf_select eval (which does a
Turn-2 revision). The HASP copy of `skill_agent_runner.py` now converts
`FINAL + non-empty pf_injections` into the base runner's existing `RETRY`
path: the injections come back as the next `Observation:` and the model
produces ONE revised FINAL (once per episode, guarded by
`ep_data["_final_revision_done"]`; disable with
`skill_config.final_revision_on_inject: false`). If the PF already rewrote
the action (Case B), revision is skipped — the rewrite wins. This matches
pf_select eval's Case-C Turn-2 exactly, including fallback: the model may
re-commit its original answer.

## Rejection sampling: the accepted code gate, wired into the filter

`training/rejection_sampling/verifier.py::SpecExampleVerifier` — the
training-time form of the accepted inference config (code3: best-of-n
accepting the first sample that passes the spec's own examples,
the same acceptance rule used at inference):

- runs the spec's `>>>` / `assert` examples and `public_test_code` on each
  candidate in the sandbox (zero API cost);
- fails → score 0 (rejected); passes → 1.0; no examples → 0.5 (neutral, kept
  and ranked by the other signals).

**How it is invoked.** RS does not build a verifier from a `verifier:` config
block — nothing in `rejection_sampling/` reads one (only `grpo/train.py`
constructs a verifier, and it imports `TeacherVerifier` directly). The gate is
therefore wired into the episode filter itself:

- `_filter_by_episode_correctness(..., spec_example_gate=bool)` runs the
  verifier on each kept episode's `final_answer` and drops score-0 episodes,
  logging `spec_example_gate: rejected N episodes`;
- `run_rs_iteration` passes `cfg["filter"].get("spec_example_gate", domain == "code")`
  — on by default for code, overridable per config;
- `flatten_to_per_step` now carries `entry_point` and `public_test_code` on
  every row so the verifier has the spec context it needs.

Math RS keeps `filter.require_exact_match` against gold (training data has
gold answers; no verifier needed).

## Configs

| config | base | changes |
|---|---|---|
| `configs/training/H1_math_rs.yaml` | math rejection sampling | `skill_library_dir: ./skills/math` (evidence PFs) |
| `configs/training/H2_code_rs.yaml` | code rejection sampling | `skill_library_dir: ./skills/code`; `filter.spec_example_gate: true` |

Model: Qwen2.5-7B-Instruct. Web configs wait for live-search quota.

## Post-training evaluation — always pf_select

`training/scripts/eval_pf_select_hasp.sh <ckpt> <tag> [datasets]` submits the
canonical n=64 protocol (HASP `pf_select/eval_models.py`: skills_off pass +
pf_select pass with the polished library) on 1×L40S. Output:
`data/model_eval/<tag>/`. Never report population-weighted deltas; report
accuracy before → after.

## Which training paths run here

All of them import cleanly — `self_improving/` is vendored here rather than
being an external checkout, so nothing needs a second repository on the path.
What differs is what each one needs at *run* time.

| path | needs | notes |
|---|---|---|
| **rejection sampling** (`rejection_sampling/`, `common/skill_rollout.py`, `sft/trainer.py`) | prompts + raw benchmark files | what `configs/training/H1,H2` use |
| SFT (`sft/train.py`) | a finished proposal round to load candidates from | reads `CandidateSkill` / `ReviewResult` |
| GRPO (`grpo/`) | a reward config; signals score the trajectories | reads the trajectory schema |
| distillation (`distill/`) | a hosted teacher, and prompts | teacher rollouts go through the API backend |
| closed loop (`closed_loop/`) | everything above, plus `wandb` | the most expensive to stand up |
| bootstrap builders (`scripts/build_bootstrap_*`) | a hosted teacher and the raw benchmark files | they generate the shared prompt/SFT data the others consume |

All of them default their skill library to `skills/`, the library this
repository ships.

## Two ways the library grows

There are two evolution paths here, and they are not the same mechanism. Both
read the four credit signals; they disagree about what drives a proposal and
what lets one in.

| | `evolving/` | `closed_loop/` + `self_improving/` |
|---|---|---|
| granularity | every N training steps | every N epochs |
| what triggers a proposal | the current checkpoint's own failures on a held-out set | a pseudo-gradient built from the signals over a run's trajectories |
| what admits one | a structural gate, then a precision screen against solutions that were already correct | a five-dimension review by a model, then a compile check |
| what it costs | CPU, on rollouts the cycle already produced | a hosted reviewer and a full pipeline |

`evolving/` is the tighter loop and the stricter gate; the closed loop is the
one that reasons about *why* a skill should exist, using the signals as credit
rather than as a filter. Neither subsumes the other, which is why both are
here — see `evolving/README.md` for the signals themselves.

