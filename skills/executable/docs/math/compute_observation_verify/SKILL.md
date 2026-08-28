---
skill_id: compute_observation_verify
name: Compute Observation Verification
version: 1
priority: 0.95
error_category: arithmetic_slip
applicable_modes: [all]
applicable_phases: [think]
system_summary: >
  Re-evaluate the self-written Observation of every compute[...] action; if the
  written value is wrong, replace it with the true value and redo the step.
anchor:
  level: step
  trigger: "a `compute[...]` action whose Observation the model wrote itself"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [has_compute_action]
    action: verify_compute_observation
    action_params: {tolerance_rel: 0.01}
---

# Compute Observation Verification

Under the ReAct template the model writes its own `Observation:` after
`Action: compute[expr]`. Those values are wrong surprisingly often and the
error silently propagates (binom(7,2)*binom(7,1) written as 140, really 147 →
final 280 instead of 294).

## Anchor (step level)
Fires on the exact step containing `Action: compute[expr]` + `Observation: v`
when `expr` is numerically evaluable (sympy) and `v` differs from the true
value by more than 1% relative. Symbolic, truncated (`...`) or approximate
observations are skipped — the PF only acts on provable mismatches.

## Action change
MODIFY the step: rewrite `Observation: v` to the true value, then regenerate
the trajectory from this step. The injected evidence is
"the Observation written for compute[expr] is 'v', but the expression actually
equals t".

## Precision

Fires whenever sympy disagrees with the Observation the model wrote. Because
both the expression and its value came from the model, the claim is
self-contained — measure the rate on solutions that already pass before relying
on it.


## Example
compute[binom(4,2)*binom(6,2)] → Observation: 216 (true 90) → lottery
probability denominator wrong → answer 241 instead of 116.
