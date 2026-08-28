---
skill_id: repetition_circuit_breaker
name: Repetition Circuit Breaker
version: 1
priority: 0.6
error_category: budget_exhaustion
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  When the FINAL answer degenerates into a repeated line / token run (a common
  budget-collapse symptom), collapse it to the single best value.
anchor:
  level: final
  trigger: "the committed answer is a degenerate repetition"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.2
    action: collapse_repetition
    action_params: {}
---

# Repetition Circuit Breaker

Budget-exhaustion failures often degenerate into the model repeating one line
or token until the cap. When such a degenerate string still reaches FINAL,
collapse it down to the intended answer.

## Detection Triggers
- FINAL arg has many lines that are nearly all identical
- A long token run with very few distinct tokens

## Avoidance Strategies
- If you notice you are repeating yourself, STOP and commit the answer once.
- Submit a single clean value, not a repeated block.

## Phase: pre_final
Are you repeating the same line? Commit the answer once and stop.

## Examples
### Example 1
**Wrong:** `Action: FINAL(7\n7\n7\n7\n7\n7)`
**Correct:** `Action: FINAL(7)`
