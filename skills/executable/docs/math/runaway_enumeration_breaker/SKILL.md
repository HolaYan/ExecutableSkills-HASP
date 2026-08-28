---
skill_id: runaway_enumeration_breaker
name: Runaway Enumeration Breaker
version: 1
priority: 0.9
error_category: premature_final
applicable_modes: [all]
applicable_phases: [think]
system_summary: >
  Detect a non-converging enumeration (>=60% repeated line patterns) and make the solver switch to a structural argument and commit an answer.
anchor:
  level: step
  trigger: "a step where most lines repeat the same pattern"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [has_repeated_enumeration]
    action: break_runaway
---
# Runaway Enumeration Breaker
The stall channel's signature: 20k–115k-char rollouts that are 88–97% repeated lines ('Try t=277 … ≠ 0'). Anchor = the step where repetition density crosses the threshold; the repeated template is quoted as evidence.

**Status (2026-08-22):** no population in the 4B eval data (its stalls stop at `Action:`); designed from Qwen3-8B non-thinking runs (88–97% repeated lines).
