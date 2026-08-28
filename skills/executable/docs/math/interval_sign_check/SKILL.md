---
skill_id: interval_sign_check
name: Interval Sign Check
version: 1
priority: 0.85
error_category: algebraic_sign_error
applicable_phases: [think]
system_summary: >
  In a step that analyses the sign of expressions over intervals, evaluate each
  sign claim at a test point inside the interval; a wrong sign is provable evidence.
anchor:
  level: step
  trigger: "a sign claim about an expression over an interval"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [has_interval_sign_analysis]
    action: test_point_signs
---
# Interval Sign Check
algebra family (amc23_37: 901 written as 900 / 902): "x < 0: −x > 0, x−2 < 0 → −x(x−2) > 0"
is false (at x = −1 it is −3). Anchor = a line with an interval condition
(`x < 0`, `0 < x < 2`, fraction bounds allowed) followed by atomic claims
`$expr ⋚ 0$`; evidence = sympy value at a test point. Provable, zero LLM.
Action: regenerate from this step with the test-point value injected.
