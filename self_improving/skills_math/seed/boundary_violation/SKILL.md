---
skill_id: boundary_violation
name: Domain & Boundary Awareness
version: 1
priority: 0.75
error_category: boundary_violation
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Respect the domain of the problem (integer constraints, positive values,
  defined expressions) and check boundary / edge values.
phases:
  pre_final:
    conditions: [has_domain_constraints]
    priority_boost: 0.3
    action: verify_domain
    action_params: {}
---

# Domain & Boundary Awareness

Math problems implicitly or explicitly restrict the solution domain: integer answers, positive reals, defined logarithms / denominators, etc. Violating the domain produces answers that are technically algebraic solutions but invalid for the problem.

## Detection Triggers
- Problem says "positive integer n" — candidate negative/fractional solutions must be rejected
- `log(x)` requires `x > 0`
- Square root `√(x-2)` requires `x ≥ 2`
- Denominator `1/(x-3)` requires `x ≠ 3`
- Probability must be in [0, 1]
- Triangle side lengths must satisfy triangle inequality
- AIME answer must be an integer 0..999

## Avoidance Strategies
- Write down all domain constraints before solving
- After finding candidate solutions, filter by domain
- For "minimum/maximum" questions: check endpoints of the domain explicitly
- For AIME: if your answer isn't an integer 0..999, something is wrong — re-derive
- Check that edge cases (n=0, n=1, largest allowed n) aren't optimal points you missed

## Phase: pre_final
Does your answer satisfy all implicit and explicit domain constraints? For AIME, is it a non-negative integer ≤ 999? If the question asks for "positive" or "integer" and your answer violates that, recompute.

## Examples
### Example 1
**Scenario:** Find n such that log(n-5) + log(n+5) = log(n² - 25).
**Wrong:** FINAL(5) — but log(0) is undefined; need n > 5
**Correct:** Equation holds for all valid n, so solution set is n > 5.

### Example 2
**Scenario:** AIME problem. Solved algebraically to get x = 1732.5.
**Wrong:** FINAL(1732.5) — AIME answers are integers
**Correct:** Re-examine — likely you made an error earlier; integer answers are required.
