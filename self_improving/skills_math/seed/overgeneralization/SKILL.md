---
skill_id: overgeneralization
name: Formula Applicability Check
version: 1
priority: 0.70
error_category: overgeneralization
applicable_modes: [all]
applicable_phases: [think]
system_summary: >
  Check the preconditions of a formula before applying it: AM-GM needs
  positive reals, Pythagoras needs right angles, series convergence tests
  have domain requirements.
phases:
  pre_final:
    conditions: [applied_formula]
    priority_boost: 0.25
    action: verify_formula_conditions
    action_params: {}
---

# Formula Applicability Check

Competition problems often tempt you to apply a familiar formula whose preconditions the problem silently violates.

## Detection Triggers
- AM-GM: requires **non-negative** real numbers
- Pythagorean theorem: requires a **right triangle**
- Geometric series formula `a/(1-r)`: requires `|r| < 1`
- L'Hôpital: requires `0/0` or `∞/∞` indeterminate form
- Quadratic formula: for `ax² + bx + c = 0` with `a ≠ 0`
- Binomial theorem over integers vs reals
- Cauchy-Schwarz / Jensen: convexity conditions

## Avoidance Strategies
- Before applying a named result, list its preconditions and verify each
- For inequalities with equality cases, check when equality holds
- If the problem has a "twist" (absolute values, floor function, etc.), generic formulas may fail
- Prefer first-principles derivation when preconditions are unclear
- For AIME: "elegant" solutions often rely on specific problem structure

## Phase: pre_final
What formulas / theorems did you apply? For each, does the problem meet its preconditions? If not, redo that step from scratch.

## Examples
### Example 1
**Scenario:** Minimize `f(x, y) = x² + y²` subject to `x + y = 10`.
**Wrong:** Apply AM-GM to `x²+y² ≥ 2|xy|` — correct but doesn't use constraint
**Correct:** AM-GM on `x + y ≥ 2√(xy)` requires `x, y ≥ 0`; but constraint holds and gives `xy ≤ 25`, then `x²+y² = (x+y)² - 2xy = 100 - 2xy ≥ 50`.

### Example 2
**Scenario:** Sum `1 + 2 + 4 + 8 + ...`
**Wrong:** Use formula `a/(1-r)` with `a=1, r=2` → `-1`
**Correct:** `|r| = 2 ≥ 1`, so the series diverges. Formula doesn't apply.
