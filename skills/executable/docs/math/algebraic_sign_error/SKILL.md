---
skill_id: algebraic_sign_error
name: Sign Tracking
version: 1
priority: 0.80
error_category: algebraic_sign_error
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Track signs through algebraic manipulations: distribution, inequality
  flipping, square-root branches, subtraction of polynomials.
anchor:
  level: step
  trigger: "the reasoning contains a negative sign, a radical or an inequality"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  pre_final:
    conditions: [has_algebra]
    priority_boost: 0.35
    action: verify_signs
    action_params: {}
---

# Sign Tracking

Sign errors are the most common silent failure in algebra competitions. They often flip an answer by -1× or invert an inequality's direction.

## Detection Triggers
- Distributing a negative: `-(a-b)` must become `-a+b`, not `-a-b`
- Multiplying/dividing an inequality by a negative → **flip** the inequality
- Taking √ of `x²=k` → both `x=√k` and `x=-√k` branches exist
- Subtracting polynomials `(ax^2 + bx + c) - (dx^2 + ex + f)` → every term in the second group flips sign
- Squaring introduces extraneous solutions, opposite-sign pairs
- `|x|` produces case split on sign

## Avoidance Strategies
- Keep a "sign ledger" beside your working when distributing negatives
- At each inequality step, explicitly note: "multiplied by negative → flipped"
- For √: always write ± and prune invalid branches by the problem constraints
- After squaring or taking absolute value: plug answer back into the ORIGINAL equation to check

## Phase: pre_final
Review every sign flip in your solution. Especially check (a) negative distributions, (b) inequality directions, (c) radical branches. If any is ambiguous, re-derive that step.

## Examples
### Example 1
**Scenario:** Solve `-(x-3) > 2`
**Wrong:** `-x-3 > 2` → `x < -5` (sign of -3 wrong, then didn't flip inequality)
**Correct:** `-x+3 > 2` → `-x > -1` → `x < 1` (flipped when multiplying by -1)

### Example 2
**Scenario:** Solve `x² = 9`
**Wrong:** FINAL(3) — missed the negative branch
**Correct:** x ∈ {3, -3}; pick branch consistent with context (e.g., "x is a length" → 3).
