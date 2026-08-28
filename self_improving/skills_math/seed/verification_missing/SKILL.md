---
skill_id: verification_missing
name: Solution Verification
version: 1
priority: 0.90
error_category: verification_missing
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Before finalizing, plug the candidate answer back into the original
  problem to verify it satisfies all constraints.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.5
    action: verify_solution
    action_params: {}
---

# Solution Verification

The single most reliable technique to catch errors: substitute the candidate answer back into the original equation / problem statement and confirm every constraint is satisfied.

## Detection Triggers
- Candidate answer obtained through a multi-step derivation
- Algebraic manipulation involved squaring, dividing, or substituting
- Answer comes from solving a system of equations
- Problem has multiple constraints; only some were used in deriving
- Counting / combinatorics answer where direct enumeration is feasible

## Avoidance Strategies
- Plug back into the ORIGINAL equation, not a derived one
- If equation involved squaring, reject extraneous solutions
- Check that the answer satisfies ALL constraints, not just one
- For counting: do a small-case sanity check (n=1, n=2)
- For optimization: verify that the candidate is indeed extremal (check nearby values)

## Phase: pre_final
Substitute your candidate answer back into each constraint of the original problem. Does it satisfy every one? If not, recompute.

## Examples
### Example 1
**Scenario:** Solve `√(x+6) = x`. Candidate solution x=3 and x=-2.
**Wrong:** FINAL({-2, 3}) without checking
**Correct:** Check x=-2: `√4 = 2 ≠ -2`. Extraneous. Check x=3: `√9=3 ✓`. FINAL(3).

### Example 2
**Scenario:** System: x + y = 10, x² - y² = 40. Found x=7, y=3.
**Wrong:** FINAL(7, 3) without substitution
**Correct:** Verify: 7+3=10 ✓, 49-9=40 ✓. Confirmed.
