---
skill_id: arithmetic_slip
name: Arithmetic Verification
version: 1
priority: 0.85
error_category: arithmetic_slip
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Catch simple arithmetic mistakes (addition, multiplication, fraction
  operations) that propagate through otherwise correct reasoning.
phases:
  pre_final:
    conditions: [has_numeric_computation]
    priority_boost: 0.4
    action: verify_arithmetic
    action_params: {}
---

# Arithmetic Verification

Competition-level math solutions often have correct structure but wrong final answers because of a single arithmetic slip at an intermediate step. Catch these before finalizing.

## Detection Triggers
- Final answer depends on a chain of ≥3 numerical operations
- Computation crosses place-value boundaries (carrying, borrowing)
- Fraction combinations or factor cancellations near the end
- Large-number multiplication / modular arithmetic
- The final step is "so the answer is X" without a recomputation

## Avoidance Strategies
- Re-evaluate the key arithmetic chain once before emitting FINAL
- For AIME-style answers (integer 0-999): sanity check plausibility
- Prefer symbolic simplification, then substitute numbers only at the end
- When reducing fractions, verify gcd explicitly
- For modular expressions, reduce mod m at every step to keep numbers small

## Phase: pre_final
Before emitting the final answer, re-derive the critical arithmetic step one more time from scratch. If the two computations disagree, investigate which one was wrong.

## Examples
### Example 1
**Scenario:** Problem asks for sum S = 1 + 2 + ... + 99. Solution uses formula n(n+1)/2 = 99·100/2.
**Wrong:** FINAL(4905) — arithmetic slip, 99·100/2 = 4950 not 4905
**Correct:** Verify: 99·100 = 9900, 9900/2 = 4950 → FINAL(4950)

### Example 2
**Scenario:** Reducing a fraction 126/189 in final step.
**Wrong:** FINAL(6/9) — incorrect, gcd(126,189)=63, 126/63=2, 189/63=3 → 2/3
**Correct:** Compute gcd explicitly, then divide.
