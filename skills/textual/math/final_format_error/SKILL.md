---
skill_id: final_format_error
name: Answer Format Compliance
version: 1
priority: 0.85
error_category: final_format_error
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Emit the final answer in the competition's expected format:
  `\boxed{X}` for MATH-500; integer 0-999 for AIME; fully simplified.
anchor:
  level: final
  trigger: "the committed answer is not in the dataset's expected form (skipped for expression answers and Game-of-24)"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.5
    action: verify_final_format
    action_params: {}
---

# Answer Format Compliance

The final answer must match the competition's grading format exactly, otherwise it's marked wrong even with correct reasoning.

## Detection Triggers
- Missing `\boxed{}` wrapper on the numerical answer
- Trailing commentary after the `\boxed{}` ("= \boxed{42} pounds")
- AIME answer outside 0..999
- Non-integer answer for AIME
- Fraction not in lowest terms
- Radical not simplified (`√8` instead of `2√2`)
- Answer stated in words instead of number
- Multiple candidate answers ("either 3 or 5")

## Avoidance Strategies
- **MATH-500**: wrap final answer in `\boxed{...}`. Just one box.
- **AIME24/25**: integer 0-999; no units, no wrapper needed in the `\boxed{}`
- Do NOT include units in the boxed expression ("\boxed{24}", not "\boxed{24 cm}")
- Simplify fractions / radicals first
- Exactly one answer — commit to it; no "or" branches
- Remove trailing periods, newlines, or commentary

## Phase: pre_final
Is your answer in canonical form? Wrapped in `\boxed{...}` for MATH-500? Integer for AIME? Fully simplified? No units inside the box?

## Examples
### Example 1
**Scenario:** AIME problem with answer 42.
**Wrong:** `Action: FINAL(42 units)` or `Action: FINAL(the answer is 42)`
**Correct:** `Action: FINAL(\boxed{42})`

### Example 2
**Scenario:** MATH-500 asks for tangent of angle.
**Wrong:** `Action: FINAL(sqrt(3))`
**Correct:** `Action: FINAL(\boxed{\sqrt{3}})`
