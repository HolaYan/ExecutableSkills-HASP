---
skill_id: simplification_incomplete
name: Full Simplification
version: 1
priority: 0.70
error_category: simplification_incomplete
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Final answer must be fully simplified: fractions in lowest terms,
  radicals rationalized / simplified, expressions factored to simplest form.
phases:
  pre_final:
    conditions: [answer_not_simplified]
    priority_boost: 0.4
    action: verify_simplification
    action_params: {}
---

# Full Simplification

Competition grading usually requires the answer in a canonical, fully-simplified form. Answers like `6/9` or `√8` are marked wrong even if numerically correct.

## Detection Triggers
- Fractions where `gcd(numerator, denominator) > 1`
- Radicals: `√(4·3)` should be `2√3`, not left as is
- `a²b/ab²` should be `a/b`
- Unresolved composite numerics: `20! / 19!` = 20
- Unevaluated functions on constants: `cos(0)` = 1, `log(1)` = 0
- Compound fractions: `(a/b)/(c/d)` = `ad/bc`

## Avoidance Strategies
- Always reduce fractions: compute gcd, divide
- Radicals: factor under the root into squarefree × perfect-square; pull out perfect-square part
- Check if exponents simplify: `x^a / x^b = x^(a-b)`
- For AIME: answer is an integer — compute it fully, no fractions/radicals
- For MATH-500: check the expected format (often "a/b in lowest terms", or "a + b√c")

## Phase: pre_final
Is your answer fully simplified? Fractions in lowest terms? Radicals in simplest form? Compute any remaining evaluable expressions.

## Examples
### Example 1
**Scenario:** MATH-500 asks for area of a triangle.
**Wrong:** FINAL(`\sqrt{12}`)
**Correct:** `\sqrt{12}` = `\sqrt{4·3}` = `2\sqrt{3}` → FINAL(`2\sqrt{3}`)

### Example 2
**Scenario:** AIME problem; intermediate value is 735/210.
**Wrong:** FINAL(735/210) — AIME wants integer
**Correct:** 735/210 = 7/2 = 3.5 — not integer, means earlier error; recompute.
