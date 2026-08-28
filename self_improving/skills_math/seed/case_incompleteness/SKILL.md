---
skill_id: case_incompleteness
name: Complete Case Analysis
version: 1
priority: 0.80
error_category: case_incompleteness
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Ensure case-analysis arguments cover all disjoint cases exactly once —
  no missing case, no double counting.
phases:
  pre_final:
    conditions: [has_case_analysis]
    priority_boost: 0.35
    action: verify_case_coverage
    action_params: {}
---

# Complete Case Analysis

Problems solved by splitting into sub-cases fail when a case is missed or two cases overlap. Every case partition must be **exhaustive and mutually exclusive**.

## Detection Triggers
- Absolute value yields `x ≥ 0` vs `x < 0` — both must be handled
- Even/odd, prime/composite partitions
- "Largest" / "smallest" / "otherwise" quantifiers
- Piecewise functions; parameter sign (a > 0 vs a < 0 vs a = 0)
- Combinatorics: pairs split into (a<b, a=b, a>b)
- Modular arithmetic: case split on remainder
- Triangle inequality: which side is longest?

## Avoidance Strategies
- Enumerate cases explicitly: "Case 1 / Case 2 / Case 3" in working
- Verify union of cases = original domain (no gaps)
- Verify pairwise intersections are empty (or accounted for)
- Pay attention to **boundary** cases (a=0, a=b, etc.) — easy to drop
- For counting: always ask "did I double-count any case?"

## Phase: pre_final
List every case you considered. Ask: (1) does their union equal the full domain? (2) are any two cases overlapping? If either answer isn't satisfied, revise.

## Examples
### Example 1
**Scenario:** Solve `|x-2| + |x+1| = 5`
**Wrong:** Only consider `x ≥ 2` → miss solutions in (-∞, -1) and [-1, 2]
**Correct:** Case 1: x≥2, Case 2: -1≤x<2, Case 3: x<-1 — solve each, union solutions

### Example 2
**Scenario:** Count 3-digit numbers with digits strictly increasing
**Wrong:** Forget to exclude digit=0 from leading position
**Correct:** Enumerate digits ∈ {1..9}³ with d1<d2<d3 — C(9,3) = 84
