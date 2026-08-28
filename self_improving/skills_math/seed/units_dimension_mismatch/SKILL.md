---
skill_id: units_dimension_mismatch
name: Dimensional Consistency
version: 1
priority: 0.60
error_category: units_dimension_mismatch
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  When a problem has physical / geometric units (length, area, volume,
  probability, ratio), ensure dimensional consistency throughout.
phases:
  pre_final:
    conditions: [has_units]
    priority_boost: 0.25
    action: verify_dimensions
    action_params: {}
---

# Dimensional Consistency

Problems mixing length (cm), area (cm²), volume (cm³), time (s), or abstract ratios need dimensional checks. Adding unlike units is a red flag.

## Detection Triggers
- Formula combines `L²` with `L` directly (perimeter + area = nonsense)
- Probability answer > 1 or < 0
- Counting answer is non-integer
- Combinations `nCk` applied where order matters (permutations needed)
- Work / energy / power confusion in physics contexts
- Percentage interpreted as fraction without division by 100

## Avoidance Strategies
- Carry units alongside each intermediate expression
- At each step verify LHS unit = RHS unit
- Check final unit matches what question asks for (volume? length? count?)
- For probability: ensure result ∈ [0, 1]
- For counts: ensure result is a positive integer
- Convert units explicitly when required (meters ↔ cm, radians ↔ degrees)

## Phase: pre_final
Does your answer have the correct units? Is a probability in [0,1]? Is a count a non-negative integer? Are lengths and areas not mixed? If any check fails, recompute.

## Examples
### Example 1
**Scenario:** Probability problem; compute P(event).
**Wrong:** FINAL(1.25) — probabilities can't exceed 1
**Correct:** There's a counting error (overcounted favorable outcomes). Recount.

### Example 2
**Scenario:** AIME: find the area of a region.
**Wrong:** Answer = perimeter length (e.g. 24) instead of area (should be L²)
**Correct:** Re-read question to confirm what's asked (area vs perimeter).
