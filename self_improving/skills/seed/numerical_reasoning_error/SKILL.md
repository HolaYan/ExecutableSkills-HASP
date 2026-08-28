---
skill_id: numerical_reasoning_error
name: Numerical Claim Verification
version: 2
priority: 0.7
error_category: numerical_reasoning_error
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  Cross-check numbers and quantities in the answer against source documents.
phases:
  post_read:
    conditions: [read_has_numbers]
    priority_boost: 0.1
    action: verify_numerical_claims
    action_params: {}
  pre_final:
    conditions: [question_has_numbers]
    priority_boost: 0.2
    action: verify_numerical_claims
    action_params: {}
---

# Numerical Claim Verification

Verify that all numerical claims in the answer are directly supported by source documents. Numerical errors often arise from misreading tables, confusing similar numbers, or incorrect arithmetic.

## Detection Triggers
- Question asks for a specific number, count, measurement, or statistic
- Answer contains numbers not present in any read document
- Multiple numbers appear in the source that could be confused
- Question requires arithmetic or comparison of numerical values

## Avoidance Strategies
- Extract the exact number from the source text — do not round or approximate
- When multiple numbers appear, identify which one answers the specific question
- If arithmetic is required, show your work step by step
- Do not confuse units (e.g., millions vs. billions, km vs. miles)

## Phase: post_read
Note all numbers in this document. Distinguish which number answers the question vs. related but different figures.

## Phase: pre_final
NUMERICAL CHECK: Verify each number in your answer appears in a READ document. If calculation is needed, double-check the arithmetic.

## Examples
### Example 1
**Scenario:** Question asks for population of a city
**Wrong:** State "2.5 million" when the source says "2.3 million"
**Correct:** Copy the exact figure from the source: "2.3 million"

### Example 2
**Scenario:** Question asks how many awards a film won
**Wrong:** Count nominations instead of wins
**Correct:** Distinguish between "nominated for 8" and "won 3"
