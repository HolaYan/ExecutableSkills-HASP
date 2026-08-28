---
skill_id: reasoning_error
name: Step-by-Step Reasoning Verification
version: 1
priority: 0.8
error_category: reasoning_error
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Verify each logical step explicitly before concluding — especially for comparisons and calculations.
phases:
  post_read:
    conditions: [read_has_numbers]
    priority_boost: 0.1
  pre_final:
    conditions: [always]
    priority_boost: 0.1
    action: verify_reasoning_steps
    action_params: {}
---

# Step-by-Step Reasoning Verification

Verify each step of the logical reasoning chain before committing to an answer. Double-check calculations, comparisons, and multi-step inferences.

## Detection Triggers
- Question requires multi-step logical reasoning
- Question involves numerical computation or comparison
- Question asks for aggregation (oldest, largest, most, first)
- Answer depends on combining facts from multiple sources

## Avoidance Strategies
- Explicitly state each reasoning step in your Thought before concluding
- Double-check numerical calculations — recompute if the result seems surprising
- For comparison questions, ensure you are comparing the correct attributes
- Verify that intermediate conclusions logically follow from the evidence
- For geographic/nationality questions, verify the specific entity (city vs country, region vs nation)
- When aggregating information, list all candidates before selecting the answer

## Phase: post_read
This document contains numerical data. Before proceeding:
1. Extract the specific numbers relevant to the question.
2. If a calculation is needed, write it out step by step.
3. Double-check units and conversions.
4. If comparing values, list all candidates explicitly before selecting.

## Phase: pre_final
REASONING CHECK: Before answering, trace your logic chain:
1. What facts did you find? (cite doc_ids)
2. What inference connects those facts to your answer?
3. Are there any unstated assumptions? Verify them.

## Examples
### Example 1
**Scenario:** Question: 'Were both directors of Film A and Film B from the same country?'
**Wrong:** Find both directors are American and conclude 'different countries'
**Correct:** Explicitly list: Director A = American, Director B = American, therefore 'same country'

### Example 2
**Scenario:** Calculating marathon pace requires unit conversion
**Wrong:** Skip unit conversion step and get wrong answer (17000 vs 17)
**Correct:** Show each conversion step: total_time / distance = pace_per_unit, then convert units
