---
skill_id: negation_oversight
name: Negation Awareness
version: 2
priority: 0.75
error_category: negation_oversight
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  Pay attention to negation words in the question to avoid answering the opposite.
anchor:
  level: final
  trigger: "the question carries a negation the reasoning never echoes"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  post_read:
    conditions: [question_has_negation]
    priority_boost: 0.1
    action: check_negation
    action_params: {}
  pre_final:
    conditions: [question_has_negation]
    priority_boost: 0.2
    action: check_negation
    action_params: {}
---

# Negation Awareness

Detect and correctly handle negation in the question. Negation oversight occurs when the agent answers what IS true instead of what is NOT true, missing words like "not", "never", "except".

## Detection Triggers
- Question contains negation words: not, never, except, other than, without, none
- Question asks for exclusions ("Which is NOT...")
- Question uses double negatives
- Answer contradicts the negation in the question

## Avoidance Strategies
- Highlight the negation word in the question before reasoning
- Restate the question to confirm you understand what is being asked
- If the question asks "which is NOT X", verify your answer is indeed not X
- Watch for subtle negation: "all except", "other than", "besides"

## Phase: post_read
NEGATION ALERT: The question has negation. As you read, track both what IS and what IS NOT stated, so you can answer the negation correctly.

## Phase: pre_final
NEGATION CHECK: Re-read the question's negation. Verify your answer respects it — are you answering what was asked, not the opposite?

## Examples
### Example 1
**Scenario:** "Which country did NOT sign the treaty?"
**Wrong:** List a country that DID sign the treaty
**Correct:** Identify a country explicitly stated as not signing

### Example 2
**Scenario:** "What was never achieved by the team?"
**Wrong:** Return an achievement the team DID accomplish
**Correct:** Return something explicitly stated as not achieved
