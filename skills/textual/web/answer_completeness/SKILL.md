---
skill_id: answer_completeness
name: Answer Completeness & Scope Check
version: 1
priority: 0.75
error_category: answer_completeness
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  Ensure answer addresses the exact scope and all parts of the question.
anchor:
  level: final
  trigger: "a multi-part question answered in fewer than three words"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [always]
    priority_boost: 0.2
    action: verify_answer_relevance
    action_params: {}
---

# Answer Completeness & Scope Check

Verify that the answer addresses all parts of the question at the correct scope — not broader, narrower, or incomplete. This combines scope alignment and partial answer detection.

## Detection Triggers
- Answer discusses a broader category than what was asked
- Answer addresses only part of a multi-faceted question
- Question specifies a particular time, place, or context that the answer ignores
- Question contains multiple sub-questions (joined by "and", multiple "?")
- Answer is suspiciously short for a complex question

## Avoidance Strategies
- Re-read the question after formulating your answer to verify alignment
- Check that your answer matches any qualifiers (time period, location, specific aspect)
- Count the number of sub-questions or aspects before answering
- For list questions, verify you have the requested number of items
- If the question asks about X in context Y, ensure your answer is about X in Y

## Phase: pre_final
COMPLETENESS & SCOPE CHECK:
1. Re-read the question and note all qualifiers (time, place, specific aspect).
2. If multi-part, list all sub-questions and verify each is addressed.
3. Check that your answer matches the exact scope — not too broad or narrow.

## Examples
### Example 1
**Scenario:** "Who was the president of France during WWI?"
**Wrong:** Return the current president of France
**Correct:** Return the president specifically during WWI (1914-1918)

### Example 2
**Scenario:** "When and where was the treaty signed?"
**Wrong:** Only provide the date without the location
**Correct:** Provide both the date and the location

### Example 3
**Scenario:** "What is the population of Tokyo proper?"
**Wrong:** Return the population of the Greater Tokyo Area
**Correct:** Return the population of Tokyo proper (the city, not the metro area)
