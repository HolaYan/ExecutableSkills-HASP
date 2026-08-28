---
skill_id: format_extraction_error
name: Precise Format Extraction
version: 3
priority: 0.7
error_category: format_extraction_error
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Answer with ONLY the requested entity — no extra text, explanation, or verbose sentences.
anchor:
  level: final
  trigger: "the answer carries a prefix, suffix or markdown wrapper around the value"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  post_read:
    conditions: [question_is_factoid]
    priority_boost: 0.1
    action: postprocess_answer
    action_params:
      strip_prefix: true
      extract_entity: true
  pre_final:
    conditions: [answer_is_verbose]
    priority_boost: 0.2
    action: postprocess_answer
    action_params:
      strip_prefix: true
      extract_entity: true
---

# Precise Format Extraction

Ensure the final answer matches the expected format — extract the core entity/value without extra explanation.

## Detection Triggers
- Question asks for a specific name, date, number, or entity
- Question uses 'who', 'what', 'when', 'where' expecting a concise answer
- Answer is a full sentence when a single word/phrase would suffice
- Answer starts with "Yes/No" followed by unnecessary explanation
- Answer is more than 10 words for a factoid question
- Answer includes reasoning or caveats instead of just the answer

## Avoidance Strategies
- Answer with ONLY the requested entity — no full sentences
- Match the granularity expected (e.g., 'Paris' not 'Paris, France')
- Strip prefixes like 'The answer is...' — output only the core answer
- For yes/no questions, answer exactly 'yes' or 'no'
- For factoid questions, provide only the specific answer (name, number, date)
- If the question asks "who", answer with just the name

## Phase: post_read
Note the EXACT form of the answer: name, date, number, or place. You will need only the core entity for your final answer.

## Phase: pre_final
FORMAT CHECK: Output ONLY the entity requested. No 'The answer is...', no explanations. For names use the common form; for yes/no answer exactly that.

## Examples
### Example 1
**Scenario:** "What is the capital of Guyana?" Gold: 'Georgetown'
**Wrong:** FINAL('The capital of Guyana is Georgetown.')
**Correct:** FINAL('Georgetown')

### Example 2
**Scenario:** "When did Richard Nixon die?" Gold: '22-Apr-94'
**Wrong:** FINAL('Richard Nixon died on April 22, 1994.')
**Correct:** FINAL('April 22, 1994')
