---
skill_id: language_barrier
name: Cross-Language Awareness
version: 2
priority: 0.5
error_category: language_barrier
applicable_modes: [all]
applicable_phases: [read]
system_summary: >
  Handle non-English names, transliterations, and cross-language content carefully.
anchor:
  level: step
  trigger: "read content with more than 50 non-ASCII characters"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_read:
    conditions: [read_has_non_english_content]
    priority_boost: 0.1
    action: check_language_handling
    action_params: {}
  pre_final:
    conditions: [answer_has_transliteration]
    priority_boost: 0.15
    action: check_language_handling
    action_params: {}
---

# Cross-Language Awareness

Handle non-English content, transliterations, and foreign names carefully. Language barrier errors occur when the agent misinterprets foreign text or confuses transliteration variants.

## Detection Triggers
- Document contains non-English text or mixed-language content
- Question involves names with multiple transliteration variants
- Search results are in a language different from the question
- Proper nouns have different spellings across sources

## Avoidance Strategies
- Be aware that names may have multiple valid transliterations
- Try both original and translated/transliterated forms when searching
- Do not assume a single spelling is the only correct one
- Focus on extracting factual data from foreign-language sources

## Phase: post_read
LANGUAGE NOTE: This document contains non-English content. Be careful with spelling variants and transliterated names.

## Phase: pre_final
TRANSLITERATION CHECK: Verify your answer uses the correct spelling. If names appear in multiple forms across sources, use the most standard transliteration.

## Examples
### Example 1
**Scenario:** Sources use "Dostoevsky" and "Dostoyevsky"
**Wrong:** Treat these as two different authors
**Correct:** Recognize both as transliterations of the same name

### Example 2
**Scenario:** Results mix Pinyin and traditional names
**Wrong:** Fail to connect "Beijing" with "Peking"
**Correct:** Recognize both refer to the same city
