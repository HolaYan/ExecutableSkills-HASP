---
skill_id: reading_comprehension_error
name: Careful Reading and Extraction
version: 1
priority: 0.7
error_category: reading_comprehension_error
applicable_modes: [all]
applicable_phases: [read, answer]
system_summary: >
  Re-read key sentences carefully; quote the exact text that answers the question.
anchor:
  level: step
  trigger: "a READ of dense content — ten or more numbers or fifteen or more names"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_read:
    conditions: [read_has_multiple_entities]
    priority_boost: 0.2
  pre_final:
    conditions: [always]
    priority_boost: 0.0
    action: verify_reading_comprehension
    action_params: {}
---

# Careful Reading and Extraction

When reading documents, carefully extract the exact information requested. Re-read key sentences, distinguish between adjacent entities, and quote the source text to confirm understanding.

## Detection Triggers
- Document contains multiple similar entities or facts close together
- The answer requires extracting a specific detail from a longer passage
- Document discusses multiple people, places, or dates that could be confused
- Question asks about a specific attribute of an entity mentioned among many

## Avoidance Strategies
- Re-read the key sentence containing the answer before extracting
- Quote the exact phrase from the document that answers the question
- Distinguish carefully between adjacent entities (e.g., father vs son, city vs country)
- Do not rely on skimming — read the relevant paragraph fully
- When a document lists multiple items, identify exactly which one matches the query
- If the document is long, use SUMMARY first to locate the relevant section, then READ for details

## Phase: post_read
CAREFUL: This document mentions multiple entities that could be confused.
1. Identify the SPECIFIC entity the question asks about.
2. Find the sentence that directly attributes the requested information to that entity.
3. Quote the exact phrase before extracting your answer.
4. Watch for adjacent names (e.g., father vs son, city vs country).

## Phase: pre_final
Before answering, mentally quote the sentence from the document that supports your answer. If you cannot recall a specific sentence, READ the document again.

## Examples
### Example 1
**Scenario:** Document mentions 'Charles Tupper Sr.' and 'Charles Tupper' — question asks about the father
**Wrong:** Extract 'Charles Tupper' without distinguishing Sr. from Jr.
**Correct:** Note the 'Sr.' suffix and correctly identify Charles Tupper Sr. as the father

### Example 2
**Scenario:** Document says 'roof of One Times Square' — question asks where the ball drops
**Wrong:** Fail to extract the location from the sentence
**Correct:** Quote: 'roof of One Times Square' -> answer is 'One Times Square'
