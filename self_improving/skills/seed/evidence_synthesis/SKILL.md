---
skill_id: evidence_synthesis
name: Evidence Synthesis & Coverage
version: 1
priority: 0.8
error_category: reasoning
applicable_modes: [all]
applicable_phases: [think, read, answer]
system_summary: >
  Track evidence from multiple sources. Ensure all key claims are supported before answering.
phases:
  post_read:
    conditions: []
    priority_boost: 0.1
  pre_final:
    conditions: [always]
    priority_boost: 0.2
---

# Evidence Synthesis & Coverage

When answering from multiple documents, systematically track which facts come from which source. Ensure all key claims in your answer are supported by at least one read document.

## Detection Triggers
- Answer combines facts from multiple documents
- Key facts in the answer are not clearly traceable to a specific source
- Question requires synthesizing information across sources
- There are gaps between what was found and what the question asks

## Avoidance Strategies
- After each READ, mentally note the key facts extracted and their source
- Before FINAL, verify that every claim in your answer has a source
- If a claim lacks source support, search for additional evidence
- Do not mix facts from different entities or time periods
- Explicitly trace each answer component to its source document

## Phase: post_read
Note the key facts from this document. How do they relate to the question? Are there still missing pieces that require additional search?

## Phase: pre_final
Before answering, verify: (1) Every fact in your answer comes from a document you read, (2) You have not confused facts from different entities or sources, (3) There are no unsupported claims.

## Examples
### Example 1
**Scenario:** Two documents read about different people with similar names
**Wrong:** Mix facts from both documents in the answer
**Correct:** Carefully attribute each fact to the correct entity and document
