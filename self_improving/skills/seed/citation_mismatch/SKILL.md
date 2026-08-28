---
skill_id: citation_mismatch
name: Citation Verification
version: 2
priority: 0.6
error_category: citation_mismatch
applicable_modes: [all]
applicable_phases: [think, read]
system_summary: >
  Verify proper nouns and claims in the answer actually appear in cited sources.
phases:
  post_read:
    conditions: [read_has_multiple_entities]
    priority_boost: 0.1
    action: verify_citations
    action_params: {}
  pre_final:
    conditions: [answer_has_proper_nouns]
    priority_boost: 0.2
    action: verify_citations
    action_params: {}
---

# Citation Verification

Verify that the specific entities, names, and facts cited in the answer are actually present in the documents that were read. Citation mismatch errors occur when the answer references information that was never in the source documents.

## Detection Triggers
- Answer mentions proper nouns not found in any read document
- Answer attributes information to a source that doesn't contain it
- Specific claims in the answer don't match what was actually read
- Answer combines information from different sources incorrectly

## Avoidance Strategies
- Before finalizing, cross-check every proper noun in your answer against read documents
- Do not introduce entities that weren't in your sources
- If you need to cite a specific source, verify the claim is actually in that source
- Be careful not to mix up information from different documents

## Phase: post_read
As you read this document, note the exact proper nouns and key facts. You will need to verify your answer uses only names and claims from documents you have actually read.

## Phase: pre_final
CITATION CHECK: Verify all proper nouns and specific facts in your answer.
1. List the key entities and claims in your answer.
2. For each, confirm it appears in a document you READ.
3. Remove or search for verification of any unsupported claims.

## Examples
### Example 1
**Scenario:** Answer mentions "Dr. Sarah Chen" but no document contained this name
**Wrong:** Include the name as if it came from a source
**Correct:** Remove the unsupported name or search for verification

### Example 2
**Scenario:** Answer says "according to the 2020 census" but the source was from 2015
**Wrong:** Attribute the data to the wrong year
**Correct:** Attribute data to the actual source year
