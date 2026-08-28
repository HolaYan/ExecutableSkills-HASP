---
skill_id: hallucination
name: Evidence-Grounded Answering
version: 1
priority: 0.9
error_category: hallucination
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Every fact in your answer must come from a retrieved document — never guess.
phases:
  post_search:
    conditions: []
    priority_boost: 0.0
  post_read:
    conditions: []
    priority_boost: 0.0
  pre_final:
    conditions: [no_read_yet]
    priority_boost: 0.4
    action: verify_hallucination_grounding
    action_params: {}
---

# Evidence-Grounded Answering

Never fabricate answers without evidence from retrieved documents. Every claim in the final answer must be traceable to a specific search result or document that was read.

## Detection Triggers
- About to provide a FINAL answer without having READ any document
- Answer contains specific facts (names, dates, numbers) not found in any retrieved source
- Answer is being generated from general knowledge without search verification
- Model output contains placeholder text like 'answer' instead of actual content

## Avoidance Strategies
- NEVER provide a FINAL answer without first SEARCH and READ at least one relevant document
- Every specific fact in your answer must come from a retrieved and read document
- If you cannot find evidence for a claim, do not include it in your answer
- If no relevant information is found after thorough searching, state that clearly rather than guessing
- Cite the specific document (doc_id) that supports your answer in your Thought
- Avoid outputting placeholder text — always provide a substantive answer or state you cannot find one

## Phase: post_search
Do NOT formulate an answer from general knowledge. Use these search results to guide your next action: either READ the most relevant result, or SEARCH again with a better query.

## Phase: post_read
Ground your answer in THIS document. Mentally identify the exact sentence that answers the question. If this document does not contain the answer, search for another source.

## Phase: pre_final
CRITICAL: You are about to answer without having READ any document. This risks hallucination. Go back and SEARCH + READ at least one relevant source before answering.

## Examples
### Example 1
**Scenario:** Question about a screenwriter. No search has been performed yet.
**Wrong:** FINAL('John Smith') based on guessing without any search
**Correct:** SEARCH for the information first, READ a relevant document, then FINAL with the verified answer

### Example 2
**Scenario:** Search results don't clearly answer the question
**Wrong:** Fabricate a plausible-sounding answer
**Correct:** Try additional searches, and if still uncertain, provide the best available answer with appropriate hedging
