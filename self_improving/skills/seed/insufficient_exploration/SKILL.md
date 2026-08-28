---
skill_id: insufficient_exploration
name: Thorough Exploration Mandate
version: 2
priority: 0.8
error_category: insufficient_exploration
applicable_modes: [all]
applicable_phases: [think, search, read]
system_summary: >
  Do not answer before reading at least one source. Try 2-3 search queries and READ documents before answering.
phases:
  post_search:
    conditions: [no_read_yet]
    priority_boost: 0.1
    action: force_read_best_doc
  pre_final:
    conditions: [no_read_yet]
    priority_boost: 0.3
    action: block_premature_final
---

# Thorough Exploration Mandate

Ensure sufficient exploration before answering. Do not give up after one search or skip reading documents. Try multiple queries and read at least 2 documents before providing a final answer.

## Detection Triggers
- Only one search query has been tried so far
- No READ action has been performed yet
- Search results exist but none have been read
- The current answer is based on snippets alone without reading full documents
- A complex multi-hop question has only had partial exploration
- Agent attempts FINAL with read_count == 0
- Agent's reasoning shows no reference to document content

## Avoidance Strategies
- Always READ at least one document before attempting FINAL
- Try at least 2-3 different search queries before concluding information is unavailable
- Do not base your answer solely on search snippets — always READ at least one full document
- For multi-hop questions, ensure each hop has been explored with at least one search
- If first searches return poor results, reformulate and try again before giving up
- Search snippets are not sufficient evidence — read the full document
- Budget permitting, read at least 2 sources for cross-verification

## Phase: post_search
You have search results but have not READ any document yet. Do NOT proceed to FINAL based on snippets alone.
1. READ the most promising result to verify the snippet information.
2. If the snippets seem insufficient, try a different search query.
3. For multi-hop questions, ensure you have explored each hop.

## Phase: pre_final
EVIDENCE CHECK: Do not answer without reading at least one source.
1. Have you READ at least one full document? If not, do so now.
2. Is your answer based on document evidence, not just prior knowledge?
3. If uncertain, read another source for verification.

## Examples
### Example 1
**Scenario:** Question requires comparing two entities. Agent only searched for one.
**Wrong:** Provide answer after searching only one entity
**Correct:** Search for both entities, READ relevant docs for each, then compare

### Example 2
**Scenario:** Search returns results with potential answer in snippets
**Wrong:** FINAL answer based on snippet without READing the document
**Correct:** READ the most relevant document to verify the snippet information before answering
