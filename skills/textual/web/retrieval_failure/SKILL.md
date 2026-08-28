---
skill_id: retrieval_failure
name: Adaptive Search Recovery
version: 2
priority: 0.8
error_category: retrieval_failure
applicable_modes: [all]
applicable_phases: [think, search]
system_summary: >
  When searches fail, reformulate queries with synonyms or decompose into sub-queries.
anchor:
  level: step
  trigger: "a search query longer than 15 words, capped per episode"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_search:
    conditions: [search_empty]
    priority_boost: 0.3
    action: reformulate_search
  pre_final:
    conditions: [no_read_yet]
    priority_boost: 0.2
    action: force_read_best_doc
---

# Adaptive Search Recovery

When initial searches fail, adapt by reformulating queries, using synonyms, or decomposing complex questions into sub-queries.

## Detection Triggers
- Search returns no results or only irrelevant results
- Search query is too vague or too specific
- Multi-hop question requires intermediate entities not yet found
- Question involves rare/long-tail entities

## Avoidance Strategies
- If first search fails, reformulate with synonyms or alternative phrasing
- For multi-hop questions, decompose into sub-questions and search each
- Try both specific and broader queries
- Do not give up after a single failed search — try at least 2-3 formulations

## Phase: post_search
No useful results. Try: (1) synonyms, (2) decompose multi-hop, (3) broaden query. Do NOT proceed to FINAL without a relevant document.

## Phase: pre_final
STOP: You have not READ any document yet. Go back and READ at least one relevant source before answering.

## Examples
### Example 1
**Scenario:** Searching for 'Tarmina director' returns no results
**Wrong:** Give up and say 'I cannot find the answer'
**Correct:** Try 'Tarmina film', 'Tarmina movie cast'

### Example 2
**Scenario:** Multi-hop: 'Capital of the country where Film X's director was born?'
**Wrong:** Search the entire question as one query
**Correct:** Step 1: 'Film X director', Step 2: 'director birthplace', Step 3: 'country capital'
