---
skill_id: search_stall_reformulate
name: Search Stall Reformulation
version: 1
priority: 0.78
error_category: search_optimization
applicable_modes: [all]
applicable_phases: [search]
system_summary: >
  When a search repeats a previous query or returns no new results, stop
  repeating and reformulate: search a distinct named entity, the original
  question, or a bridging sub-question.
anchor:
  level: step
  trigger: "a repeated search, or empty results after two attempts"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_search:
    conditions: []
    priority_boost: 0.2
---

# Search Stall Reformulation

Degenerate loops (the same query issued repeatedly, or searches that return
nothing) waste the step budget and end in a guessed answer. Break the loop by
changing the query, not repeating it.

## Detection Triggers
- The current query exactly repeats a recent query
- The last searches returned empty / no usable results
- Several searches in a row with no new information

## Avoidance Strategies
- Do not reissue a query that already failed.
- Search a specific named entity from the question instead of the whole sentence.
- Decompose multi-hop questions: find the bridging entity first, then the target.
- Try the original question verbatim if you have been over-paraphrasing it.

## Phase: post_search
Did this query return anything new? If not, change it — search a specific
entity or a sub-question rather than repeating.

## Examples
### Example 1
**Wrong:** SEARCH("director of film X") x4 returning nothing.
**Correct:** SEARCH("film X") to find the director's name, then search that name.
