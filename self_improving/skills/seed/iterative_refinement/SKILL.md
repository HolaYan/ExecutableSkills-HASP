---
skill_id: iterative_refinement
name: Iterative Search Refinement
version: 1
priority: 0.7
error_category: search_optimization
applicable_modes: [all]
applicable_phases: [think, search, read]
system_summary: >
  After each search or read, assess what's still missing and refine your next query accordingly.
phases:
  post_read:
    conditions: []
    priority_boost: 0.1
  post_search:
    conditions: []
    priority_boost: 0.1
---

# Iterative Search Refinement

After each search and read, evaluate what information you still need. Use what you've learned to formulate better, more targeted queries rather than repeating similar searches.

## Detection Triggers
- Multiple searches with similar queries returning same results
- Read documents provide partial but incomplete information
- Intermediate entities found but not yet followed up
- Search results tangentially related but not directly answering the question

## Avoidance Strategies
- After each READ, identify what new information you learned and what's still missing
- Use newly discovered entity names or facts to refine subsequent searches
- Avoid repeating the same or very similar queries
- If stuck, try searching from a different angle (e.g., search for the answer entity directly)
- Use specific names, dates, or facts learned from previous reads in new queries

## Phase: post_read
What did you learn from this document? What's still missing? Use any new names, dates, or facts to refine your next search query.

## Phase: post_search
Review these results in light of what you already know. Are any results more promising than others? READ the most relevant one.

## Examples
### Example 1
**Scenario:** First search found the director is "John Ford", but need his birthplace
**Wrong:** SEARCH("director birthplace") — too vague
**Correct:** SEARCH("John Ford birthplace") — use the specific name learned
