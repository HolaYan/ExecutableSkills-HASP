---
skill_id: search_depth_controller
name: Adaptive Search Depth Control
version: 1
priority: 0.75
error_category: search_optimization
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Adjust search thoroughness based on question complexity. Multi-hop questions need more searches and reads.
phases:
  pre_final:
    conditions: [always]
    priority_boost: 0.2
---

# Adaptive Search Depth Control

Match your search effort to the question's complexity. Simple factoid questions may need 1-2 searches, while multi-hop or comparison questions need 3+ searches and 2+ document reads.

## Detection Triggers
- Attempting to answer a complex question with too few searches
- Only one document read for a multi-hop question
- Question complexity does not match exploration depth
- Budget still available but agent rushing to answer

## Avoidance Strategies
- For simple factoid questions: at least 1 search + 1 read
- For multi-hop questions: at least 2 searches + 2 reads
- For comparison questions: at least 1 search + 1 read PER entity
- Do not rush to FINAL when budget allows more exploration
- Use remaining budget to verify and cross-check before answering

## Phase: pre_final
Is your exploration depth appropriate for this question's complexity? Multi-hop and comparison questions typically need more searches and reads than simple factoid questions.

## Examples
### Example 1
**Scenario:** Multi-hop question, only 1 search done, 8 steps remaining
**Wrong:** FINAL with unverified guess
**Correct:** Continue searching to find all intermediate entities
