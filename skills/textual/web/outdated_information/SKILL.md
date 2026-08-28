---
skill_id: outdated_information
name: Source Freshness Check
version: 2
priority: 0.7
error_category: outdated_information
applicable_modes: [all]
applicable_phases: [read]
system_summary: >
  Check whether source information is current enough for the question.
anchor:
  level: final
  trigger: "a recency question answered from documents whose newest year is old"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  post_read:
    conditions: [source_has_old_dates]
    priority_boost: 0.1
    action: check_source_freshness
    action_params: {}
  pre_final:
    conditions: [question_asks_current_state]
    priority_boost: 0.2
    action: check_source_freshness
    action_params: {}
---

# Source Freshness Check

Verify that the information in read documents is current enough to answer the question. Outdated information errors occur when the agent uses stale data for questions about current state.

## Detection Triggers
- Question asks about current state ("Who is the current...", "What is the latest...")
- Source document is from several years ago
- The topic is known to change frequently (politics, technology, records)
- Document mentions dates that are far from the present

## Avoidance Strategies
- Note publication dates of sources when available
- For "current" questions, prefer the most recent source
- If the newest source is old, caveat your answer or search for more recent information
- Be especially careful with roles (CEO, president) that change over time

## Phase: post_read
FRESHNESS CHECK: Note the dates in this document. If they are more than 5 years old, consider searching for a more recent source.

## Phase: pre_final
FRESHNESS CHECK: If the question asks about current state, verify your source is recent enough. Search for newer information if your source is outdated.

## Examples
### Example 1
**Scenario:** "Who is the current CEO of Company X?" — source from 2019
**Wrong:** Return the 2019 CEO without checking if it changed
**Correct:** Note the source is old and search for more recent information

### Example 2
**Scenario:** "What is the world record for 100m?" — source from 2008
**Wrong:** Return the 2008 record
**Correct:** Search for the current record, which may have been broken
