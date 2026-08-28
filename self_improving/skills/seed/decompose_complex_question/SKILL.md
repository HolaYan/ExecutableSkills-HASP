---
skill_id: decompose_complex_question
name: Complex Question Decomposition
version: 1
priority: 0.85
error_category: reasoning
applicable_modes: [all]
applicable_phases: [think, search]
system_summary: >
  Break multi-hop and comparative questions into sub-questions. Search for each sub-question separately before combining.
phases:
  post_search:
    conditions: [search_has_conflicts]
    priority_boost: 0.2
  pre_final:
    conditions: [no_read_yet]
    priority_boost: 0.3
---

# Complex Question Decomposition

For multi-hop questions (e.g., "Where was X's father born?"), decompose into sub-questions and search each separately. Never attempt to answer a multi-hop question with a single search.

## Detection Triggers
- Question contains relative clauses (who, which, whose, that)
- Question asks about a property of an entity defined by another property
- Question contains "and" connecting two distinct information needs
- Question requires chaining 2+ facts from different sources

## Avoidance Strategies
- Identify the reasoning hops before searching (e.g., hop1: find X's father, hop2: find birthplace)
- Search for each hop separately with focused queries
- Verify each intermediate entity before proceeding to the next hop
- Do not combine multiple hops into a single search query
- READ documents for each intermediate fact before proceeding

## Phase: post_search
This is a multi-hop question. Have you identified all intermediate entities? If not, decompose the question and search for each part separately.

## Phase: pre_final
STOP: This question requires multiple reasoning steps. Have you verified each intermediate fact from a source document? If not, go back and search for the missing pieces.

## Examples
### Example 1
**Scenario:** "Where was the director of Tarmina born?"
**Wrong:** SEARCH("Where was the director of Tarmina born?") — too complex for one query
**Correct:** Step 1: SEARCH("Tarmina film director"), Step 2: READ to find director name, Step 3: SEARCH("director_name birthplace")

### Example 2
**Scenario:** "What nationality is the spouse of the person who wrote X?"
**Wrong:** Single search with the full question
**Correct:** SEARCH("who wrote X") → READ → SEARCH("author_name spouse") → READ → SEARCH("spouse_name nationality")
