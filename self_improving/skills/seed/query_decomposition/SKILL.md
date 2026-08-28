---
skill_id: query_decomposition
name: Smart Query Decomposition
version: 1
priority: 0.75
error_category: search_optimization
applicable_modes: [all]
applicable_phases: [think, search]
system_summary: >
  Break complex search queries into focused sub-queries. Use simple, specific queries rather than long natural language questions.
phases:
  post_search:
    conditions: [search_empty]
    priority_boost: 0.3
---

# Smart Query Decomposition

Complex queries with multiple clauses often fail. Break them into simple, focused sub-queries. Use entity names and key attributes rather than full natural language questions.

## Detection Triggers
- Search query is longer than 10 words
- Search query contains multiple clauses or conditions
- Search returns no results or irrelevant results
- Query mixes multiple information needs in one search

## Avoidance Strategies
- Keep search queries to 3-8 words focused on the key entity and attribute
- For multi-hop questions, search for one hop at a time
- Use the entity name + key attribute as the query (e.g., "Albert Einstein birthplace")
- If a query fails, try simpler reformulations: remove adjectives, use core nouns only
- Try both the full name and abbreviated forms

## Phase: post_search
Your query may be too complex. Try breaking it into simpler parts. Focus on the core entity + attribute.

## Examples
### Example 1
**Scenario:** Need to find where a film director was born
**Wrong:** SEARCH("Where was the director of the film Tarmina born?")
**Correct:** SEARCH("Tarmina film director") → find name → SEARCH("director_name birthplace")

### Example 2
**Scenario:** Complex compound query
**Wrong:** SEARCH("What is the population of the capital of the country where Einstein was born?")
**Correct:** SEARCH("Einstein birthplace") → SEARCH("Germany capital") → SEARCH("Berlin population")
