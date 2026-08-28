---
skill_id: wrong_entity_confusion
name: Entity Disambiguation
version: 1
priority: 0.8
error_category: wrong_entity_confusion
applicable_modes: [all]
applicable_phases: [think, search, read]
system_summary: >
  When names are ambiguous, verify using dates, locations, or roles to disambiguate.
phases:
  post_search:
    conditions: [search_has_similar_entities]
    priority_boost: 0.2
  post_read:
    conditions: [read_has_multiple_entities]
    priority_boost: 0.1
    action: verify_entity_disambiguation
    action_params: {}
---

# Entity Disambiguation

Verify that the entity being discussed is the correct one when names are ambiguous. Check distinguishing attributes like dates, locations, and roles to disambiguate same-name entities.

## Detection Triggers
- Search results return multiple entities with the same or similar name
- The entity name is common (e.g., 'The Storm', 'John Smith', 'Gentleman')
- Question mentions a work title that could refer to multiple works (book, film, song)
- Historical figures with similar names appear in results

## Avoidance Strategies
- When multiple entities share a name, verify using distinguishing attributes (year, creator, genre, nationality)
- For works of art/media, include the creator or year in your search to disambiguate
- For people, check birth/death dates, nationality, and profession to confirm identity
- If 'Film X director' returns multiple films named X, add year or other context to the search
- Never assume the first search result is the correct entity — verify against the question context
- Explicitly state which entity you are referring to in your Thought

## Phase: post_search
DISAMBIGUATION REQUIRED: Multiple entities share similar names in these results.
1. Note which specific entity the question refers to (check year, creator, medium, context).
2. Add disambiguating terms to your next search (e.g., year, creator name, genre).
3. Do NOT read a document until you are confident it is about the correct entity.
4. If unsure, search '[entity name] disambiguation' or '[entity name] [year]'.

## Phase: post_read
This document discusses multiple entities. Verify you are extracting information about the CORRECT one by checking distinguishing attributes (dates, nationality, profession).

## Examples
### Example 1
**Scenario:** Question about 'The Storm' author — multiple works share this title
**Wrong:** Accept Rachel Hawkins (recent author) without checking
**Correct:** Search 'The Storm Kate Chopin' or check which 'The Storm' the question refers to

### Example 2
**Scenario:** Question about 'Gentleman' music genre — could be Lou Bega song or Guy Ritchie film
**Wrong:** Return the genre of the wrong 'Gentleman'
**Correct:** Disambiguate by searching 'Gentleman Lou Bega genre' with artist context
