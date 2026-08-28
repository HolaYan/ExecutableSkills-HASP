---
skill_id: comparison_analyzer
name: Comparison Question Analyzer
version: 1
priority: 0.8
error_category: reasoning
applicable_modes: [all]
applicable_phases: [think, search, answer]
system_summary: >
  For comparison questions (A vs B, earlier/later, more/less), research both entities before comparing.
phases:
  post_search:
    conditions: []
    priority_boost: 0.1
  pre_final:
    conditions: [always]
    priority_boost: 0.2
---

# Comparison Question Analyzer

For questions comparing two or more entities (who came first, which is bigger, what's the difference), ensure you research ALL entities before making the comparison. Never compare based on partial information.

## Detection Triggers
- Question contains comparison words: "earlier", "later", "first", "before", "after", "more", "less", "bigger", "smaller", "difference", "compared to"
- Question mentions two or more named entities to compare
- Question asks "which" among multiple options
- Question uses superlatives: "most", "least", "best", "worst"

## Avoidance Strategies
- Identify ALL entities being compared before starting your search
- Search for each entity separately to get accurate information
- READ documents about each entity before making the comparison
- Create a mental comparison table with the relevant attribute for each entity
- Double-check that you're comparing the same attribute (e.g., both birth years, not birth vs death)

## Phase: post_search
This is a comparison question. Have you found information about ALL entities being compared? If not, search for the missing entity.

## Phase: pre_final
Before comparing, verify: (1) You have data for ALL entities, (2) You're comparing the same attribute, (3) Your comparison logic is correct (e.g., earlier = smaller year).

## Examples
### Example 1
**Scenario:** "Which film came out first, A or B?"
**Wrong:** Find release date of A only and guess about B
**Correct:** SEARCH and READ for both A and B, then compare release dates

### Example 2
**Scenario:** "Who was born first, Person X or Person Y?"
**Wrong:** Find X's birth year and assume Y is later
**Correct:** Find both birth years from reliable sources, then compare
