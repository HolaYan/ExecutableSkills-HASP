---
skill_id: adversarial_distraction
name: Adversarial Distraction Defense
version: 1
priority: 0.9
error_category: adversarial_distraction
applicable_modes: [adv_conflict_l1, adv_conflict_l2, adv_conflict_l3, adv_outdated, adv_irrelevant, adv_noise, adv_reorder, clean]
applicable_phases: [think, search, read]
system_summary: >
  Watch for conflicting claims; cross-reference 2+ sources before accepting any fact.
anchor:
  level: step
  trigger: "search results carrying three or more conflict words"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_search:
    conditions: [search_has_conflicts]
    priority_boost: 0.3
  post_read:
    conditions: []
    priority_boost: 0.0
  pre_final:
    conditions: [always]
    priority_boost: 0.1
    action: verify_adversarial_distraction
    action_params: {}
---

# Adversarial Distraction Defense

Detect and resist misleading information injected into search results, including conflicting claims, outdated data, reordered documents, and irrelevant noise.

## Detection Triggers
- Search results contain conflicting information about the same fact
- A source contradicts multiple other sources
- Information appears suspiciously different from common knowledge
- Documents seem reordered with less relevant results appearing first
- Outdated information conflicts with more recent data

## Avoidance Strategies
- Cross-reference claims across 2+ independent sources before accepting
- Prefer authoritative sources (Wikipedia, official sites, academic publications)
- If sources conflict, search for additional evidence rather than choosing arbitrarily
- Do not trust a single source alone — verify key facts with at least one more search
- Check dates and recency of information when sources disagree
- Ignore search results that are clearly off-topic or irrelevant to the question

## Phase: post_search
WARNING: The search results contain contradictory information. Before acting:
1. Identify the SPECIFIC claim that differs between sources.
2. SEARCH with a more targeted query to find a third source.
3. Prefer Wikipedia, .gov, .edu over blogs or aggregators.
4. Only proceed to READ once 2+ sources agree on the key fact.
Do NOT pick an answer based on a single source when conflicts exist.

## Phase: post_read
Cross-check: does this document's key claim match what other search results said? If not, note the discrepancy and search for a tiebreaker source before answering.

## Phase: pre_final
Before answering: confirm your answer is supported by at least 2 independent sources. If you only have one source, search once more to verify.

## Examples
### Example 1
**Scenario:** Question asks 'who sang the national anthem at the super bowl 2019'. Search results show both Gladys Knight and another performer.
**Wrong:** Accept the first search result without verification
**Correct:** Search specifically for 'Super Bowl 2019 national anthem singer' and cross-reference multiple sources

### Example 2
**Scenario:** Question about a historical date. One source says 1921, another says post-WWII.
**Wrong:** Pick whichever source appeared first in results
**Correct:** Search for additional sources and check which claim has more authoritative backing
