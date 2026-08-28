---
skill_id: misinformation_detector
name: Cross-Source Fact Verification
version: 1
priority: 0.85
error_category: adversarial_defense
applicable_modes: [all, adv_conflict_l1, adv_conflict_l2, adv_conflict_l3]
applicable_phases: [think, read, answer]
system_summary: >
  Cross-check critical claims across sources. Flag and investigate contradictions between documents.
anchor:
  level: step
  trigger: "two or more documents read that disagree on dates or facts"
  evidence: "deterministic"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  post_read:
    conditions: []
    priority_boost: 0.2
  pre_final:
    conditions: [always]
    priority_boost: 0.3
---

# Cross-Source Fact Verification

Actively check for contradictions between documents. When sources disagree on key facts, investigate further rather than picking one arbitrarily. Prioritize authoritative and recent sources.

## Detection Triggers
- Two or more documents provide conflicting information on the same fact
- A document contains claims that contradict common knowledge
- Search results show significant disagreement
- Answer is based on information from a single potentially unreliable source

## Avoidance Strategies
- Compare key facts across all documents you have read
- When contradictions are found, search for additional sources to resolve
- Consider source reliability: Wikipedia, official sites, and academic sources are generally more reliable
- Consider recency: newer sources may have updated information
- Do not ignore contradictions — always investigate them

## Phase: post_read
Compare the facts in this document with what you learned from previous documents. Are there any contradictions? If so, note them and seek additional sources.

## Phase: pre_final
Before answering, check: Are all your sources consistent on the key facts? If there are contradictions, have you resolved them with additional evidence?

## Examples
### Example 1
**Scenario:** Doc A says "population is 50,000", Doc B says "population is 500,000"
**Wrong:** Average them or pick one randomly
**Correct:** SEARCH for a more authoritative source (census data, official statistics)

### Example 2
**Scenario:** One adversarial document claims a different answer than two other documents
**Wrong:** Trust the adversarial document because it was read most recently
**Correct:** Trust the majority (2 sources agree vs 1 disagrees), and verify with one more search
