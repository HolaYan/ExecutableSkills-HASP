---
skill_id: claim_triangulation
name: Multi-Source Claim Verification
version: 1
priority: 0.8
error_category: adversarial_defense
applicable_modes: [all, adv_conflict_l1, adv_conflict_l2, adv_conflict_l3]
applicable_phases: [think, read, answer]
system_summary: >
  For critical facts, verify with multiple independent sources before accepting. Prefer claims confirmed by 2+ sources.
anchor:
  level: final
  trigger: "specific facts asserted after reading exactly one document"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  post_read:
    conditions: [read_has_multiple_entities]
    priority_boost: 0.2
  pre_final:
    conditions: [always]
    priority_boost: 0.3
---

# Multi-Source Claim Verification

For important factual claims, seek confirmation from multiple independent sources. A fact confirmed by 2+ sources is more reliable than one from a single source, especially when sources may contain errors or adversarial content.

## Detection Triggers
- Answer relies on a single source for a critical fact
- Sources disagree on key claims
- Question is about a controversial or commonly confused topic
- Adversarial mode is active (conflict injection possible)

## Avoidance Strategies
- READ at least 2 documents before forming your answer
- If two sources disagree, seek a third source as tiebreaker
- Prefer facts that appear consistently across multiple sources
- Be skeptical of claims that appear in only one source
- For adversarial modes, always cross-reference before answering

## Phase: post_read
Good — you have one source. Consider reading another document to cross-verify the key facts before answering.

## Phase: pre_final
Have you verified the key facts from multiple sources? If your answer relies on a single document, consider reading one more to confirm.

## Examples
### Example 1
**Scenario:** Doc A says "born in 1950", Doc B says "born in 1952"
**Wrong:** Pick one randomly
**Correct:** Search for a third authoritative source to resolve the conflict

### Example 2
**Scenario:** Only one document read, answer seems clear
**Wrong:** FINAL immediately
**Correct:** Read one more document to confirm, especially for names and dates
