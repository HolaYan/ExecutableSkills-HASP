---
skill_id: entity_constraint_check
name: Answer Entity Constraint Check
version: 1
priority: 0.82
error_category: multi_hop_reasoning_failure
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Before finalizing a multi-hop or comparison answer, verify the answer is the
  entity actually asked for (the relative/attribute of the bridging entity, not
  the bridging entity itself) and that it satisfies every constraint, with the
  comparison direction correct.
anchor:
  level: final
  trigger: "a multi-hop or comparison question, checked once per episode"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
---

# Answer Entity Constraint Check

The largest web failure mode is finalizing an answer that IS present in the
retrieved evidence but is the WRONG one: the bridging entity instead of the
target, the wrong hop, the wrong attribute, or an inverted comparison.

## Detection Triggers
- Multi-hop question (possessive chain "X's Y", relative pronoun who/which/whose/that)
- Comparison question (first/earlier/later/older/more/less/which)
- Answer about a named entity that appears in evidence but may be the wrong one

## Avoidance Strategies
- Re-read the question's final ask: which entity/attribute is requested?
- Confirm your answer is the TARGET, not the bridging entity you searched through.
- For comparisons, check the direction (earlier vs later) and that BOTH sides
  were evaluated, not just one.
- Confirm the answer satisfies every qualifier in the question (date, role, place).

## Phase: pre_final
Is your answer the exact entity asked for, satisfying every constraint? For a
comparison, is the direction right and were both entities compared?

## Examples
### Example 1
**Question:** "Where did the father of X die?"
**Wrong:** returns where X died (the bridging entity).
**Correct:** returns where X's father died (the target).
