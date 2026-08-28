---
skill_id: comparison_evidence_completeness
name: Comparison Evidence Completeness
version: 1
priority: 0.3
error_category: insufficient_exploration
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  A comparison question (which … first / older / more) answered with evidence about only one side cannot be decided; search for the other side.
anchor:
  level: final
  trigger: "a comparison question naming two or more entities"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [is_comparison_one_sided]
    action: search_other_side
---
# Comparison Evidence Completeness
2Wiki/HotpotQA comparison questions ('which film's director died first'): the model often answers after reading about one entity. Anchor = the side without evidence.

**Status:** dormant. Containment over entity names cannot see the failure this
targets, which is a missing *relation* rather than a missing entity.
