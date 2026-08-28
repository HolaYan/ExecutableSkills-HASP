---
skill_id: unsupported_known_result
name: Unsupported Known Result
version: 1
priority: 0.75
error_category: overgeneralization
applicable_phases: [think]
system_summary: >
  A step that invokes "a known result / well-known formula" must have that
  result verified: true as stated, and applicable under this problem's conditions.
anchor:
  level: step
  trigger: "the reasoning cites a 'known result' or 'well-known' fact"
  evidence: "helper"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [cites_known_result]
    action: verify_cited_result
---
# Unsupported Known Result
geometry / misread families (aime24_4: "a known result … n·r = R", actually
n·r ≤ R). Mid-reasoning counterpart of unsupported_final_answer. Anchor =
citation phrases; evidence = LLM audit scoped to "is the cited result true and
applicable"; needs model consent + fallback gate (not provable).
