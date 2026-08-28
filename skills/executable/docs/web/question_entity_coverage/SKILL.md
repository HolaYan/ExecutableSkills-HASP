---
skill_id: question_entity_coverage
name: Question Entity Coverage
version: 1
priority: 0.3
error_category: retrieval_failure
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  A named entity in the question that no search query or retrieved page mentions means the answer lacks evidence about it; search for it before finishing.
anchor:
  level: final
  trigger: "the question names an entity and searches were issued"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_uncovered_entity]
    action: search_missing_entity
---
# Question Entity Coverage
no_evidence family (45% of web wrong cases): the retrieval never reached one of the question's entities. Anchor = the entity itself.

**Status:** dormant. The failure it targets is "the entity was searched but the
fact was not retrieved", which entity containment cannot see.
