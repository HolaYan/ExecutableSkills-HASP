---
skill_id: evidence_answer_consistency
name: Evidence–Answer Consistency
version: 1
priority: 0.85
error_category: reading_comprehension_error
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  The retrieved evidence often already states the correct answer (45% of wrong
  rollouts); check whether any passage answers the question differently from
  the candidate, and quote it.
anchor:
  level: final
  trigger: "documents were read and an answer was committed"
  evidence: "helper"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_observations]
    action: teacher_evidence_consistency
---
# Evidence–Answer Consistency
Web analog of "had it and lost it": gold is inside an observation in 45% of
wrong rollouts. No deterministic form (needs reading); LLM evidence scoped to
"quote the passage and the answer it gives". Needs consent + fallback gate.
