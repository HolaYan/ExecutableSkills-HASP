---
skill_id: answer_grounding_check
name: Answer Grounding Check
version: 1
priority: 0.9
error_category: hallucination
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  The final answer must occur in the retrieved evidence; an answer that appears in
  no search result or read page is unsupported — cite the passage or search again.
anchor:
  level: final
  trigger: "a committed answer that is not a yes/no or bare number, with at least one observation to check it against"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_observations]
    action: check_answer_grounding
---
# Answer Grounding Check
A wrong web answer often appears in none of the retrieved evidence. Skips
yes/no and numeric answers; grounds each comma/and-separated part separately.
Fires far more often on wrong answers than on correct ones — measure the
rate on your own episodes before relying on it.
Action: inject "answer X does not appear in any retrieved evidence; cite or search again".
