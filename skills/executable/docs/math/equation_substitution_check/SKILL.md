---
skill_id: equation_substitution_check
name: Equation Substitution Check
version: 1
priority: 0.85
error_category: verification_missing
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  If the problem states an equation in one unknown and asks for it, substitute the final answer and check the residual; a non-zero residual is provable evidence.
anchor:
  level: final
  trigger: "the problem states an equation in one unknown and asks for it"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_single_variable_equation]
    action: substitute_answer
---
# Equation Substitution Check
verification_missing made concrete. Fires rarely (AIME seldom states a bare equation) but exactly.

**Status (2026-08-22):** trigger never matched on AIME/AMC/MATH500/Olympiad — dormant until a dataset states bare one-unknown equations.
