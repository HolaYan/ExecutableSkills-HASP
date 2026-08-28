---
skill_id: signature_conformance_check
name: Signature Conformance Check
version: 1
priority: 0.9
error_category: wrong_return_type
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  The submission must define the specification's entry point with the specified arity; the grader calls it by name.
anchor:
  level: final
  trigger: "the spec declares an entry point and the submission defines a function"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_entry_point]
    action: check_signature
---
# Signature Conformance Check
bigcodebench/LCB submissions that rename the function or add parameters fail every test regardless of logic. Anchor = the submission's def line.
