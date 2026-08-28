---
skill_id: exception_contract_check
name: Exception Contract Check
version: 1
priority: 0.85
error_category: missing_exception
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  If the specification says the function should raise exception X for some
  condition and the submission never raises X, the grader's assertRaises will fail.
anchor:
  level: final
  trigger: "the spec says an exception should be raised for some input"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [spec_requires_exception]
    action: check_exception_contract
---
# Exception Contract Check
bigcodebench specs state exception contracts explicitly, which is where this
check has the most to work with. Static, so it is cheap and precise by
construction.
8 failing / 0 passing solutions fire.
