---
skill_id: spec_example_check
name: Spec Example Check
version: 1
priority: 0.95
error_category: wrong_output
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Execute the submitted code on the specification's own examples (doctests,
  asserts, public tests) in a sandbox; a failing example is provable evidence.
anchor:
  level: final
  trigger: "the spec carries its own `>>>` / `assert` examples, or the runtime supplied public tests"
  evidence: "executed"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_spec_examples]
    action: run_spec_examples
    action_params: {timeout_s: 8}
---
# Spec Example Check
Code analog of compute_observation_verify. Of 172 wrong cases whose spec has
examples, 76 fail those very examples. Evidence = the failing call, the got
value and the expected value; the model redoes the function with that in view.
