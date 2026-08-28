---
skill_id: api_attribute_probe
name: API Attribute Probe
version: 1
priority: 0.9
error_category: wrong_library_api
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Resolve every imported-module attribute chain the code calls (pd.X.Y, np.X)
  in the sandbox; a chain that does not exist is provable evidence of a wrong
  library API.
anchor:
  level: final
  trigger: "the submission calls an attribute chain on a module it imports"
  evidence: "executed"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [uses_library_api]
    action: probe_api_attributes
---
# API Attribute Probe
Replaces the reminder-style `search_restructuring` (library/API hints) with an
executed check. The sandbox runs with tight memory limits and no network, so
a heavyweight import can fail there for reasons unrelated to the submission —
treat a fire as a hint, not a proof.
