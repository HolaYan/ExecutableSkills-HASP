---
skill_id: edge_input_probe
name: Edge Input Probe
version: 1
priority: 0.1
error_category: runtime_error
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Call the entry point on typed edge inputs ([] / '' / 0) in the sandbox; an exception is evidence (a wrong value on an edge case is not decidable without the spec).
anchor:
  level: final
  trigger: "explicitly enabled via step_context['enable_edge_probe']"
  evidence: "executed"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_typed_signature]
    action: probe_edge_inputs
---
# Edge Input Probe
Runtime-error family: IndexError/ValueError on empty input is the most common crash in mbpp+/humaneval+ failures.

**Status:** inverted in practice — a passing solution often raises on empty
input because the spec demands it, so this probe accuses correct code.
Off by default; enable it per rollout with `step_context['enable_edge_probe']`.
