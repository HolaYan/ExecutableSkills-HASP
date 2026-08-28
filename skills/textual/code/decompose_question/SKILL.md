---
skill_id: decompose_question
name: Decompose Multi-Clause Spec into Build Steps
version: 1
priority: 0.83
error_category: missing_substep
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  When a coding spec has multiple action clauses (validate / compute /
  generate / plot / return ...), turn it into a numbered build plan, write
  helper functions for non-trivial sub-steps, and only assemble FINAL after
  every intermediate is verified.
anchor:
  level: final
  trigger: "a spec longer than 200 characters describing three or more steps"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
    action: decompose_spec
    action_params: {}
---

# Decompose Multi-Clause Spec into Build Steps

bigcodebench tasks often pack 4-6 distinct sub-steps into one prompt (input
validation + compute + plot + return tuple). Trying to write a single 30-line
function in one pass routinely drops or mis-orders a sub-step.

## Detection Triggers
- Spec ≥ 200 chars and contains ≥ 3 action verbs
  (validate / raise / compute / generate / return / plot / extract / parse /
  filter / sort / group / merge / cluster / fit / transform / save / load …)
- Multiple "The function should …" sentences
- Multiple `>>>` docstring examples

## Avoidance Strategies
- Translate each action clause into a build step before writing code.
- Write a small helper for any non-trivial sub-step; name + comment it.
- Compute and store each intermediate in a named variable; only `return` at
  the end after every intermediate is set.
- Trace the docstring example through your code mentally before FINAL.

## Phase: pre_final
Did you address every "should …" clause in the spec, in order? Is each one
visible as a step / helper in the final code?

## Examples
### Example 1
**Spec:** "Validate that `data` is a DataFrame. Run KMeans. Plot clusters with
centroids. Return (labels, axes)."
**Plan:** (1) `if not isinstance(data, pd.DataFrame): raise ValueError`.
(2) `KMeans(...).fit_predict(data)` → `labels`, `.cluster_centers_` →
`centroids`. (3) `fig, ax = plt.subplots(); ax.scatter(...); ax.scatter(centroids…)`.
(4) `return (labels, ax)`.
