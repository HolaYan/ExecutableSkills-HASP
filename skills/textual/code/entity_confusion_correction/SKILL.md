---
skill_id: entity_confusion_correction
name: Return Type Alignment with Spec
version: 1
priority: 0.91
error_category: wrong_return_type
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Verify the runtime type of every element of the function's `return ...`
  matches the type the spec declares (`plt.Axes`, `np.ndarray`, `pd.DataFrame`,
  `list`, `tuple`, …). Deterministic repair for the canonical confusions:
  `plt.scatter()` → `plt.Axes`, `np.array(...)` → `list`.
anchor:
  level: final
  trigger: "the spec declares a return type"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.4
    action: align_return_type
    action_params: {}
---

# Return Type Alignment with Spec

The largest single cause of "code passes my eye-test but fails the grader"
on bigcodebench is returning the wrong TYPE. The algorithm produces the right
data but wraps it in the wrong object: `plt.scatter()` (a `PathCollection`)
instead of `plt.Axes`; `np.array(...)` instead of a Python `list`; a
`PathCollection` from `ax.scatter()` instead of `ax` itself.

## Detection Triggers
- Spec contains a declared return type:
  `plt.Axes` / `np.ndarray` / `pd.DataFrame` / `pd.Series` /
  `tuple` / `list` / `dict` / `str` / `int` / `float` / `bool`.
- A `return ...` is present in the FINAL code.

## Deterministic Repairs
- Spec wants `plt.Axes` and code returns a name bound to `plt.scatter(...)`
  → rewrite the return to `plt.gca()`.
- Spec wants `list` (and not `ndarray`) and code returns
  `np.array(...)` / `np.zeros(...)` etc. → wrap with `.tolist()`.

## Avoidance Strategies
- Read the "should output with:" / "Returns:" block FIRST and write the
  exact return shape down.
- For plots, prefer `fig, ax = plt.subplots(); ax.scatter(...); return ax`.
- For sklearn outputs that tests `assertEqual` against Python lists, add
  `.tolist()`.
- For tuples with one element, remember the trailing comma `(x,)`.

## Phase: pre_final
Does every element in your `return ...` match the declared type EXACTLY?
If not, fix the wrapper, not the algorithm.

## Examples
### Example 1
**Spec return:** `(np.ndarray, plt.Axes)`.
**Wrong:** `scatter = plt.scatter(...); return (labels, scatter)` →
`PathCollection`, not `Axes`.
**Correct:** `fig, ax = plt.subplots(); ax.scatter(...); return (labels, ax)`
or `plt.scatter(...); return (labels, plt.gca())`.
