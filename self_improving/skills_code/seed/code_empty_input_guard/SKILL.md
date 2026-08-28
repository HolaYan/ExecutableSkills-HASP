---
skill_id: code_empty_input_guard
name: Empty-Input Early Return Guard
version: 1
priority: 0.9
error_category: missing_edge_case
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  HumanEval+ Plus tests and many BCB tests include empty-input edge cases
  (empty list, empty string). If your function indexes the first parameter
  at `[0]` or calls `min(param)`/`max(param)` without checking emptiness,
  it crashes on `[]` / `""`. ALWAYS guard against the empty case at the top
  of the function — return the type-appropriate default (`[]` for List,
  `""` for str, `0` for int, `False` for bool).
phases:
  pre_final:
    conditions: []
    priority_boost: 0.25
---

# Empty-Input Early Return Guard

A surgical PF inserts `if not <param>: return <default>` at the start of any
function whose first parameter is annotated `List[...]` / `list` / `str` /
`Tuple[...]` / `Set[...]` AND whose body indexes `[0]` or calls min/max on
the param AND has no existing length / emptiness guard.

## When the auto-fix fires
1. First parameter has a list-like or string annotation.
2. Function body uses `param[0]` or `min(param)` / `max(param)`.
3. No existing `if not <param>` / `if len(<param>) == 0` guard.

## What gets patched
Two lines prepended to the function body, indented to match the body:
```python
if not param:
    return DEFAULT  # [] / "" / 0 / False / None — inferred from return annotation
```

## What this protects against
- HE+ Plus tests like `rolling_max([])` / `is_palindrome("")` that crash
  on empty input.
- BCB edge cases where the model assumes non-empty.
