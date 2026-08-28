---
skill_id: code_specific_exception_type
name: Replace Generic Exception With Specific Type
version: 1
priority: 0.75
error_category: wrong_exception_type
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Tests often `assertRaises(SpecificError, ...)`. `raise Exception("msg")`
  is the model's default fallback but it is strictly worse than a precise
  type. Use `FileNotFoundError`, `ValueError`, `PermissionError`,
  `ConnectionError`, `TypeError` etc. based on the situation —
  the message text is usually a strong hint.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.15
---

# Replace Generic Exception With Specific Type

A surgical PF maps `raise Exception("msg")` → `raise <SpecificType>("msg")`
based on keyword analysis of the message:

| Message keyword | Replacement type |
|---|---|
| not found / does not exist / no such | FileNotFoundError |
| invalid / bad / malformed / parse | ValueError |
| permission / denied / forbidden | PermissionError |
| connection / network / unreachable | ConnectionError |
| type / wrong type / expected | TypeError |

## When the auto-fix fires
The candidate has at least one `raise Exception(LITERAL_MSG)` whose message
matches one of the keyword groups above.

## What gets patched
The matched `raise Exception(MSG)` becomes `raise <SpecificType>(MSG)`. Other
raises in the same function are independent (each handled separately).

## What this protects against
BCB tests that use `assertRaises(FileNotFoundError, task_func, ...)` —
generic Exception fails the type-narrow assertion.
