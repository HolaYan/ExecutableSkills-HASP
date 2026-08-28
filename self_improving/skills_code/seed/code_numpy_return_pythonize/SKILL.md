---
skill_id: code_numpy_return_pythonize
name: Convert Numpy Array Returns to Python Lists
version: 1
priority: 0.8
error_category: numpy_python_type_mismatch
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  When you `return np.array(...)`, `return np.zeros(...)`, etc., the test's
  `assertEqual([1.0, 2.0], result)` fails because `[np.float64(1.0), ...]
  != [1.0, ...]` element-wise type mismatch. ALWAYS wrap numpy array returns
  with `.tolist()` so Python-typed asserts succeed.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.15
---

# Convert Numpy Array Returns to Python Lists

A surgical PF rewrites `return np.<arrayfunc>(...)` to
`return np.<arrayfunc>(...).tolist()` so unittest equality assertions on
plain Python lists / floats pass.

## When the auto-fix fires
The FINAL contains `return np.array(...)` / `np.zeros(...)` / `np.ones(...)`
/ `np.full(...)` / `np.arange(...)` / `np.linspace(...)` etc. AND not already
chained with `.tolist()` / `.item()` / `list(...)`.

## What gets patched
The matching `np.<func>(...)` expression is wrapped with `.tolist()`.

## What this protects against
BCB tests that assert exact equality against Python literal lists, where
numpy scalars compare element-wise but show as `np.float64(0.0) != 0.0` in
the diff.
