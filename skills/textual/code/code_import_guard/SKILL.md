---
skill_id: code_import_guard
name: Missing Import Guard
version: 1
priority: 0.85
error_category: runtime_error
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Before finalizing, ensure every module/name the code uses is imported.
  bigcodebench runtime crashes are dominated by used-but-unimported modules
  (numpy as np, pandas as pd, collections.Counter, itertools.*,
  functools.reduce, heapq, typing.*). Prepend the missing imports.
anchor:
  level: final
  trigger: "the code uses a name it never imports"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
    action: prepend_missing_imports
    action_params: {}
---

# Missing Import Guard

A large fraction of code runtime crashes (especially on library-heavy
bigcodebench tasks) are `ModuleNotFoundError` / `NameError` from using a
module or name that was never imported — the model relies on the prompt's
imports that aren't part of the submitted function.

## Detection Triggers
- Code uses `np.` / `pd.` / `plt.` with no `import numpy as np` etc.
- Code uses `math.` / `re.` / `heapq.` / `itertools.` with no `import` of it
- Code calls `Counter(...)` / `defaultdict(...)` / `deque(...)` /
  `combinations(...)` / `reduce(...)` without the matching `from ... import`
- Type hints `List[...]` / `Dict[...]` / `Optional[...]` with no
  `from typing import ...`

## Avoidance Strategies
- Make the submitted code self-contained: import every module/name it uses.
- Put imports at the top of the FINAL code block.
- Do not assume the harness provides imports from the prompt.

## Phase: pre_final
Does your FINAL code import every module and name it references? Add any
missing `import` / `from ... import ...` lines at the top.

## Examples
### Example 1
**Wrong:** code uses `Counter(words)` with no import.
**Correct:** prepend `from collections import Counter`.

### Example 2
**Wrong:** code uses `np.array(...)` with no import.
**Correct:** prepend `import numpy as np`.
