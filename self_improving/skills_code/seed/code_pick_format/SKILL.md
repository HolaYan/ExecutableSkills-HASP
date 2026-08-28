---
skill_id: code_pick_format
name: Pick Functional vs Stdin Format
version: 1
priority: 0.8
error_category: format_mismatch
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Decide format from starter_code: starter has `class Solution` → complete
  that class, no top-level driver. Otherwise → stdin/stdout script. NEVER
  use class Solution for stdin problems — the sandbox will report "Solution
  not defined" and you score 0.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.2
---

# Pick Functional vs Stdin Format

Two distinct LCB problem styles, two distinct submission formats.
Mixing them is an instant 0/N — the sandbox can't recover.

## Detection Triggers
- Problem `Starter code` block contains `class Solution:` → **functional**
- Problem `Starter code` block is empty / absent / shows raw stdin parsing
  → **stdin/stdout**
- Atcoder / Codeforces problem text uses "Standard Input" / "Standard Output"
  → **stdin/stdout** (regardless of starter_code presence)

## The Format Protocol (mandatory)

### If functional (LeetCode style):
```python
class Solution:
    def methodName(self, arg1: Type, arg2: Type) -> ReturnType:
        # ... implementation ...
        return answer
```
- Match the EXACT method name and signature from starter_code
- Do not write `if __name__ == "__main__"` or any input() calls
- Do not print — return the value

### If stdin (Atcoder / Codeforces style):
```python
n = int(input())
arr = list(map(int, input().split()))
# ... compute ...
print(answer)
```
- No class Solution, no method
- Read from stdin per the input format
- Print to stdout per the output format

## Examples
### Example 1 — Atcoder problem written as stdin
**Problem starter_code:** (none — atcoder_abc384_c)
**Wrong format:** `class Solution:\n    def perfect_standings(self, scores): ...`
→ Sandbox runs `Solution()` lookup → NameError → 0 score
**Correct format:** read `scores = list(map(int, input().split()))`, compute,
`print(name)` for each name.

### Example 2 — LeetCode problem written as stdin
**Problem starter_code:** `class Solution:\n    def twoSum(self, nums, target):`
**Wrong format:** `nums = list(map(int, input().split())); ...; print(...)`
→ Sandbox can't find Solution.twoSum → 0 score
**Correct format:** Complete `class Solution: def twoSum(...)` and return.
