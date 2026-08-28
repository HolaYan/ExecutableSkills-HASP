---
skill_id: code_sandbox_quick_check
name: Sandbox Quick-Check on Docstring Examples
version: 2
priority: 0.95
error_category: wrong_answer
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Before treating your draft solution as final, mentally run it on EACH
  `>>> ` example shown in the function docstring (or the first `assert` in
  the prompt for MBPP-style tasks). If your traced output diverges from
  the expected output, the algorithm is wrong on that exact case — fix the
  divergence point before submitting. The grader runs the SAME docstring
  examples plus stricter hidden ones; if your code fails the visible
  examples, hidden tests will fail too. Do not submit code you have not
  traced against every visible example.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
---

# Sandbox Quick-Check on First Public Example

Wrong-answer is by far the largest LCB failure mode (~95% of fails on
Qwen2.5-7B). Most wrong-answers come from algorithms that look right
in the abstract but break on the very first concrete example. This skill
forces a single concrete trace before submission.

## Detection Triggers
- Draft solution complete, about to FINAL
- Public example(s) present in the problem statement
- The function returns / prints something computable in O(small)

## The Quick-Check Protocol (mandatory before FINAL)
1. Take the FIRST public example's `input` exactly as given.
2. Run your algorithm by hand on that input — line by line, no shortcuts.
3. Note the value your code would return / print at the end.
4. Compare it against the expected `output`.
5. If they match: emit FINAL.
   If they diverge: identify the FIRST step where your output deviates
   from what's expected. Fix that step. Re-trace from the start. Only
   FINAL once a clean trace matches expected output.

## What this skill PROTECTS against
- Correct-looking but actually-wrong algorithms (off-by-one, wrong base
  case, missed branch, wrong accumulator init, wrong comparison direction).
- Solutions that work on degenerate cases but break on the example.

## What this skill IS NOT for
- Performance / TLE concerns (those need complexity reasoning, not tracing).
- Format / wrapper issues (covered by `code_pick_format`).
- Syntax errors (covered by `code_teacher_syntax_fix`).

## Examples
### Example — Off-by-one caught by trace
**Problem (excerpt):** Find the longest subarray whose sum is divisible by k.
**Public example:** `nums=[2,7,6,1,4,5], k=3 → 4`
**Wrong code traces to 3** because the prefix-sum hash records `pre[0]=0`
at index 0 instead of index -1 → the "from index 0" subarrays get length-1
short. Tracing: pre = [0,2,9,15,16,20,25] (mod 3 → [0,2,0,0,1,2,1]). First
mod=0 already at index 0 → max_len = i - first[mod] = 0, then 2 - 0 = 2,
then 3 - 0 = 3 — but expected 4. Divergence detected → fix: initialize
`first = {0: -1}` so the prefix at "before index 0" is recognized.
**Correct code emits 4** after the fix.
