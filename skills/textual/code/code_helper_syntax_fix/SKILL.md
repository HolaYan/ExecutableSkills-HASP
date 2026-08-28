---
skill_id: code_helper_syntax_fix
name: Teacher Syntax Fix on FINAL
version: 1
priority: 0.6
error_category: syntax_error
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Before submitting FINAL, ensure your Python compiles. Common syntax
  pitfalls: unmatched brackets/quotes, mixed tabs and spaces, missing
  colons after `def`/`if`/`for`, dangling commas in function signatures,
  unterminated triple-quoted strings. If your code does not parse the
  sandbox returns 0 on every test — costs an entire sample.
anchor:
  level: final
  trigger: "the submission does not parse as Python"
  evidence: "helper"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.1
---

# Teacher Syntax Fix on FINAL

Roughly 1-3% of LCB failures are pure SyntaxError — code that doesn't
even compile. These are usually small typos (a missing `]`, an extra
`:` somewhere, mixed indentation) that render the entire submission
worthless. This skill is the fallback PF: it runs `ast.parse` on the
draft FINAL, and if it raises SyntaxError, asks the PF helper to
make the smallest possible edit that lets it parse.

## Detection Triggers
- Draft FINAL contains Python code
- `ast.parse(code)` raises SyntaxError

## The Protocol
1. Locate the line `ast.parse` flagged.
2. Make the smallest possible edit — just enough for the file to parse.
3. Do not refactor or re-derive the algorithm; preserve every other line.
4. Re-emit FINAL with the corrected code.

## What this skill IS NOT for
- Algorithm errors (covered by `code_sandbox_quick_check`).
- Format mismatch like `class Solution` for stdin problems (covered by
  `code_pick_format`).
