---
skill_id: code_teacher_logic_fix
name: Teacher-Reviewed Logic Rescue on Failed Public Examples
version: 1
priority: 0.95
error_category: wrong_logic
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  Before treating your draft as final, run it mentally on every `>>>` /
  `assert` example shown in the docstring. If even one fails, the
  algorithm is wrong on a visible case; the hidden tests will fail too.
  Fix the divergence point before submitting.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.3
---

# Teacher-Reviewed Logic Rescue on Failed Public Examples

When the student's FINAL fails the docstring-derived `public_test_code`,
this PF asks a teacher LLM (GPT-4o) to rewrite the function. The teacher
sees the original problem, the student's broken code, and the failing-test
diagnostic; it returns a corrected implementation. The PF runs the teacher's
rewrite against the same public examples and only swaps it in if it passes
— never replacing a candidate that already works.

## Why this exists
- 95% of remaining HE+/MBPP+ regressions vs direct-answer baseline are real
  algorithmic errors. Surgical regex PFs can't fix wrong logic.
- `code_teacher_syntax_fix` only handles SyntaxError; it doesn't help when
  the code parses but computes the wrong answer.
- The teacher LLM has the capacity to read the problem and produce a
  corrected algorithm given the failing trace.

## Activation
1. FINAL action in the code domain.
2. `step_context.public_test_code` is non-empty (HumanEval/MBPP rows).
3. Candidate fails `evaluate_with_test_script(code, public_test_code)`.

## Replacement criterion
Teacher output must:
- Parse with `ast.parse`.
- Pass `evaluate_with_test_script(teacher_code, public_test_code)`.

If either check fails, the original (broken) FINAL stands; we do not
fall back to a still-buggy helper rewrite.
