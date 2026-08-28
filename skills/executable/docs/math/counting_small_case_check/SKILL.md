---
skill_id: counting_small_case_check
name: Counting Small-Case Check
version: 1
priority: 0.90
error_category: counting_error
applicable_modes: [all]
applicable_phases: [think]
system_summary: >
  When a step states a parametric counting formula, falsify or confirm it by
  brute-force enumeration on a small instance; if the formula disagrees with
  the enumeration, redo the step with the enumerated values as evidence.
anchor:
  level: step
  trigger: "a counting or combinatorial claim with a small parameter"
  evidence: "executed"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [has_parametric_count]
    action: enumerate_small_case
    action_params: {max_n: 6, timeout_s: 10}
---

# Counting Small-Case Check

Counting errors in the mined wrong cases are structural, not arithmetic:
C(13−k, k) written where C(12−k, k−1) is right (amc23_38); "any two parallel
chords form a rectangle" (aime24_17); C(5,a) assignments where alternation
makes the assignment unique (aime24_14). A regex cannot see these — but every
one of them is a claim about a parametric family, so it can be tested on n=3,
4, 5 by enumeration.

## Anchor (step level)
Fires on a step that states a closed-form count with a free parameter
(`\binom{n-k}{k}`, `n!`, `2^n`, `C(n, k)`) next to counting language
("number of", "ways", "subsets", "paths", ...).

## Evidence (executed)
The judge model writes a short python snippet defining `claimed(n)` (the
step's formula) and `brute(n)` (direct enumeration of the objects the
problem defines) and prints both for small n. The sandbox runs it; a
mismatch is the evidence: "for n=4 the step's formula gives 35 but direct
enumeration gives 20". Agreement on all small n → OK, no intervention.

## Action change
Regenerate from this step with the enumerated values injected; the fallback
gate (two samples agree, same answer type) still applies.

## Notes
- Provable when the enumeration is faithful; the judge is told to enumerate
  the ORIGINAL objects, never to re-derive a formula.
- Requires `anchor/sandbox.py` (timeout, rlimits, no network).
