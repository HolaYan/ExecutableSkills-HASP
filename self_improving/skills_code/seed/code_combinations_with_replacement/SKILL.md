---
skill_id: code_combinations_with_replacement
name: Allow Repetition in Combinations When Searching for Multiplicative Decompositions
version: 1
priority: 0.7
error_category: combinatorics_repetition
applicable_modes: [all]
applicable_phases: [pre_final]
system_summary: >
  When the problem asks for k items whose PRODUCT equals N (e.g. "is N a
  product of 3 primes"), use `itertools.combinations_with_replacement`,
  NOT `itertools.combinations`. The latter forbids picking the same element
  twice, so 8 = 2*2*2 will be missed.
phases:
  pre_final:
    conditions: []
    priority_boost: 0.15
---

# Allow Repetition in Combinations When Searching for Multiplicative Decompositions

A surgical PF replaces `combinations(seq, k)` with
`combinations_with_replacement(seq, k)` when the prompt mentions
"multiply" / "product" / "3 prime numbers" or shows examples like 8 = 2*2*2.

## When the auto-fix fires
1. Code uses `combinations(...)` from itertools.
2. Prompt suggests repeated elements are allowed.
3. Code is not already using `combinations_with_replacement`.

## What gets patched
Both the import and the call-site:
- `from itertools import combinations` → `from itertools import combinations_with_replacement`
- `combinations(...)` → `combinations_with_replacement(...)`

## What this protects against
HE+ HumanEval_75-style problems where the test includes inputs that
factorize with repeated factors (e.g. `is_multiply_prime(8) == True`).
