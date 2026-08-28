---
skill_id: claimed_unique_solution_search
name: Claimed-Unique-Solution Search
version: 1
priority: 0.85
error_category: case_incompleteness
applicable_phases: [think]
system_summary: >
  When a step asserts that some solutions are the only ones, brute-force a
  small range for counterexamples; an uncovered solution is evidence.
anchor:
  level: step
  trigger: "a uniqueness claim — 'the only', 'unique', 'exactly one'"
  evidence: "executed"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  per_action:
    conditions: [claims_unique_solution]
    action: search_counterexample
    action_params: {timeout_s: 10}
---
# Claimed-Unique-Solution Search
case / algebra families (aime24_7: "by symmetry only a=b=c=100 works" — misses
(0,100,200) and (99,99,102)). Anchor = phrases "the only solution(s)", "only
when", "no other solutions", "unique solution". Evidence = the judge writes a
bounded brute-force search (sandboxed); any solution not covered by the claim
is the evidence. Action: regenerate from this step with the counterexamples
injected. Shares the sandbox with counting_small_case_check.
