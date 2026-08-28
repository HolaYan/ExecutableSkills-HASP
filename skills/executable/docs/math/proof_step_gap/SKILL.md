---
skill_id: proof_step_gap
name: Proof Step Justification
version: 1
priority: 0.65
error_category: proof_step_gap
applicable_modes: [all]
applicable_phases: [think]
system_summary: >
  Each proof step must be explicitly justified. Avoid "clearly," "obviously,"
  or jumps that skip essential algebraic manipulation.
anchor:
  level: step
  trigger: "the reasoning leans on 'clearly' / 'obviously' / 'it follows'"
  evidence: "reminder"
  action: "inject the verdict and redo the work from the anchored step"
phases:
  pre_final:
    conditions: [has_proof]
    priority_boost: 0.25
    action: verify_proof_steps
    action_params: {}
---

# Proof Step Justification

Even in a numerical answer problem, reasoning chains that skip steps can hide errors. For MATH-500 problems that require derivation, gaps also cost partial credit.

## Detection Triggers
- "Clearly, …" / "Obviously, …" / "It follows that …" without explicit manipulation
- Large inferential leap: expressions transform by ≥2 non-trivial steps
- Citing a theorem without stating which
- Induction without base case or hypothesis
- Contradiction proof without stating the assumption
- "WLOG" without justifying why loss-of-generality is safe

## Avoidance Strategies
- For each non-trivial step, state the rule / theorem / manipulation used
- Induction: explicit base case, induction hypothesis, inductive step
- If a step "follows from symmetry," name the symmetry
- When simplifying, show the canonical form on both sides
- For WLOG: explain which cases the argument covers

## Phase: pre_final
Can someone who didn't solve this problem follow your reasoning from start to finish? Are there ≥2-step jumps that should be broken down?

## Examples
### Example 1
**Scenario:** Prove `∑k=1^n k² = n(n+1)(2n+1)/6`
**Wrong:** "Clearly this is the formula, so answer is 385 for n=10"
**Correct:** Derive via induction OR use the formula (state it), then plug n=10.

### Example 2
**Scenario:** WLOG assume x ≤ y ≤ z.
**Wrong:** "WLOG x ≤ y ≤ z, so …" without mentioning symmetry of variables
**Correct:** "The expression is symmetric in x, y, z. WLOG x ≤ y ≤ z."
