---
skill_id: substitution_invalid
name: Valid Substitution Check
version: 1
priority: 0.70
error_category: substitution_invalid
applicable_modes: [all]
applicable_phases: [think, answer]
system_summary: >
  Variable substitutions must preserve the equality / inequality they act
  on; inverse transforms must be properly handled.
phases:
  pre_final:
    conditions: [has_substitution]
    priority_boost: 0.3
    action: verify_substitution
    action_params: {}
---

# Valid Substitution Check

When you substitute `u = g(x)` to simplify, the map between `x` and `u` must be carefully tracked. Common errors: forgetting to transform the domain, losing multi-valued branches, or not reverting to the original variable.

## Detection Triggers
- Trigonometric substitution (`x = sin θ`, `u = tan(x/2)`)
- Substitution that's not injective (e.g. `u = x²`) — need to split into branches
- Change of variable in definite integrals — bounds must transform too
- Polynomial substitution `y = x² + x` — introduces spurious solutions
- Limit computation after variable substitution

## Avoidance Strategies
- Note the **domain** of the new variable and how the old one maps to it
- For non-injective substitutions, split into monotone branches
- After computing in new variable, substitute back carefully
- For inequalities: check monotonicity of substitution over domain
- Definite integrals: transform both limits AND differential

## Phase: pre_final
Did your substitution preserve the question's domain? Did you translate the final answer back to the original variable? If the substitution wasn't injective, did you handle all branches?

## Examples
### Example 1
**Scenario:** Solve `x⁴ - 5x² + 4 = 0` via `y = x²`.
**Wrong:** Solve `y² - 5y + 4 = 0` → `y = 1, 4` → FINAL(1, 4). 
**Correct:** Substitute back: `x² = 1` → `x = ±1`; `x² = 4` → `x = ±2`. Four solutions.

### Example 2
**Scenario:** Evaluate ∫₀^π sin(x) dx with u = sin(x), du = cos(x) dx.
**Wrong:** `du/cos(x)` and get stuck — substitution not useful here
**Correct:** Direct integration: [-cos(x)]₀^π = 2.
