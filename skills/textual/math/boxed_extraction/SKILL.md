---
skill_id: boxed_extraction
name: Clean Answer Extraction
version: 1
priority: 0.9
error_category: final_format_error
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  When the FINAL answer is a sentence or explanation wrapped around the real
  value, extract the clean value (last \boxed{...}, "answer is X", or last
  numeric/fraction token) and submit only that.
anchor:
  level: final
  trigger: "a verbose answer carrying an explicit \\boxed{} or 'answer is' marker"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: []
    priority_boost: 0.4
    action: extract_clean_answer
    action_params: {}
---

# Clean Answer Extraction

A large share of wrong-answer math failures are cases where the reasoning
reached the correct value but the FINAL action emitted a verbose sentence
("Therefore the answer is \boxed{7}.") instead of the bare value. The grader
compares the extracted answer, so the surrounding prose can cause a miss.

## Detection Triggers
- FINAL arg longer than a bare value (>40 chars) or containing prose
  ("the", "answer", "therefore", "so", "thus")
- A `\boxed{...}` or "answer is X" pattern embedded in a longer string
- A trailing explanation after the value

## Avoidance Strategies
- Submit ONLY the final value as the FINAL action argument.
- Prefer the last `\boxed{...}` content; else the value after "answer is";
  else the last numeric / fraction token.
- Do not include the derivation, units, or commentary in FINAL.

## Phase: pre_final
Is your FINAL argument just the value, or is it a sentence? Strip everything
except the answer itself.

## Examples
### Example 1
**Wrong:** `Action: FINAL(After simplifying, the answer is \boxed{7}.)`
**Correct:** `Action: FINAL(7)`
