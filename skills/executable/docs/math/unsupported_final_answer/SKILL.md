---
skill_id: unsupported_final_answer
name: Unsupported Final Answer
version: 1
priority: 0.80
error_category: premature_final
applicable_modes: [all]
applicable_phases: [answer]
system_summary: >
  Refuse a final answer that the text admits was guessed or taken from a
  "known result" instead of derived; demand the explicit derivation.
anchor:
  level: final
  trigger: "a guess phrase right before the answer is committed"
  evidence: "deterministic"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [guess_phrase_before_finish]
    action: demand_derivation
    action_params: {window_chars: 1200}
---

# Unsupported Final Answer

A large share of committed-wrong rollouts have no locatable error step:
the model never derived the answer. Tails read "I will go with 12", "given
the time, I'll go with", "in similar problems the answer is typically 16",
"this is a known problem, the answer is 1/2".

## Anchor
Fires on the step that commits `finish[...]` when one of the guess/abandon
phrases occurs within 1200 chars before it.

## Action change
Do not accept the finish. Inject "the final answer is committed right after
the phrase '…' — it was not derived; carry out the computation explicitly"
and regenerate from that point. Requires the model's consent (dual gate) and
the fallback gate — unlike compute_observation_verify it is not provable.

## Precision

The trigger is a phrase the model wrote immediately before committing — a guess
marker, not a property of the problem. Measure its rate on solutions that
already pass before relying on it.

