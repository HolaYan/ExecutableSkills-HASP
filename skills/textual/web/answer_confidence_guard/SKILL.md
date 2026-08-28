---
skill_id: answer_confidence_guard
name: Answer Confidence Guard
version: 1
priority: 0.7
error_category: answer_confidence_guard
applicable_modes: [all]
applicable_phases: [think, read, answer]
system_summary: >
  When a later step abandons an answer that earlier evidence supported, compare
  the two against the documents actually read and keep the better-supported one.
anchor:
  level: final
  trigger: "the committed answer differs from one stated earlier after a read"
  evidence: "helper"
  action: "inject the verdict and redo the committed answer"
phases:
  pre_final:
    conditions: [has_read]
    priority_boost: 0.15
    action: verify_answer_confidence
    action_params: {}
---

# Answer Confidence Guard

Catches over-refinement: the agent reads a source, states the right answer, then
keeps searching and talks itself into a worse one. The regression is invisible
to every check that looks only at the final answer, because the final answer is
self-consistent — the evidence for the *earlier* answer is what got dropped.

## Anchor

This PF anchors on a claim the model itself wrote: a candidate answer asserted
in an earlier `Thought` ("the answer is X", "X is the answer", "therefore X"),
recorded only when that step had already read a document. At FINAL, if the
committed answer differs from that candidate, the PF fires. A rollout that
never stated an earlier candidate, or whose candidate matches the FINAL, is
never touched.

## Detection Triggers

- An earlier step, taken after a READ, states a candidate answer
- The committed FINAL differs from it (case- and whitespace-insensitive)
- The extracted candidate is between 2 and 99 characters (longer matches are
  sentences, not answers, and are discarded)

## Avoidance Strategies

- When you change an answer you already stated, say what new evidence forced
  the change; if there is none, the earlier answer stands
- Additional search that returns nothing relevant is not a reason to revise
- Re-read the passage that supported the first answer before abandoning it

## Evidence

Helper-backed, not deterministic. The PF helper receives the question, both
answers, and the tail of the documents read, and must reply with exactly one of
`KEEP_OLD` / `USE_NEW`. Only `KEEP_OLD` produces an intervention
(`MODIFY_ACTION` back to the earlier answer); anything else — including a
failed or unparseable helper call — leaves the rollout untouched.

## Phase: pre_final

ANSWER CONFIDENCE CHECK:
1. Did you state a different answer earlier, after reading a source?
2. If so, name the evidence that made you change it.
3. If you cannot name any, return the earlier answer.

## Examples

### Example 1
**Scenario:** Reads a page giving the director as "Bong Joon-ho", then searches
twice more, finds a producer's name, and commits that instead.
**Wrong:** Commit the producer's name because it appeared more recently
**Correct:** Keep "Bong Joon-ho" — nothing in the later results contradicted it

### Example 2
**Scenario:** Reads that a treaty was signed in 1919, then sees an unrelated
1920 date in a snippet and switches.
**Wrong:** Revise to 1920 on snippet proximity alone
**Correct:** Keep 1919 — the read document is stronger evidence than a snippet

### Example 3
**Scenario:** States "Tokyo" after reading, then changes to "Greater Tokyo Area"
after further reading that genuinely distinguishes the two.
**Wrong:** Revert to "Tokyo" mechanically because it came first
**Correct:** Accept the change — this is a real correction, and the helper
should answer `USE_NEW`

## Status

Registered in `skills/textual/web/dynamic_program_functions.py`. This card was
missing until 2026-08-25, so the PF was registered but never appeared in the
pf_select menu and could not be selected. Its rescue/broke rates are therefore
**unmeasured** — treat it as untested until it has been screened against a
correct-set control.
