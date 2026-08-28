"""Multi-point pf_select: fire at every step boundary, on dual consent.

Today's pf_select calls `exec_pf` **once**, at FINAL. This module adds the
step-level channel the design asks for: at each action boundary, intervene only
when the anchor and the model *both* say this step needs it.

    for each step boundary of the rollout:
        anchor side — does any skill's Detect fire on THIS step?
        model side  — shown the fired skill and the claim it questions,
                      does the model name a concrete error in that claim?
        both agree  → Repair produces evidence for that step
    FINAL boundary  → today's dispatch, unchanged

The FINAL path is untouched on purpose. It is the configuration every measured
gain came from, so the step channel is strictly additive: if no step reaches
dual consent, the rollout is exactly what it is today.

## Why the consent prompt is written the way it is

An earlier version of this gate consented to almost every anchor-fired step —
it was not a gate at all. The cause was the question it asked:

    "Select the PFs that are genuinely relevant to this step."

Relevance is nearly always true. Any step containing numbers makes "check the
arithmetic" relevant, and presenting the whole PF menu turns it into a picking
task, which biases toward picking. The intersection with the anchor's own fires
then made agreement almost automatic.

So this module asks a different question. It shows only the skill that fired
and the specific claim the step makes, asks whether that claim is *wrong*, and
requires the model to name the error. `NO` is the default and an unparseable
answer counts as `NO`. Consent becomes a verification act rather than a
selection, which is what makes it capable of disagreeing.

Nothing here is measured yet. `HASP_STEP_DISPATCH` is off by default.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ENABLED = os.environ.get("HASP_STEP_DISPATCH", "0") not in ("0", "false", "False")

#: How many step-level interventions one rollout may accumulate. The evidence
#: for each is injected together, and a long list of complaints reads as noise
#: rather than as a correction, so this stays small.
MAX_STEPS_PER_ROLLOUT = int(os.environ.get("HASP_STEP_MAX", "2"))

CONSENT_TMPL = (
    "Here is a problem and one step from a solution to it.\n\n"
    "Problem:\n{question}\n\n"
    "Earlier steps (abbreviated):\n{prefix}\n\n"
    "The step in question:\n{step}\n\n"
    "A check called `{skill_id}` looks for this specific failure:\n  {scope}\n\n"
    "Is the step wrong in that specific way?\n"
    "Answer on the first line with exactly one of:\n"
    "  NO\n"
    "  YES: <the specific error, naming the wrong value or claim>\n\n"
    "Answer NO unless you can name the error. A step that is merely incomplete, "
    "or that you would have written differently, is not wrong.\n\nAnswer:"
)

_YES = re.compile(r"^\s*YES\s*[:\-]\s*(.+)", re.I)


def parse_consent(text: str) -> Optional[str]:
    """-> the named error when the model consents, else None.

    Anything that is not an explicit `YES: <reason>` counts as refusal — an
    empty generation, a hedge, or a restatement of the step. Silence must mean
    no, or the gate drifts back to consenting to everything.
    """
    first = (text or "").strip().splitlines()
    if not first:
        return None
    m = _YES.match(first[0])
    if not m:
        return None
    reason = m.group(1).strip()
    return reason if len(reason) >= 8 else None


@dataclass
class StepHit:
    """One (step, skill) pair where the anchor fired."""
    step_idx: int
    n_steps: int
    skill_id: str
    char_start: int
    step_text: str
    scope: str = ""
    consent_reason: Optional[str] = None
    evidence: str = ""

    @property
    def agreed(self) -> bool:
        return bool(self.consent_reason)


def _library():
    from skills.pf_template import steps as _steps
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore
    return _PF_REGISTRY, _steps


def anchor_hits(question: str, response: str, active_ids: Sequence[str],
                domain: str = "math", max_per_rollout: int = 8) -> List[StepHit]:
    """Anchor side — run each selected skill's Detect against each step.

    A skill is asked about a step in isolation: the step is the `raw_reasoning`
    it sees, so a step-level Detect answers about *that* step rather than about
    the whole rollout. Skills anchored at `final` are skipped here; the FINAL
    dispatch already owns them.
    """
    registry, split = _library()
    hits: List[StepHit] = []
    all_steps = split(response)
    for st in all_steps:
        for sid in active_ids:
            pf = registry.get(sid)
            if pf is None:
                continue
            anchor = getattr(type(pf), "pf_anchor", None)
            if anchor is None or anchor.level != "step":
                continue
            ctx = {"question": question, "raw_reasoning": st.text, "thought": st.text,
                   "domain": domain, "step_index": st.idx, "_pf_fire_counts": {}}
            try:
                if not pf.should_activate(ctx, "STEP", ""):
                    continue
            except Exception:
                continue
            hits.append(StepHit(step_idx=st.idx, n_steps=len(all_steps), skill_id=sid,
                                char_start=st.char_start, step_text=st.text,
                                scope=getattr(type(pf), "pf_summary", "")))
            if len(hits) >= max_per_rollout:
                return hits
    return hits


def build_consent_prompts(question: str, response: str, hits: Sequence[StepHit],
                          prefix_chars: int = 2000, step_chars: int = 2000) -> List[str]:
    """One verification question per hit — no menu, no selection."""
    out = []
    for h in hits:
        prefix = response[: h.char_start]
        out.append(CONSENT_TMPL.format(
            question=question,
            prefix=(prefix[-prefix_chars:] if len(prefix) > prefix_chars else prefix).strip() or "(start)",
            step=h.step_text.strip()[:step_chars],
            skill_id=h.skill_id, scope=h.scope or "(no summary)"))
    return out


def apply_consent(hits: Sequence[StepHit], generations: Sequence[str]) -> List[StepHit]:
    """Record the model's verdict; return only the hits it consented to."""
    for h, g in zip(hits, generations):
        h.consent_reason = parse_consent(g)
    return [h for h in hits if h.agreed]


def repair_agreed(question: str, response: str, agreed: Sequence[StepHit],
                  domain: str = "math",
                  max_steps: int = MAX_STEPS_PER_ROLLOUT) -> List[StepHit]:
    """Repair side — the skill produces its evidence for the consented step.

    The skill's own Repair runs, so a deterministic checker still gets the last
    word: where it finds nothing concrete, the model's stated reason is used
    instead, tagged so it is never mistaken for a recomputed verdict.
    """
    registry, _ = _library()
    done: List[StepHit] = []
    for h in sorted(agreed, key=lambda x: x.step_idx)[:max_steps]:
        pf = registry.get(h.skill_id)
        if pf is None:
            continue
        ctx = {"question": question, "raw_reasoning": h.step_text, "thought": h.step_text,
               "domain": domain, "step_index": h.step_idx, "_pf_fire_counts": {}}
        try:
            iv = pf.intervene(ctx, "STEP", "")
        except Exception:
            iv = None
        text = getattr(iv, "context_text", "") if iv is not None else ""
        if text and "Before finalizing, double-check" not in text:
            h.evidence = text
        else:
            # No recomputed verdict for this step — carry the model's own
            # reason, marked as such.
            h.evidence = (f"[{h.skill_id} @step {h.step_idx + 1}/{h.n_steps}] "
                          f"reviewing this step you identified: {h.consent_reason}")
        done.append(h)
    return done


def feedback_block(hits: Sequence[StepHit]) -> str:
    """The injected text for the step channel, ordered by position."""
    if not hits:
        return ""
    lines = [h.evidence for h in sorted(hits, key=lambda x: x.step_idx) if h.evidence]
    if not lines:
        return ""
    return ("The following steps were checked and found to contain errors:\n\n"
            + "\n\n".join(lines))


def run(question: str, response: str, active_ids: Sequence[str], generate: Callable,
        domain: str = "math", max_per_rollout: int = 8,
        max_steps: int = MAX_STEPS_PER_ROLLOUT) -> Tuple[str, Dict[str, Any]]:
    """The whole step channel for one rollout. -> (feedback, trace).

    `generate(prompts) -> list[str]` is supplied by the caller so this module
    stays free of any inference backend and can be tested without a GPU.
    """
    hits = anchor_hits(question, response, active_ids, domain, max_per_rollout)
    trace: Dict[str, Any] = {"anchor_hits": len(hits), "consented": 0, "repaired": 0,
                             "steps": []}
    if not hits:
        return "", trace
    gens = generate(build_consent_prompts(question, response, hits))
    agreed = apply_consent(hits, gens)
    trace["consented"] = len(agreed)
    if not agreed:
        return "", trace
    repaired = repair_agreed(question, response, agreed, domain, max_steps)
    trace["repaired"] = len(repaired)
    trace["steps"] = [dict(step=h.step_idx, skill=h.skill_id,
                           reason=(h.consent_reason or "")[:120]) for h in repaired]
    return feedback_block(repaired), trace


def run_batch(items: Sequence[Tuple[str, str, Sequence[str]]], generate_batch: Callable,
              domain: str = "math", max_per_rollout: int = 8,
              max_steps: int = MAX_STEPS_PER_ROLLOUT
              ) -> Tuple[List[str], List[Dict[str, Any]]]:
    """The step channel over many rollouts, with ONE batched consent call.

    `items` is (question, response, active_skill_ids) per rollout and
    `generate_batch(prompts) -> list[str]` is called exactly once with every
    consent question from every rollout. Per-rollout generation would be one
    engine round-trip per rollout, which is unusable at n=64.

    -> (feedback per rollout, trace per rollout). A rollout with no anchor hit,
    or none the model consents to, gets an empty string and is left exactly as
    it was — this channel can only add.
    """
    per_hits: List[List[StepHit]] = []
    prompts: List[str] = []
    spans: List[Tuple[int, int]] = []          # (start, end) into prompts
    for question, response, ids in items:
        hits = anchor_hits(question, response, ids, domain, max_per_rollout)
        per_hits.append(hits)
        start = len(prompts)
        prompts.extend(build_consent_prompts(question, response, hits))
        spans.append((start, len(prompts)))

    gens = generate_batch(prompts) if prompts else []
    if len(gens) < len(prompts):               # a short batch must not misalign
        gens = list(gens) + [""] * (len(prompts) - len(gens))

    feedback: List[str] = []
    traces: List[Dict[str, Any]] = []
    for (question, response, _ids), hits, (a, b) in zip(items, per_hits, spans):
        tr: Dict[str, Any] = {"anchor_hits": len(hits), "consented": 0, "repaired": 0,
                              "steps": []}
        if not hits:
            feedback.append("")
            traces.append(tr)
            continue
        agreed = apply_consent(hits, gens[a:b])
        tr["consented"] = len(agreed)
        if not agreed:
            feedback.append("")
            traces.append(tr)
            continue
        repaired = repair_agreed(question, response, agreed, domain, max_steps)
        tr["repaired"] = len(repaired)
        tr["steps"] = [dict(step=h.step_idx, skill=h.skill_id,
                            reason=(h.consent_reason or "")[:120]) for h in repaired]
        feedback.append(feedback_block(repaired))
        traces.append(tr)
    return feedback, traces


def summarise(traces: Sequence[Dict[str, Any]]) -> str:
    """One line for the eval log — consent rate is the number to watch.

    An earlier step-level design consented to nearly every anchor-fired step,
    which is what made it a worse intervention than the FINAL path. If this
    figure comes back near 100%, the gate is not gating and the rest of the
    numbers do not matter yet.
    """
    n = len(traces) or 1
    hits = sum(t["anchor_hits"] for t in traces)
    yes = sum(t["consented"] for t in traces)
    rep = sum(t["repaired"] for t in traces)
    with_fb = sum(1 for t in traces if t["repaired"])
    rate = (yes / hits) if hits else 0.0
    return (f"[step] {hits} anchor hits over {n} rollouts → consented {yes} "
            f"({rate:.0%}) → repaired {rep}; {with_fb}/{n} rollouts got step evidence")
