"""Turn a Case-B (MODIFY_ACTION) skill into a Case-C (INJECT_CONTEXT) program.

Why this exists: a Case-B skill applies its correction directly, so a wrong
correction silently replaces a right answer and there is no fallback. The
detection is usually sound; the delivery is what removes the safety floor.

So the detection logic is kept and only the delivery changes. A wrapped PF runs
its original `should_activate` / `intervene`; when the result is a rewrite, the
rewrite is not applied — it is *stated* as evidence and the model redoes the
step itself:

    [code_split_whitespace @final] this solution calls `.split(' ')`, but the
    problem asks about whitespace-separated words, where `.split(' ')` keeps
    empty strings on consecutive spaces. The correct call is `.split()`.
    Apply this and give the corrected answer.

That is the shape of every PF that ever rescued: a concrete claim about
something the model itself wrote, plus the corrected value.

Two properties the wrapper must preserve:

  * **fallback-to-original.** Injection cannot corrupt an answer the way a
    rewrite can — the model may ignore the evidence and re-commit what it had.
    This is what keeps broke at 0, and it is why this direction is safe.
  * **an empty NOOP reason.** `_build_feedback` prints `[sid] reason` for every
    *activated* PF when no injection was produced, so a NOOP carrying
    "no_change" would manufacture junk Case-C feedback out of a PF that decided
    to do nothing. Several wrapped PFs do exactly that; the wrapper blanks it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from skills_agent.skills.program_functions import (
        Intervention, InterventionType, ProgramFunction, register_pf, _PF_REGISTRY,
    )
except ImportError:  # pragma: no cover — the runtime uses whichever path resolves
    from src.skills_agent.skills.program_functions import (
        Intervention, InterventionType, ProgramFunction, register_pf, _PF_REGISTRY,
    )

logger = logging.getLogger(__name__)

from skills.pf_template import REDO_TEXT as _REDO, format_change as _fmt_change  # single source


def _fmt_change(original: str, proposed: str) -> str:
    """State the change concretely, and briefly.

    Short values are quoted in full — that is the whole verdict. Long ones
    (code) are reduced to the lines that actually differ, because injecting a
    whole rewritten program invites the model to copy it back without reading
    it, and a verdict it does not read is a verdict that does not work.
    """
    o, p = (original or "").strip(), (proposed or "").strip()
    if not p:
        return ""
    if len(o) <= 120 and len(p) <= 120:
        return f"it should be `{p}` rather than `{o}`."

    ol, pl = o.splitlines(), p.splitlines()
    same = set(ol)
    changed = [l for l in pl if l not in same][:6]
    if changed and len("\n".join(changed)) <= MAX_SHOW:
        body = "\n".join("    " + l.strip() for l in changed)
        return "the corrected form differs here:\n" + body
    return f"the corrected form is:\n{p[:MAX_SHOW]}"


def as_injection(skill_id: str, *, note: str = "") -> Optional[type]:
    """Re-register `skill_id` so its rewrites become injected evidence.

    Must run AFTER the original registration (the executable half is loaded
    last, so a later `register_pf` of the same id wins). Returns the new class,
    or None when the id is not registered — a missing id is logged, never
    raised, so one stale entry cannot stop a library from loading.
    """
    inst = _PF_REGISTRY.get(skill_id)
    if inst is None:
        logger.warning("inject_adapter: %s is not registered; nothing to wrap", skill_id)
        return None
    base = type(inst)

    class _Injected(base):                       # type: ignore[misc, valid-type]
        """`base`, with MODIFY_ACTION rerouted to INJECT_CONTEXT."""
        _wrapped_from = base
        _note = note

        def intervene(self, step_context, action_type, arg, helper=None):
            try:
                from skills.pf_template import call_base_intervene
                iv = call_base_intervene(base, self, step_context, action_type, arg, helper)
            except Exception as e:               # a wrapped PF must not break dispatch
                logger.warning("inject_adapter: %s raised %s", skill_id, e)
                return Intervention(type=InterventionType.NOOP, skill_id=skill_id, reason="")

            if iv is None:
                return Intervention(type=InterventionType.NOOP, skill_id=skill_id, reason="")

            if iv.type is InterventionType.MODIFY_ACTION:
                where = "@final" if (action_type or "").upper() == "FINAL" else "@step"
                reason = (iv.reason or "").strip()
                # Two different rewrites hide behind MODIFY_ACTION, and they
                # need different sentences. Rewriting the ARG proposes a better
                # value; rewriting the action TYPE (FINAL -> SEARCH) says the
                # rollout is not entitled to answer yet.
                new_type = (iv.new_action_type or action_type or "").upper()
                is_control = new_type != (action_type or "").upper()
                change = "" if is_control else _fmt_change(arg, iv.new_action_arg or "")
                if not change and not reason:
                    # nothing concrete to say — silence beats a vague nudge
                    return Intervention(type=InterventionType.NOOP, skill_id=skill_id, reason="")
                parts = [f"[{skill_id} {where}]"]
                if self._note:
                    parts.append(self._note)
                elif reason:
                    parts.append(reason.replace("_", " ") + ".")
                if is_control:
                    parts.append(f"Do not answer yet — take a {new_type} action first, "
                                 f"then answer from what it returns.")
                else:
                    if change:
                        parts.append(change)
                    parts.append(_REDO)
                step_context.setdefault("pf_anchors", []).append(
                    dict(pf=skill_id, level="final" if where == "@final" else "step",
                         via="inject_adapter"))
                return Intervention(type=InterventionType.INJECT_CONTEXT,
                                    context_text=" ".join(parts),
                                    reason="rewrite stated as evidence",
                                    skill_id=skill_id)

            if iv.type is InterventionType.NOOP and (iv.reason or ""):
                # a non-empty NOOP reason is printed as feedback by
                # _build_feedback; a PF that chose to stay silent must stay silent
                return Intervention(type=InterventionType.NOOP, skill_id=skill_id, reason="")
            return iv

    _Injected.__name__ = f"Injected_{base.__name__}"
    _Injected.__qualname__ = _Injected.__name__
    register_pf(skill_id)(_Injected)
    return _Injected


def as_injections(skill_ids, notes: Optional[Dict[str, str]] = None) -> List[str]:
    """Wrap many; returns the ids actually wrapped."""
    notes = notes or {}
    done = []
    for sid in skill_ids:
        if as_injection(sid, note=notes.get(sid, "")) is not None:
            done.append(sid)
    if done:
        logger.info("inject_adapter: %d Case-B PFs now inject evidence: %s",
                    len(done), ", ".join(done))
    return done
