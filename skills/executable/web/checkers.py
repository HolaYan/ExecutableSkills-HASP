"""web — pure checker functions.

Extracted from the old `evidence_pfs.py`, which mixed three things: these
functions, a PF base class, and the registrations. Registration now lives in
`skills.py` (one declaration per skill) and this file holds only what a checker
needs to be: a function from (context, answer) to a verdict string, with no
knowledge of the PF runtime.

Imported by `skills.py` and by the offline harnesses in `anchor/` — the same
implementation the measured numbers came from.
"""


from __future__ import annotations


import re


import sys


from pathlib import Path


from typing import Any, Dict, Optional


_HASP = Path(__file__).resolve().parents[3]


if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))


try:
    from skills_agent.skills.program_functions import Intervention, InterventionType, ProgramFunction, register_pf
except ImportError:  # pragma: no cover
    from src.skills_agent.skills.program_functions import Intervention, InterventionType, ProgramFunction, register_pf


def _norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def _contains(hay, needle):
    h, n = " ".join(_norm(hay)), " ".join(_norm(needle))
    return bool(n) and n in h


_YN = re.compile(r"^\s*(yes|no|true|false|none|neither|both|same|different)\b", re.I)


_NUMERIC = re.compile(r"^[\d.,%/ -]+$")


def _observations(ctx: Dict[str, Any]) -> str:
    parts = [str(ctx.get("all_read_contents") or ""), str(ctx.get("last_search_results_text") or "")]
    for h in ctx.get("action_history") or []:
        if isinstance(h, dict):
            parts.append(str(h.get("observation") or h.get("obs") or h.get("result") or ""))
        else:
            parts.append(str(h))
    return " ".join(parts)


def answer_grounding(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Deterministic: the committed answer must occur (per part) in some observation."""
    ans = str(answer or "").strip()
    if not ans or _YN.match(ans) or _NUMERIC.match(ans):
        return None
    # Strip format shells first: "**Answer:** Paris." IS grounded if Paris is
    # -- flagging it as ungrounded was this checker's worst false verdict, on
    # a case that belongs to format_extraction_error anyway.
    core = re.sub(r"^\**\s*(?:final\s+|short\s+)?answers?\s*\**\s*[:\-]\s*|"
                  r"^the\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*|[`*_#]", "", ans, flags=re.I).strip(" .`")
    obs = _observations(ctx)
    if not obs.strip():
        return None
    parts = [p.strip() for p in re.split(r",|;| and | or |\(|\)", core) if p.strip()] or [core]
    if any(_contains(obs, p) for p in parts):
        return None
    # Say what the evidence DOES hold, not only what it lacks: a bare "not
    # grounded" left the model re-arguing for its answer, while a free-form
    # re-read (the control) found the right one. Candidates give it the same
    # material, plus the diagnosis.
    ents = re.findall(r"[A-Z][\w'\u2019.-]+(?:\s+[A-Z][\w'\u2019.-]+)*", obs)
    nums = re.findall(r"\b\d{2,4}\b", obs)
    seen, cand = set(), []
    for t in ents + nums:
        tl = t.lower()
        if tl not in seen and tl not in core.lower() and len(t) > 2:
            seen.add(tl); cand.append(t)
        if len(cand) >= 6: break
    hint = f" The evidence mentions: {', '.join(cand)}." if cand else ""
    return (f"the final answer {core[:80]!r} does not appear in any retrieved evidence (search "
            f"results or read pages).{hint} Re-answer using only what the evidence "
            f"actually states")


# A capitalised run: one or more capitalised words, optionally joined by a
# connective ("University of Oxford"). The original form required the words to
# be glued to a connective with no plain-space alternative, so "Eiffel Tower"
# and every single-word title ("Parasite") matched nothing -- which is why the
# entity-coverage checkers hardly ever fired on real rollouts.
_ENT = re.compile(r"(?:\"([^\"]{3,80})\"|“([^”]{3,80})”"
                  r"|\b([A-Z][\w'’.-]+(?:\s+(?:(?:of|the|de|von|van|and|&|la|le|du)\s+)?"
                  r"[A-Z][\w'’.-]+)*))")


_CMP = re.compile(r"\b(which|who|whose)\b.*\b(first|earlier|earliest|older|oldest|younger"
                  r"|later|latest|more|most|less|least|larger|largest|longer|longest|higher"
                  r"|highest|lower|lowest|taller|tallest|shorter|shortest|bigger|biggest"
                  r"|smaller|smallest|faster|fastest|slower|heavier|deeper|wider|before|after)\b", re.I)


def _question_entities(q: str):
    out = []
    for m in _ENT.finditer(q):
        e = next(g for g in m.groups() if g)
        e = re.sub(r"^(Which|What|Who|Where|When|How|The|In|On|Of|Does|Did|Is|Are|Was|Were)\s+", "", e).strip()
        # Single-word titles ("Parasite") are entities too; what gets dropped
        # is a bare question word left over after the prefix strip.
        if e and e not in out and _norm(e) and \
                _norm(e)[0] not in {"when", "what", "who", "where", "which", "how",
                                    "why", "the", "a", "an", "is", "was", "did",
                                    "in", "on", "at", "by", "for", "from", "with",
                                    "to", "of", "and", "or", "if", "as", "it"}:
            out.append(e)
    return out


def _queries(ctx: Dict[str, Any]) -> str:
    qs = []
    for h in ctx.get("action_history") or []:
        if isinstance(h, dict):
            qs.append(str(h.get("query") or h.get("arg") or h.get("action_arg") or ""))
    return " ".join(qs)


def question_entity_coverage(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Anchor = the question entity that was never searched for nor retrieved."""
    ents = _question_entities(str(ctx.get("question", "")))
    if not ents:
        return None
    seen = _queries(ctx) + " " + _observations(ctx)
    missing = [e for e in ents if not _contains(seen, e)]
    if not missing or len(missing) == len(ents):   # all missing = observations not exposed; stay silent
        return None
    return dict(
        verdict=(f"the question names {missing[0]!r} but no search query or retrieved page mentions it; "
                 f"the answer cannot be complete without evidence about {missing[0]!r}"),
        search=missing[0])


def comparison_evidence_completeness(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Comparison questions need evidence on BOTH sides; anchor = the side without evidence."""
    q = str(ctx.get("question", ""))
    if not _CMP.search(q):
        return None
    ents = _question_entities(q)
    if len(ents) < 2:
        return None
    obs = _observations(ctx)
    if not obs.strip():
        return None
    have = [e for e in ents if _contains(obs, e)]
    if len(have) >= 2 or not have:
        return None
    missing = [e for e in ents if e not in have][0]
    return dict(
        verdict=(f"this is a comparison question, but the retrieved evidence only covers {have[0]!r} and "
                 f"nothing about {missing!r}; the comparison cannot be decided from one side"),
        search=missing)
