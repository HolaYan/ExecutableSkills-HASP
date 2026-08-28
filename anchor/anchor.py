"""Anchor v2 — locate the step that actually needs modification, then act there.

Why this exists
---------------
The current pf_select dispatch has no notion of WHERE an error is:

  - Case B rewrites the already-extracted final answer (format fixes only);
  - Case C appends a generic "[System Feedback] double-check your arithmetic"
    at the END of the rollout and regenerates the whole answer.

Over the paired base-model rollouts this was built from, that design fixed a
handful
committed-wrong answers. The stable gains came from the stall channel (PF
intervening where a rollout stopped and making it finish) plus the
fallback-to-original safety; this module targets the second channel.

Anchor v2 keeps the fallback safety but changes the intervention geometry:

  1. segment the committed reasoning into steps,
  2. run deterministic per-step checkers, take the EARLIEST step with a
     concrete, evidence-backed failure -> the anchor,
  3. truncate the trajectory at the anchor and regenerate from there with the
     evidence injected ("[Anchor] step 7 asserts 49680+7452-23 = 27792761,
     but it equals 57109. Redo this step."),
  4. verify-and-fallback: if the anchored branch does not commit a parseable
     answer, keep the original one (never break a committed answer).

This module is pure CPU / no vLLM: segmentation, checking, anchor location,
and prompt construction. Generation happens in the caller (eval harness or a
future pf_select_loop integration).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from hasp_config import protocol as _protocol

_P = _protocol()

# ── Step segmentation ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    idx: int
    char_start: int
    char_end: int
    text: str


_STEP_BOUNDARY = re.compile(
    r"(?:\n\s*\n)"                        # blank line
    r"|(?=\n#{2,4}\s)"                    # markdown heading
    r"|(?=\n(?:Step|STEP)\s*\d+)"         # explicit Step N
    r"|(?=\n-{3,}\n)"                     # hrule
)


def segment_steps(text: str, min_len: int = _P.segmentation.base_min_len,
                  max_steps: int = _P.segmentation.base_max_steps) -> list[Step]:
    """Split reasoning text into step-sized chunks.

    Paragraph-level segmentation: blank lines, markdown headings, "Step N"
    markers. A chunk shorter than `min_len` is folded into the step before it
    (and a short preceding step absorbs the one after), so one-line
    connectives ("So:", "---") don't become steps of their own.
    """
    if not text:
        return []
    raw: list[tuple[int, int]] = []
    pos = 0
    for m in _STEP_BOUNDARY.finditer(text):
        end = m.start()
        if end > pos:
            raw.append((pos, end))
        pos = m.end()
    if pos < len(text):
        raw.append((pos, len(text)))

    # merge short chunks forward
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and (e - s) < min_len:
            merged[-1] = (merged[-1][0], e)
        elif merged and (merged[-1][1] - merged[-1][0]) < min_len:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    steps = [Step(i, s, e, text[s:e]) for i, (s, e) in enumerate(merged[:max_steps])]
    return steps


# ── Deterministic per-step checkers ──────────────────────────────────────
#
# Each checker: (step) -> Optional[AnchorEvidence]. Return the FIRST concrete
# failure found in the step, with a human-readable verdict that can be
# injected into the regeneration prompt. Checkers must be precise: a false
# anchor sends the regeneration to the wrong place, which is worse than no
# anchor (we fall back to end-of-text feedback in that case).


@dataclass(frozen=True)
class AnchorEvidence:
    checker: str
    step_idx: int
    span: tuple[int, int]          # char span WITHIN the step text
    claim: str                     # the text of the claim that failed
    verdict: str                   # concrete, prompt-ready explanation


_NUM = r"-?\d[\d,]*(?:\.\d+)?"
# `a op b = c`, chains allowed (we check each adjacent pair). Plain arithmetic
# only — latex fractions etc. are handled by the sympy checker below.
_ARITH_RE = re.compile(
    rf"({_NUM})\s*([+\-*/×÷])\s*({_NUM})\s*=\s*({_NUM})"
)
_POW_RE = re.compile(rf"({_NUM})\s*(?:\^|\*\*)\s*(\d+)\s*=\s*({_NUM})")
_MOD_RE = re.compile(rf"({_NUM})\s*(?:mod|%)\s*({_NUM})\s*=\s*({_NUM})", re.IGNORECASE)


def _to_f(s: str) -> float:
    return float(s.replace(",", ""))


# Characters that indicate the matched fragment is part of a LARGER expression
# (a chain, a fraction, an exponent, a subscript, latex braces). Anchoring on a
# fragment of a larger expression produced most of the v1 false positives
# ("x^2 - 1 = 0" -> "2 - 1 = 0"; "4 + 16 + 4 = 24" -> "16 + 4 = 24").
_LCONTEXT_BAD = set("+-*/×÷^_{\\dD")   # d/D guard: digit handled separately
_RCONTEXT_BAD = set("+-*/×÷^_{.\\")


def _standalone(text: str, start: int, end: int) -> bool:
    """True iff text[start:end] is not embedded in a longer math expression."""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i >= 0 and (text[i].isdigit() or text[i] in _LCONTEXT_BAD or text[i].isalpha()):
        # preceded by a digit / operator / exponent / variable -> mid-expression
        # (allow sentence punctuation and brackets that OPEN a standalone claim)
        if text[i] not in ".,;:!?)]$":
            return False
    j = end
    while j < len(text) and text[j] in " \t":
        j += 1
    if j < len(text):
        ch = text[j]
        if ch == ".":  # decimal continuation vs sentence period
            return not (j + 1 < len(text) and text[j + 1].isdigit())
        if ch.isdigit() or ch in _RCONTEXT_BAD:
            return False
    return True


_FLOOR_HINT = re.compile(r"\\l?floor|\\lceil|⌊|⌈|\bfloor\b|\bceil\b|\brounded?\b|\binteger part\b", re.I)
_APPROX_HINT = re.compile(r"\\approx|≈|approximately|about|roughly|~", re.I)


def check_arithmetic(step: Step) -> Optional[AnchorEvidence]:
    """Recompute literal `a op b = c` claims that stand alone (not fragments
    of a longer chain / fraction / exponent) in a step with no floor/approx
    language nearby."""
    for m in _ARITH_RE.finditer(step.text):
        a, op, b, c = m.groups()
        if not _standalone(step.text, m.start(), m.end()):
            continue
        ctx = step.text[max(0, m.start() - 60): m.end() + 20]
        if _FLOOR_HINT.search(ctx) or _APPROX_HINT.search(ctx):
            continue
        try:
            av, bv, cv = _to_f(a), _to_f(b), _to_f(c)
        except ValueError:
            continue
        got = {"+": av + bv, "-": av - bv, "*": av * bv, "×": av * bv,
               "/": av / bv if bv else float("nan"), "÷": av / bv if bv else float("nan")}[op]
        if got != got:  # nan
            continue
        # integer division stated as exact quotient ("7/2 = 3") is usually
        # floor-intent even without the word; skip non-exact divisions whose
        # stated result is the floor.
        if op in "/÷" and cv == int(cv) and int(av) // max(1, int(bv)) == int(cv) and got != cv:
            continue
        ok = abs(got - cv) <= max(1e-9, 1e-6 * max(abs(got), abs(cv)))
        if not ok:
            gs = f"{got:.6f}".rstrip("0").rstrip(".")
            return AnchorEvidence(
                checker="arithmetic",
                step_idx=step.idx, span=m.span(), claim=m.group(0),
                verdict=f"this step asserts {m.group(0)}, but {a} {op} {b} = {gs}",
            )
    for m in _POW_RE.finditer(step.text):
        a, b, c = m.groups()
        if not _standalone(step.text, m.start(), m.end()):
            continue
        try:
            got = _to_f(a) ** int(b); cv = _to_f(c)
        except (ValueError, OverflowError):
            continue
        if abs(got - cv) > max(1e-9, 1e-6 * max(abs(got), abs(cv))):
            gs = f"{got:.6f}".rstrip("0").rstrip(".")
            return AnchorEvidence("power", step.idx, m.span(), m.group(0),
                                  f"this step asserts {m.group(0)}, but {a}^{b} = {gs}")
    for m in _MOD_RE.finditer(step.text):
        a, b, c = m.groups()
        if not _standalone(step.text, m.start(), m.end()):
            continue
        try:
            av, bv, cv = int(_to_f(a)), int(_to_f(b)), int(_to_f(c))
        except ValueError:
            continue
        if bv and (av % bv) != cv:
            return AnchorEvidence("modular", step.idx, m.span(), m.group(0),
                                  f"this step asserts {m.group(0)}, but {av} mod {bv} = {av % bv}")
    return None


_BOUND_PATTERNS = [
    # (regex, verdict template) — quantities with hard ranges. The value group
    # must be the COMPLETE right-hand side: `(?![\d./^*+\-])` after the number
    # rejects fractions ("sin θ = 300/500" — 300 alone is not the value),
    # decimals continuing, products, etc.
    (re.compile(rf"probability(?:\s+is|\s*[:=])\s*({_NUM})(?!\s*[\d./^*+\-])", re.I),
     lambda v: f"a probability of {v} is outside [0, 1]" if not 0 <= v <= 1 else None),
    (re.compile(rf"\bcos(?:ine)?\s*(?:\\theta|θ|\([^)]{{0,12}}\)|[A-Za-z])?\s*=\s*({_NUM})(?![\d./^*+\-°])", re.I),
     lambda v: f"a cosine value of {v} is outside [-1, 1]" if not -1 <= v <= 1 else None),
    (re.compile(rf"\bsin(?:e)?\s*(?:\\theta|θ|\([^)]{{0,12}}\)|[A-Za-z])?\s*=\s*({_NUM})(?![\d./^*+\-°])", re.I),
     lambda v: f"a sine value of {v} is outside [-1, 1]" if not -1 <= v <= 1 else None),
]


def _value_is_math_fragment(text: str, start: int) -> bool:
    """Light left-context check for bound values: the value must not be the
    tail of a larger numeric/latex expression. (Prose like 'probability is'
    on the left is fine — that's the normal case.)"""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i >= 0 and (text[i].isdigit() or text[i] in "./^_{\\-")


def check_bounds(step: Step) -> Optional[AnchorEvidence]:
    for rx, judge in _BOUND_PATTERNS:
        for m in rx.finditer(step.text):
            if _value_is_math_fragment(step.text, m.start(1)):
                continue
            try:
                v = _to_f(m.group(1))
            except ValueError:
                continue
            verdict = judge(v)
            if verdict:
                return AnchorEvidence("bounds", step.idx, m.span(), m.group(0), verdict)
    return None


_SYMPY_EQ_RE = re.compile(r"\$([^$\n]{3,120})\$")


def check_sympy_equations(step: Step) -> Optional[AnchorEvidence]:
    """Verify $...=...$ latex equalities whose sides are both numeric-ish.

    Only fires when BOTH sides parse and evaluate to concrete numbers —
    symbolic identities are skipped (too many false positives).
    """
    try:
        from sympy.parsing.latex import parse_latex  # heavy; lazy import
    except Exception:
        return None
    for m in _SYMPY_EQ_RE.finditer(step.text):
        expr = m.group(1)
        if "=" not in expr or "\\le" in expr or "\\ge" in expr or "<" in expr or ">" in expr:
            continue
        lhs_s, rhs_s = expr.split("=", 1)
        if "=" in rhs_s:            # chains: check first link only
            rhs_s = rhs_s.split("=", 1)[0]
        try:
            lhs = parse_latex(lhs_s).evalf()
            rhs = parse_latex(rhs_s).evalf()
            lv, rv = float(lhs), float(rhs)
        except Exception:
            continue
        if abs(lv - rv) > max(1e-6, 1e-4 * max(abs(lv), abs(rv))):
            return AnchorEvidence(
                "sympy_eq", step.idx, m.span(), m.group(0),
                f"this step asserts ${expr}$, but the left side evaluates to "
                f"{lv:.6g} and the right side to {rv:.6g}",
            )
    return None


CHECKERS: list[Callable[[Step], Optional[AnchorEvidence]]] = [
    check_arithmetic,
    check_bounds,
    check_sympy_equations,
]


# ── Answer-drift anchor (trajectory-level, no gold needed) ───────────────
#
# Mining result: 21% of committed-wrong rollouts literally state the gold
# value somewhere in their reasoning and then commit a different one. These
# "had it, lost it" cases are invisible to per-step arithmetic checkers (the
# per-step math is often fine — the model changes its mind) but visible as
# DRIFT in the sequence of final-answer-shaped claims: \boxed{X}, "the answer
# is X", "m + n = X", ... The anchor is the point where the model abandoned
# the earlier value; the evidence is the (X -> Y) switch itself.
#
# Precision matters because a drift anchor on a CORRECT final answer could
# steer regeneration back to the wrong earlier value (the one failure mode
# that would make broke > 0). So claims are restricted to final-answer-shaped
# statements, not arbitrary intermediate equalities.

_CLAIM_PATTERNS = [
    re.compile(r"\\boxed\s*\{\s*([^{}]{1,40})\s*\}"),
    re.compile(r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|:|=)\s*\$?\\?(?:boxed\{)?\s*([-+]?[\d][\d,]*(?:\.\d+)?(?:/\d+)?)", re.I),
    re.compile(r"(?:therefore|thus|hence|so)\b[^.\n]{0,60}?\b(?:m\s*\+\s*n|a\s*\+\s*b|p\s*\+\s*q|sum|total|value|result)\s*(?:is|=)\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)", re.I),
]


def _norm_claim(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = re.sub(r"^\\text\{([^}]*)\}$", r"\1", s)
    s = s.rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_answer_claims(reasoning: str) -> list[tuple[int, str]]:
    """[(char_pos, normalized_value)] of final-answer-shaped claims, in order."""
    claims: list[tuple[int, str]] = []
    for rx in _CLAIM_PATTERNS:
        for m in rx.finditer(reasoning):
            v = _norm_claim(m.group(1))
            if v and len(v) <= 24:
                claims.append((m.start(), v))
    claims.sort()
    # collapse consecutive duplicates
    out: list[tuple[int, str]] = []
    for pos, v in claims:
        if not out or out[-1][1] != v:
            out.append((pos, v))
    return out


def check_answer_drift(reasoning: str, final_answer: Optional[str] = None) -> Optional[AnchorEvidence]:
    """Anchor at the point where the model switched away from an earlier
    final-answer claim. Returns None when the claim sequence is constant."""
    claims = extract_answer_claims(reasoning)
    if len(claims) < 2:
        return None
    final_v = _norm_claim(final_answer) if final_answer else claims[-1][1]
    # earliest claim that differs from the final committed value, and the
    # position where the model first left it
    prev = None
    for i, (pos, v) in enumerate(claims):
        if v != final_v:
            prev = (i, pos, v)
            break
    if prev is None:
        return None
    i, pos_x, x = prev
    # drift point = first claim after it with a different value
    for pos_y, y in claims[i + 1:]:
        if y != x:
            steps = segment_steps(reasoning)
            step_idx = next((s.idx for s in steps if s.char_start <= pos_y < s.char_end), len(steps) - 1)
            return AnchorEvidence(
                checker="answer_drift", step_idx=step_idx, span=(pos_y, pos_y),
                claim=f"{x} -> {y}",
                verdict=(f"earlier in the solution you concluded the answer is {x}, "
                         f"but from this point on you switch to {y} (final: {final_v}). "
                         f"Re-examine which derivation is right instead of switching silently"),
            )
    return None


def locate_anchor_v2(reasoning: str, final_answer: Optional[str] = None) -> AnchorResult:
    """Drift anchor first (trajectory-level, high precision), then the
    per-step deterministic checkers."""
    ev = check_answer_drift(reasoning, final_answer)
    if ev is not None:
        steps = segment_steps(reasoning)
        trunc = steps[ev.step_idx].char_start if steps else ev.span[0]
        return AnchorResult(True, ev, steps, truncate_at=trunc)
    return locate_anchor(reasoning)


# ── Anchor location ──────────────────────────────────────────────────────


@dataclass
class AnchorResult:
    anchored: bool
    evidence: Optional[AnchorEvidence] = None
    steps: list[Step] = field(default_factory=list)
    # char offset in the ORIGINAL text at which regeneration should start
    # (start of the anchored step). None when not anchored.
    truncate_at: Optional[int] = None


def locate_anchor(reasoning: str,
                  checkers: Optional[list] = None,
                  skip_final_fraction: float = 0.0) -> AnchorResult:
    """Earliest concrete failure across all steps.

    skip_final_fraction: optionally ignore anchors inside the last X of the
    text (an anchor ON the final answer line degenerates to Case-C-style
    end-of-text feedback; usually we still accept it).
    """
    steps = segment_steps(reasoning)
    if not steps:
        return AnchorResult(False)
    limit = len(reasoning) * (1.0 - skip_final_fraction)
    for step in steps:
        if step.char_start > limit:
            break
        for chk in (checkers or CHECKERS):
            ev = chk(step)
            if ev is not None:
                return AnchorResult(True, ev, steps, truncate_at=step.char_start)
    return AnchorResult(False, None, steps)


# ── Anchored regeneration prompt ─────────────────────────────────────────

ANCHOR_NOTE_TMPL = (
    "\n\n[Anchor Check] A verification tool found a concrete error at this "
    "point in your solution: {verdict}.\n"
    "The reasoning before this point has not been flagged. Redo the work "
    "from here, fixing this error, and carry the solution through to a "
    "final answer. End with your final answer in \\boxed{{}}.\n\n"
    "Corrected continuation:\n"
)


def build_anchored_prompt(t1_prompt: str, reasoning: str, res: AnchorResult) -> Optional[str]:
    """prompt = original prompt + reasoning[:anchor] + targeted note.

    Returns None when not anchored (caller falls back to the existing
    end-of-text Case-C feedback, or to no intervention at all).
    """
    if not res.anchored or res.truncate_at is None or res.evidence is None:
        return None
    prefix = reasoning[: res.truncate_at].rstrip()
    return t1_prompt + prefix + ANCHOR_NOTE_TMPL.format(verdict=res.evidence.verdict)


def accept_regeneration(new_answer: Optional[str], original_answer: str) -> str:
    """Verify-and-fallback: keep the original committed answer unless the
    anchored branch commits a parseable answer of its own. This is the rule
    that preserved 0-broke in the current system; do not weaken it."""
    if new_answer is None or not str(new_answer).strip():
        return original_answer
    return str(new_answer).strip()
