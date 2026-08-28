"""math — helper functions the skills share.

Moved verbatim from the old `programs.py`; these are the parsing and
normalisation utilities that the answer-shape skills need. Moving rather
than retyping is deliberate — see tests/pf_parity.py.
"""
from __future__ import annotations

import re
from fractions import Fraction
from math import gcd
from typing import Any, Dict, Optional

_MAX_FIRES_PER_PF = 1

def _question(step_context: Dict[str, Any]) -> str:
    return str(step_context.get("question") or step_context.get("query") or "")
_BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_PREFIX_RE = re.compile(r"(?:final\s*answer|the\s*answer\s*is|answer)\s*[:=]?\s*\$?\\?(?:boxed\{)?([^\n$]+)",
                               re.IGNORECASE)

def _explicit_answer(text: str) -> Optional[str]:
    """Extraction restricted to EXPLICIT markers (\\boxed / 'answer is').

    Unlike `extract_math_answer`, this does NOT fall back to 'last numeric
    token' — that fallback mangles expression answers (e.g. GameOf24's
    `13*(5-2)` -> `-2`). Use this for rewrites that must never corrupt a
    valid expression.
    """
    if not text:
        return None
    s = str(text)
    boxes = _BOX_RE.findall(s)
    if boxes:
        return boxes[-1].strip()
    m = list(_ANSWER_PREFIX_RE.finditer(s))
    if m:
        cand = m[-1].group(1).strip().rstrip("}").strip()
        # the marker introduces a VALUE, not a clause: cut at the first
        # clause boundary ("55, which completes the problem" -> "55")
        cand = re.split(r"[,;]|\s+(?:which|so|and|because|hence|therefore|thus)\b",
                        cand, 1)[0]
        cand = re.sub(r"[\s.,;:]+$", "", cand)
        return cand or None
    return None

def _is_expression(s: str) -> bool:
    """True if the answer looks like an arithmetic expression (operators between
    operands / parenthesised), which must be preserved verbatim — NOT wrapped
    or reduced to a single number."""
    s = str(s).strip()
    boxes = _BOX_RE.findall(s)
    core = boxes[-1] if boxes else s
    has_paren_op = bool(re.search(r"[()]", core)) and bool(re.search(r"[+\-*/]", core))
    op_count = len(re.findall(r"(?<=[\d)])\s*[+\-*/]\s*(?=[\d(])", core))
    return has_paren_op or op_count >= 2

def _is_gameof24(question: str) -> bool:
    q = (question or "").lower()
    return ("evaluates to 24" in q or "make 24" in q
            or ("each exactly once" in q and "expression" in q)
            or ("using the numbers" in q and "24" in q))

def _is_verbose_answer(arg: str) -> bool:
    a = (arg or "").strip()
    if len(a) > 40:
        return True
    # contains prose / explanation rather than a bare value
    return bool(re.search(r"\b(the|is|answer|therefore|so|thus|because|since)\b", a, re.I))

def _reduce_fraction(arg: str) -> Optional[str]:
    """Reduce a \\frac{a}{b} or a/b to lowest terms; return new string or None."""
    m = re.search(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", arg) or re.search(r"\b(-?\d+)\s*/\s*(-?\d+)\b", arg)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if b == 0:
        return None
    g = gcd(abs(a), abs(b))
    if g <= 1:
        return None
    na, nb = a // g, b // g
    if "\\frac" in m.group(0):
        repl = f"\\frac{{{na}}}{{{nb}}}"
    else:
        repl = f"{na}/{nb}"
    return arg[: m.start()] + repl + arg[m.end():]
_OK_RE = re.compile(r"^\s*(?:OK|CORRECT|LOOKS\s*OK|FINE)\b", re.IGNORECASE)
_ISSUE_RE = re.compile(r"^\s*(?:ISSUE|ERROR|WRONG|PROBLEM)\b", re.IGNORECASE)
