"""Hand-written, deterministic step-level PF checkers (from the wrong-case review).

Each checker: (step_text, full_response, step_char_start) -> Optional[dict]
    {"pf": skill_id, "verdict": <prompt-ready evidence>, "fix": <optional
     corrected step text>}
They are the anchor side of the dual-consent gate: precise, zero-LLM, and
anchored to the exact step. Derived from `WRONG_CASES_REVIEW.md`:

compute_observation_verify
    The ReAct template makes the policy WRITE ITS OWN Observation after
    `Action: compute[expr]`. In the mined wrong cases those self-written
    values are frequently wrong (binom(7,2)*binom(7,1) -> "140", really 147;
    binom(4,2)*binom(6,2) -> "216", really 90). Re-evaluate `expr` with sympy
    and compare. A mismatch is a provable error at a known step; the fix is
    to rewrite the Observation with the true value and regenerate from there.
    Fires whenever sympy disagrees with the Observation the model wrote
    rollouts — and the correct-set fires are ALSO genuine miscomputations
    that happened not to matter, so correcting them is safe.

unsupported_final_answer
    The locator-blind errors (106/321) are dominated by answers that were
    never derived: "I will go with 12", "given the time ...", "known result",
    "in similar problems the answer is typically 16". Trigger on those
    phrases in the tail before finish[]; the action is to refuse the guess
    and demand an explicit derivation. Fires more often on wrong than on
    correct rollouts (2.3x enrichment) — this one needs the model's consent
    and the fallback gate, it is not provable like the first.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import sympy
from sympy import (Abs, Rational, binomial, ceiling, cos, exp, factorial, floor, gcd, lcm, log,
                   pi, sin, sqrt, tan)

from hasp_config import protocol as _protocol

_P = _protocol()

# ── compute_observation_verify ───────────────────────────────────────────

_CMP = re.compile(r"Action:\s*compute\[(.+?)\]\s*\n\s*Observation:\s*([^\n]+)", re.I | re.S)
_NUMS = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_TRIG = re.compile(r"\b(sin|cos|tan)\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)")
_LOCALS = dict(binomial=binomial, factorial=factorial, sqrt=sqrt, gcd=gcd, lcm=lcm, floor=floor,
               ceiling=ceiling, Abs=Abs, pi=pi, sin=sin, cos=cos, tan=tan, log=log, exp=exp)
_FUNC_WORDS = re.compile(r"binomial|factorial|sqrt|gcd|lcm|floor|ceiling|Abs|pi|sin|cos|tan|log|exp")


def _to_sympy_expr(expr: str):
    """Best-effort translation of the compute[] argument into sympy. Returns
    None when the expression is symbolic / truncated / unparseable — we only
    act on claims we can actually verify."""
    if "..." in expr or "…" in expr or len(expr) > 300:
        return None
    e = expr.replace("^", "**").replace("×", "*").replace("·", "*").replace("\\times", "*").replace("\\cdot", "*")
    e = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", e)
    e = re.sub(r"\\binom\{([^}]*)\}\{([^}]*)\}", r"binomial(\1,\2)", e)
    e = re.sub(r"\bC\((\d+),\s*(\d+)\)", r"binomial(\1,\2)", e)
    e = re.sub(r"\bbinom\(", "binomial(", e)
    e = re.sub(r"(\d+)!", r"factorial(\1)", e)
    e = e.replace("\\sqrt", "sqrt").replace("$", "").replace("\\", "")
    # trig with integer-looking arguments beyond 2*pi are degrees in contest math
    def _deg(m):
        v = float(m.group(2))
        return f"{m.group(1)}(({m.group(2)})*pi/180)" if abs(v) > 6.3 else m.group(0)
    e = _TRIG.sub(_deg, e)
    if re.search(r"[a-zA-Z]", _FUNC_WORDS.sub("", e)):
        return None  # free variables -> symbolic, skip
    try:
        v = sympy.sympify(e, locals=_LOCALS)
        return float(v.evalf())
    except Exception:
        return None


def _observation_value(obs: str) -> Optional[float]:
    if re.search(r"execute_result|error|undefined|approx|≈|\babout\b", obs, re.I):
        return None
    # The written value must BE a number, not merely contain one: grabbing
    # "4" out of "4*sqrt(13)" produced false verdicts and wrong step rewrites
    # on correct rollouts. Take the tail after the last "=", strip wrappers,
    # and require the whole token to parse.
    tail = obs.split("=")[-1] if "=" in obs else obs
    tail = tail.replace(",", "").strip().strip("$ .;:")
    tail = re.sub(r"^\\boxed\{(.*)\}$", r"\1", tail)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", tail):
        return None
    s = tail.replace(" ", "")
    try:
        return float(Rational(s)) if "/" in s else float(s)
    except Exception:
        return None


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6g}"


def check_compute_observation(step_text: str, *_args) -> Optional[dict]:
    for m in _CMP.finditer(step_text):
        expr, obs = m.group(1).strip(), m.group(2).strip()
        truth = _to_sympy_expr(expr)
        written = _observation_value(obs)
        if truth is None or written is None or not math.isfinite(truth):
            continue
        if abs(truth - written) <= max(_P.checkers.compute_tolerance_abs,
                                       _P.checkers.compute_tolerance_rel * max(abs(truth), abs(written))):
            continue
        fixed = step_text[: m.start(2)] + _fmt(truth) + step_text[m.end(2):]
        return dict(
            pf="compute_observation_verify",
            verdict=(f"the Observation written for `compute[{expr}]` is {obs[:40]!r}, but the "
                     f"expression actually equals {_fmt(truth)}"),
            fix=fixed,
            truth=_fmt(truth), written=_fmt(written),
            span=(m.start(), m.end()),
        )
    return None


# ── unsupported_final_answer ─────────────────────────────────────────────

_GUESS = re.compile(
    r"\b(I will guess|I'?ll guess|my (?:best )?guess|guess that|I will go with|I'?ll go with|"
    r"go with (?:the|my|that)|given the time|time constraint|"
    r"I (?:need|have) to (?:stop|decide|make a decision)|make a decision|"
    r"known (?:problem|result|answer)|similar problems|typically (?:the answer|is)|"
    r"is often (?:the answer|equal)|I think the answer is|upon reflection)\b", re.I)
_FINISH = re.compile(r"Action:\s*finish\s*\[", re.I)


def check_unsupported_final(step_text: str, full_response: str = "", step_start: int = 0) -> Optional[dict]:
    """Fires on the step that commits (or leads into) finish[] when the
    surrounding text admits the answer was guessed / taken from a 'known
    result' rather than derived."""
    if not _FINISH.search(step_text):
        return None
    window = full_response[max(0, step_start - 1200): step_start] + step_text if full_response else step_text
    m = _GUESS.search(window)
    if not m:
        return None
    return dict(
        pf="unsupported_final_answer",
        verdict=(f"the final answer is committed right after the phrase {m.group(0)!r} — it was not "
                 f"derived from the work above. Do not guess: carry out the computation explicitly "
                 f"and commit only what the derivation supports"),
        fix=None,
        span=(m.start(), m.end()),
    )


# ── counting_small_case_check ────────────────────────────────────────────
#
# Counting errors are structural — a wrong closed form
# such as C(13-k, k) instead of C(12-k, k-1), or "any two parallel chords
# pair into a rectangle". No regex can see that, but almost every such claim
# is parametric, so it can be FALSIFIED by brute-force enumeration on a small
# instance. The anchor side (this trigger) only says "this step states a
# parametric count"; the evidence is produced by the judge writing a tiny
# enumeration (see step_gate.CODE_TMPL) and the sandbox running it.

_PARAM_COUNT = re.compile(
    r"(\\binom\{[^}]*[a-zA-Z][^}]*\}\{[^}]*\}|\\binom\{[^}]*\}\{[^}]*[a-zA-Z][^}]*\}|"
    r"\bC\([^)]*[a-zA-Z][^)]*\)|\b[a-zA-Z]\s*!|\b\d+\s*\^\s*\{?[a-zA-Z]|\b[a-zA-Z]\s*\^\s*\{?\d)"
)
_COUNT_WORDS = re.compile(r"\b(number of|count|ways|subsets|arrangements|sequences|paths|configurations|colou?rings|tuples|pairs)\b", re.I)


_STATED_COUNT = re.compile(
    r"((?:C\(\s*\d+\s*,\s*\d+\s*\)|\\binom\{\d+\}\{\d+\}|binom(?:ial)?\(\s*\d+\s*,\s*\d+\s*\))"
    r"(?:\s*[*x\u00d7\u00b7]\s*(?:C\(\s*\d+\s*,\s*\d+\s*\)|\\binom\{\d+\}\{\d+\}|binom(?:ial)?\(\s*\d+\s*,\s*\d+\s*\)))*)"
    r"\s*=\s*(-?[\d,]+)\b")


def check_stated_count(step_text: str, *_args):
    """A concrete combinatorial claim `C(n,k)[*C(m,j)...] = V`, recomputed.

    This is the executable half `check_counting_trigger` only ever promised:
    the trigger returned `verdict=None, needs_enumeration=True` for an
    enumeration stage that never made it into this library, so the skill
    detected its targets and then had nothing to say. Fully-numeric claims
    need no enumeration harness -- sympy evaluates them directly.
    """
    for m in _STATED_COUNT.finditer(step_text):
        expr, stated = m.group(1), m.group(2).replace(",", "")
        truth = _to_sympy_expr(expr)
        if truth is None or not math.isfinite(truth):
            continue
        try:
            if abs(truth - float(stated)) < 0.5:
                continue
        except ValueError:
            continue
        return dict(pf="counting_small_case_check",
                    verdict=(f"the reasoning states `{expr} = {stated}`, but "
                             f"{expr} actually equals {_fmt(truth)}"),
                    fix=_fmt(truth), span=m.span())
    return None


def check_counting_trigger(step_text: str, *_args) -> Optional[dict]:
    """Anchor-side trigger only (no deterministic verdict): a parametric
    closed-form count stated in this step. Returns a dict with verdict=None so
    the evidence stage knows to run the enumeration path."""
    if _PARAM_COUNT.search(step_text) and _COUNT_WORDS.search(step_text):
        return dict(pf="counting_small_case_check", verdict=None, fix=None, needs_enumeration=True, enum_kind="count")
    return None


# ── interval_sign_check ──────────────────────────────────────────────────
#
# algebra family: sign analysis by interval ("x < 0: -x > 0, x-2 < 0 → product
# positive → overall negative") gets a sign wrong (amc23_37: 901 -> 900/902).
# Every such line is a conjunction of atomic claims `expr ⋚ 0` under an
# interval condition, so pick a test point in the interval and evaluate each
# atomic claim with sympy. Provable, zero LLM.

from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                        standard_transformations)
_TR = standard_transformations + (implicit_multiplication_application,)
_NUMF = r"-?(?:\\frac\{-?\d+\}\{\d+\}|\d+(?:\.\d+)?(?:/\d+)?|\\?infty)"
_COND = re.compile(rf"\$?\s*({_NUMF})?\s*(<|\\le|≤|<=)?\s*([a-zA-Z])\s*(<|\\le|≤|<=|>|\\ge|≥|>=)\s*({_NUMF})\s*\$?")
_ATOM = re.compile(r"\$\s*([^$<>=]{1,60}?)\s*(<|>|\\le|\\ge|≤|≥|<=|>=)\s*0\s*\$")


def _parse(expr: str, var: str):
    e = expr.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("^", "**")
    e = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", e).replace("\\cdot", "*").replace("\\times", "*")
    e = e.replace("{", "(").replace("}", ")").replace("\\", "")
    if re.search(r"[a-zA-Z]", e.replace(var, "")):
        return None  # other symbols -> skip
    try:
        return parse_expr(e, transformations=_TR, local_dict={var: sympy.Symbol(var)})
    except Exception:
        return None


def _test_point(m) -> Optional[float]:
    lo, _lo_op, var, op, hi = m.groups()
    def num(s):
        if s is None: return None
        if "infty" in s: return -1e9 if s.startswith("-") else 1e9
        fm = re.match(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
        if fm: return (-1 if fm.group(1) else 1) * float(fm.group(2)) / float(fm.group(3))
        return float(Rational(s)) if "/" in s else float(s)
    lo_v, hi_v = num(lo), num(hi)
    if op in (">", "\\ge", "≥", ">="):           # x > hi_v  (single-sided)
        lo_v, hi_v = hi_v, 1e9
    if lo_v is None: lo_v = -1e9
    if hi_v is None: hi_v = 1e9
    if lo_v >= hi_v: return None
    if lo_v <= -1e8: return hi_v - 1.0
    if hi_v >= 1e8: return lo_v + 1.0
    return (lo_v + hi_v) / 2.0


def check_interval_sign(step_text: str, *_args) -> Optional[dict]:
    for line in step_text.splitlines():
        conds = list(_COND.finditer(line))
        if not conds:
            continue
        cm = conds[0]
        var = cm.group(3); t = _test_point(cm)
        if t is None:
            continue
        seg_end = conds[1].start() if len(conds) > 1 else len(line)
        for am in _ATOM.finditer(line[cm.end():seg_end]):
            expr, op = am.group(1), am.group(2)
            e = _parse(expr, var)
            if e is None:
                continue
            try:
                v = float(e.subs(sympy.Symbol(var), t).evalf())
            except Exception:
                continue
            if not math.isfinite(v) or abs(v) < 1e-9:
                continue
            claimed_pos = op in (">", "\\ge", "≥", ">=")
            if (v > 0) != claimed_pos:
                return dict(
                    pf="interval_sign_check",
                    verdict=(f"in the interval {cm.group(0).strip('$ ')} the step claims ${expr.strip()} {op} 0$, "
                             f"but at the test point {var}={t:g} the expression equals {v:.4g}, i.e. it is "
                             f"{'positive' if v > 0 else 'negative'}"),
                    fix=None, span=(am.start(), am.end()))
    return None


# ── claimed_unique_solution_search (anchor-side trigger; evidence by search) ─
_UNIQUE = re.compile(r"\b(the only solutions?|only (?:possible )?solution|only when\b|no other solutions?|"
                     r"must be the only|the unique solution|are the only|is the only)\b", re.I)


def check_unique_solution_trigger(step_text: str, *_args) -> Optional[dict]:
    if _UNIQUE.search(step_text):
        return dict(pf="claimed_unique_solution_search", verdict=None, fix=None, needs_enumeration=True, enum_kind="solutions")
    return None


# ── runaway_enumeration_breaker ──────────────────────────────────────────
# The stall channel's signature: 20k–115k-char rollouts that are 88–97%
# repeated lines ("Try t=277: ... ≠ 0. Try t=278: ..."). Fires on the step
# where repetition density first crosses the threshold — that is the anchor —
# with the repeated template quoted. Action: stop enumerating, reason
# structurally, then finish.
def check_runaway_enumeration(step_text: str, full_response: str = "", step_start: int = 0,
                              min_lines: int = 30, ratio: float = 0.6) -> Optional[dict]:
    # 30, not 120: the Detect gate is >30 lines and the trigger says "a step
    # where most lines repeat" -- a 36-line window that is 88% one template is
    # already a runaway, and the old 4x-higher bar made the skill structurally
    # silent on every rollout its own Detect admitted.
    prefix = full_response[: step_start + len(step_text)] if full_response else step_text
    lines = [l.strip() for l in prefix.splitlines() if l.strip()]
    if len(lines) < min_lines:
        return None
    seen = {}; dup = 0
    for l in lines:
        key = re.sub(r"\d+", "#", l)[:80]          # numbers masked: "Try t=#: ... ≠ #"
        if key in seen: dup += 1
        seen[key] = seen.get(key, 0) + 1
    r = dup / len(lines)
    if r < ratio:
        return None
    tmpl, n = max(seen.items(), key=lambda kv: kv[1])
    return dict(pf="runaway_enumeration_breaker",
                verdict=(f"{dup} of the {len(lines)} lines so far repeat an earlier line pattern (most common: "
                         f"{tmpl[:60]!r} x{n}); this enumeration is not converging. Stop enumerating, find the "
                         f"structural argument (bound, parity, factorisation), and commit an answer"),
                fix=None, span=(0, len(step_text)))


# ── equation_substitution_check ──────────────────────────────────────────
# verification_missing made concrete: if the PROBLEM states an equation in
# exactly one unknown and the final answer is that unknown's value, substitute
# and check. Provable; fires rarely but exactly.
_EQ = re.compile(r"([^=\n$]{2,60}?)\s*=\s*([^=\n$]{1,60})")
_SYM = re.compile(r"(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z])")


def check_equation_substitution(question: str, answer: str) -> Optional[dict]:
    ans = str(answer).strip().strip("$")
    try:
        val = sympy.Rational(ans) if re.fullmatch(r"-?\d+(/\d+)?", ans) else sympy.nsimplify(float(ans))
    except Exception:
        return None
    for m in _EQ.finditer(question.replace("\\left", "").replace("\\right", "")):
        lhs, rhs = m.group(1), m.group(2)
        syms = set(_SYM.findall(lhs + rhs)) - {"e"}
        if len(syms) != 1:
            continue
        v = syms.pop()
        if not re.search(rf"(find|value of|solve for|what is)\s+\$?\\?{v}\b", question, re.I):
            continue
        try:
            L = sympy.parse_expr(re.sub(r"\^", "**", lhs).replace("{", "(").replace("}", ")").replace("\\", ""),
                                 transformations=_TR, local_dict={v: sympy.Symbol(v)})
            R = sympy.parse_expr(re.sub(r"\^", "**", rhs).replace("{", "(").replace("}", ")").replace("\\", ""),
                                 transformations=_TR, local_dict={v: sympy.Symbol(v)})
            d = float((L - R).subs(sympy.Symbol(v), val).evalf())
        except Exception:
            continue
        if abs(d) > 1e-6:
            # the equation is single-variable and machine-parsed, so solving it
            # is one more sympy call -- and a solution that substitutes back
            # clean is its own proof
            solved = None
            try:
                for cand in sympy.solve(L - R, sympy.Symbol(v)):
                    if not cand.is_real:
                        continue
                    if abs(float((L - R).subs(sympy.Symbol(v), cand).evalf())) < 1e-9:
                        solved = (str(cand) if cand == sympy.nsimplify(cand)
                                  else f"{float(cand.evalf()):.6g}")
                        break
            except Exception:
                solved = None
            return dict(pf="equation_substitution_check",
                        verdict=(f"substituting the final answer {v} = {ans} into the problem's equation "
                                 f"{lhs.strip()} = {rhs.strip()} gives a residual of {d:.4g}, so it does not satisfy the equation"),
                        fix=None, solved=solved, span=m.span())
    return None
