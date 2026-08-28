"""Program Functions for the MATH domain (AIME24 / AIME25 / MATH-500).

Each skill's `action: verify_X` in SKILL.md corresponds to a PF class below.
Unlike web_search PFs (mix of cheap regex + teacher-assisted), MATH PFs are
nearly all teacher-assisted: a single LLM call per FINAL to check for the
category of error.

Activation policy (shared across all math PFs):
  - Fire ONLY on action_type == "FINAL"
  - Fire ONLY once per episode (gated by `_pf_fire_counts` in step_context)
  - PF helper is gpt-4o by default; if unavailable, PF degrades to NOOP
  - Intervention type: INJECT_CONTEXT with "[SKILL_ID] PF helper-flagged issue: ..."
    so the agent can revise on retry (handled upstream by the agent loop)

When PF helper agrees the answer is fine, PF returns NOOP. When PF helper
disagrees, PF injects a concise warning + retry hint.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from .program_functions import (
    Intervention,
    InterventionType,
    ProgramFunction,
    register_pf,
)
from .quota import note_api_error

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

_MAX_FIRES_PER_PF = 1    # math PFs rarely need to fire more than once


def _already_fired(step_context: Dict[str, Any], skill_id: str) -> bool:
    counts = step_context.get("_pf_fire_counts", {})
    return counts.get(skill_id, 0) >= _MAX_FIRES_PER_PF


def _render_problem(step_context: Dict[str, Any]) -> str:
    q = step_context.get("question") or step_context.get("query") or ""
    return str(q).strip()


def _render_trajectory_snippet(step_context: Dict[str, Any], max_chars: int = 2000) -> str:
    """Concatenate the model's reasoning for this episode (thought + prior actions)."""
    bits: list = []
    thought = step_context.get("thought") or ""
    if thought:
        bits.append(str(thought))
    # action_history: list of prior (action_type, arg) tuples, if tracked
    hist = step_context.get("action_history") or []
    for h in hist[-5:]:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            bits.append(f"{h[0]}({str(h[1])[:120]})")
    text = "\n".join(bits)
    return text[:max_chars]


def _teacher_verify(
    teacher, prompt: str, skill_id: str, max_tokens: int = 300,
) -> Optional[str]:
    """Single-shot PF helper call with quota-aware error handling.
    Returns PF helper's response text, or None on failure."""
    if teacher is None:
        return None
    try:
        # PF helper.generate_from_messages is the standard interface used
        # by skill_handlers; fall back to PF helper.generate for simpler wrappers.
        if hasattr(teacher, "generate_from_messages"):
            return teacher.generate_from_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.0,
            )
        return teacher.generate(prompt=prompt, temperature=0.0, max_tokens=max_tokens)
    except Exception as e:
        if note_api_error(e):
            raise  # quota / auth — abort
        logger.warning("[math-PF:%s] PF helper call failed: %s", skill_id, e)
        return None


# PF helper response protocol:
#   First line of response must be one of: OK / ISSUE
#   If ISSUE: remainder is a 1-2 sentence explanation for the agent
_OK_RE = re.compile(r"^\s*(?:OK|CORRECT|LOOKS\s*OK|FINE)\b", re.IGNORECASE)
_ISSUE_RE = re.compile(r"^\s*(?:ISSUE|ERROR|WRONG|PROBLEM)\b", re.IGNORECASE)


def _parse_teacher_verdict(resp: str):
    """Return (verdict, explanation). verdict ∈ {"ok", "issue", "unknown"}."""
    if not resp:
        return "unknown", ""
    first_line = resp.strip().splitlines()[0] if resp.strip() else ""
    rest = resp.strip()[len(first_line):].strip()
    if _OK_RE.match(first_line):
        return "ok", ""
    if _ISSUE_RE.match(first_line):
        return "issue", rest[:300] or first_line
    # Fallback: look for "no issue" vs "issue" anywhere
    if re.search(r"\b(no error|no issue|correct|looks good|seems right)\b", resp, re.I):
        return "ok", ""
    if re.search(r"\b(error|mistake|wrong|incorrect|inconsistent)\b", resp, re.I):
        return "issue", resp[:300]
    return "unknown", resp[:200]


# ============================================================================
# Shared PF base — all math PFs follow the same pre-final PF helper-verify flow
# ============================================================================

class _MathVerifyPF(ProgramFunction):
    """Generic pre-FINAL math verification PF.

    Subclasses set:
      - error_hint (str): skill-specific description used in the PF helper prompt
      - priority (float): unused here (handled by selector)
      - trigger_check (callable, optional): cheap heuristic gate BEFORE the
        PF helper call, to save API cost. If returns False, PF NOOPs silently.
    """
    needs_helper = True
    error_hint: str = ""             # override per-skill
    retry_hint: str = ""             # injected to agent on verdict=issue
    require_final: bool = True       # most PFs only fire on FINAL

    def should_activate(self, step_context, action_type, arg) -> bool:
        if self.require_final and (action_type or "").upper() != "FINAL":
            return False
        if _already_fired(step_context, self.skill_id):
            return False
        # Skill-specific trigger (cheap regex / heuristic). Default: always.
        return self.trigger(step_context, action_type, arg)

    def trigger(self, step_context, action_type, arg) -> bool:
        return True

    def intervene(self, step_context, action_type, arg, helper=None) -> Intervention:
        # Increment fire count
        counts = step_context.setdefault("_pf_fire_counts", {})
        counts[self.skill_id] = counts.get(self.skill_id, 0) + 1

        if helper is None:
            return Intervention(type=InterventionType.NOOP, skill_id=self.skill_id,
                                reason="helper unavailable")

        prompt = self._build_prompt(step_context, arg)
        resp = _teacher_verify(helper, prompt, self.skill_id, max_tokens=200)
        verdict, explanation = _parse_teacher_verdict(resp or "")

        if verdict == "ok":
            return Intervention(type=InterventionType.NOOP, skill_id=self.skill_id,
                                reason="helper: ok")

        if verdict == "issue":
            msg = explanation or self.retry_hint or self.error_hint
            return Intervention(
                type=InterventionType.INJECT_CONTEXT,
                context_text=f"[{self.skill_id}] Before finalizing: {msg}",
                reason=f"helper flagged {self.skill_id}",
                skill_id=self.skill_id,
            )
        # unknown / no response → do not block
        return Intervention(type=InterventionType.NOOP, skill_id=self.skill_id,
                            reason="helper: unknown")

    def _build_prompt(self, step_context, answer) -> str:
        problem = _render_problem(step_context)
        trace = _render_trajectory_snippet(step_context)
        return (
            f"You are a math proof checker. Given a problem, the student's "
            f"reasoning, and their final answer, check for ONE specific issue:\n"
            f"  {self.error_hint}\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"STUDENT REASONING (truncated):\n{trace}\n\n"
            f"STUDENT FINAL ANSWER: {str(answer)[:200]}\n\n"
            f"Reply in EXACTLY this format on the first line:\n"
            f"  OK   — if no such issue is detected\n"
            f"  ISSUE: <1-sentence explanation> — if the issue is present\n"
            f"Be strict but only flag the specific issue above. Do not "
            f"nit-pick other problems."
        )


# ============================================================================
# The 11 skill PFs
# ============================================================================

@register_pf("arithmetic_slip")
class ArithmeticSlipPF(_MathVerifyPF):
    error_hint = ("Check whether the final arithmetic chain (sums, products, "
                  "fraction reductions, modular results) is numerically correct. "
                  "Do NOT flag for style or method — only if a number is actually wrong.")

    def trigger(self, step_context, action_type, arg) -> bool:
        # Cheap gate: only fire if answer contains numbers or the reasoning
        # does ≥3 numeric ops (has ±, *, /, =)
        txt = (step_context.get("thought") or "") + str(arg)
        n_ops = len(re.findall(r"[+\-*/=]", txt))
        return n_ops >= 3


@register_pf("algebraic_sign_error")
class AlgebraicSignErrorPF(_MathVerifyPF):
    error_hint = ("Check for algebraic sign errors: distributing a negative, "
                  "flipping inequality signs when multiplying by a negative, "
                  "missing the ± branch of √, sign slip in polynomial subtraction.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        # Fire if reasoning mentions negatives / inequalities / roots
        return bool(re.search(r"[-−]|sqrt|√|<|>|\\leq|\\geq|\\neq", txt))


@register_pf("case_incompleteness")
class CaseIncompletenessPF(_MathVerifyPF):
    error_hint = ("Check whether the case analysis covers ALL cases of the "
                  "problem (exhaustive + mutually exclusive). Look for missing "
                  "boundary cases, missed cases in |x| splits, even/odd, etc.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        return bool(re.search(r"case\b|\|x\||absolute\s+value|WLOG|without\s+loss", txt, re.I))


@register_pf("boundary_violation")
class BoundaryViolationPF(_MathVerifyPF):
    error_hint = ("Check whether the final answer respects all domain constraints "
                  "from the problem: positive integer? integer 0..999 (AIME)? "
                  "log/sqrt domain positive? denominator nonzero?")

    def trigger(self, step_context, action_type, arg) -> bool:
        ans = str(arg).strip()
        # Fire for AIME-like problems (look for "integer" in problem) OR if
        # answer looks non-integer / negative
        q = step_context.get("question") or ""
        is_aime = bool(re.search(r"positive integer|find the value|find m\+n|\\boxed\{[^}]*\}",
                                   q + ans, re.I))
        if is_aime:
            return True
        return bool(re.search(r"\d+\.\d+|-\d+", ans))


@register_pf("substitution_invalid")
class SubstitutionInvalidPF(_MathVerifyPF):
    error_hint = ("Check whether any variable substitution (u-sub, trig-sub, "
                  "y = x², etc.) was applied correctly: domain preserved, "
                  "branches handled, answer mapped back to the original variable.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        return bool(re.search(r"let\s+\w\s*=|substitut|change\s+of\s+variable|u\s*=", txt, re.I))


@register_pf("simplification_incomplete")
class SimplificationIncompletePF(_MathVerifyPF):
    error_hint = ("Check whether the final answer is fully simplified: "
                  "fraction in lowest terms, radical simplified (2√3 not √12), "
                  "integer computed out (not 735/210), unreduced composite gone.")

    def trigger(self, step_context, action_type, arg) -> bool:
        ans = str(arg).strip()
        # Fire if answer still contains reducible-looking forms
        if re.search(r"\\sqrt\{[^}]*\d", ans):    # radical might be reducible
            return True
        if re.search(r"\\frac\{(\d+)\}\{(\d+)\}", ans):    # fraction — maybe not lowest
            return True
        return False


@register_pf("overgeneralization")
class OvergeneralizationPF(_MathVerifyPF):
    error_hint = ("Check whether any named formula/theorem used in the solution "
                  "has its preconditions met: AM-GM needs non-negatives; "
                  "geometric series needs |r|<1; L'Hôpital needs 0/0 or ∞/∞; "
                  "Pythagoras needs right angle; etc.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        named = [
            "AM-GM", "Cauchy-Schwarz", "Jensen", "Pythagoras", "Pythagorean",
            "L'Hopital", "L'Hôpital", "binomial theorem", "geometric series",
            "arithmetic series", "telescoping", "mean value",
        ]
        return any(k.lower() in txt.lower() for k in named)


@register_pf("units_dimension_mismatch")
class UnitsDimensionMismatchPF(_MathVerifyPF):
    error_hint = ("Check for unit / dimension inconsistencies: probability ∈ [0,1], "
                  "integer counts, area/volume not mixed with length, right units.")

    def trigger(self, step_context, action_type, arg) -> bool:
        q = step_context.get("question") or ""
        ans = str(arg)
        # Fire for probability / counting / geometry questions
        if re.search(r"probability|probab|expected|count|area|volume|perimeter", q, re.I):
            return True
        # Sanity check: probability-like answer but > 1 or < 0
        m = re.search(r"(-?\d+\.?\d*)", ans)
        if m:
            try:
                v = float(m.group(1))
                if v > 1.1 or v < -0.1:
                    if "probab" in q.lower():
                        return True
            except ValueError:
                pass
        return False


@register_pf("final_format_error")
class FinalFormatErrorPF(_MathVerifyPF):
    """This one is CHEAP — no PF helper call needed for format check."""
    needs_helper = False

    def should_activate(self, step_context, action_type, arg) -> bool:
        if (action_type or "").upper() != "FINAL":
            return False
        if _already_fired(step_context, self.skill_id):
            return False
        return True

    def intervene(self, step_context, action_type, arg, helper=None) -> Intervention:
        counts = step_context.setdefault("_pf_fire_counts", {})
        counts[self.skill_id] = counts.get(self.skill_id, 0) + 1

        ans = str(arg).strip()
        q = step_context.get("question") or ""
        # Detect AIME from problem text
        is_aime = bool(re.search(r"\bAIME\b", q) or re.search(r"integer 0.*999|find m\s*\+\s*n|find p\s*\+\s*q", q, re.I))

        issues = []
        if is_aime:
            # AIME answers must be integer 0..999 with no \boxed, no units
            m = re.match(r"^-?\d+$", ans)
            if not m:
                issues.append("AIME answer must be a bare integer 0..999, no \\boxed{} / no units.")
            else:
                v = int(ans)
                if v < 0 or v > 999:
                    issues.append(f"AIME answer {v} out of range [0, 999].")
        else:
            # MATH-500 answers should be \boxed{...}
            if "\\boxed{" not in ans:
                issues.append("MATH-500 answer should be wrapped in \\boxed{...}.")
            if re.search(r"(units?|cm|m\b|kg|sec|degrees?|meters?|seconds?)", ans, re.I):
                issues.append("Do not include units inside \\boxed{...}.")

        if not issues:
            return Intervention(type=InterventionType.NOOP, skill_id=self.skill_id,
                                reason="format ok")
        msg = "; ".join(issues)
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=f"[final_format_error] Fix format before finalizing: {msg}",
            reason=f"format issues: {msg[:100]}",
            skill_id=self.skill_id,
        )


@register_pf("proof_step_gap")
class ProofStepGapPF(_MathVerifyPF):
    error_hint = ("Check whether the reasoning skips essential algebraic steps "
                  "('clearly', 'obviously', or multi-step leaps) that may hide "
                  "errors. Flag only if skipping caused a demonstrable gap.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        return bool(re.search(r"\b(clearly|obviously|it follows|therefore|hence)\b", txt, re.I))


@register_pf("verification_missing")
class VerificationMissingPF(_MathVerifyPF):
    error_hint = ("Check whether the student PLUGGED THEIR ANSWER BACK INTO the "
                  "original equation / problem constraints. If they didn't, point out "
                  "which constraint they didn't check.")

    def trigger(self, step_context, action_type, arg) -> bool:
        txt = step_context.get("thought") or ""
        # Fire unless student explicitly mentions verification
        mentioned = re.search(r"\b(check|verify|substitute|plug.*back|plug.*in)\b", txt, re.I)
        return not mentioned
