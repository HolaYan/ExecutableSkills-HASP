"""math PF skills — every skill writes out both modules.

    Detect   should_activate(state)  — does this state and proposed action
                                       match the failure pattern?
    Repair   intervene(state)        — what should happen instead, as a typed
                                       intervention: redirect, inject, or abstain.

Nothing is hidden behind a builder: the Repair body shows, in order, the search
for the strongest verdict the skill can give — recompute, then the helper, then
the continuation for a stalled rollout, then the family reminder, then abstain.

Three things restored here that the previous rewrite had dropped:

  * **the per-family Detects.** Each family carries its own cheap gate, so a
    skill that does not match this rollout never activates — without them every
    selected skill is recorded as activated on every rollout.
  * **the family reminders.** On a stalled rollout any injected feedback
    produces a second turn that finishes the answer, so firing at all matters
    there even when the text is generic.
  * **the anchor as data**, so injected text carries where it attached.

A recomputed verdict takes precedence over a reminder: it names the wrong value
and the right one, which is what a model can act on.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[3]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills.pf_template import (  # noqa: E402
    Anchor, Ctx, abstain, answer_finding, continuation, first_finding,
    helper_verdict, inject, pf_skill, redirect, reminder, stalled, verdict,
)
from anchor.pf_checkers import (  # noqa: E402
    check_stated_count,
    check_compute_observation, check_counting_trigger, check_equation_substitution,
    check_interval_sign, check_runaway_enumeration, check_unique_solution_trigger,
    check_unsupported_final,
)
from anchor.anchor import check_bounds  # noqa: E402
from skills.executable.math.helpers import (  # noqa: E402
    _explicit_answer, _is_expression, _is_gameof24, _is_verbose_answer,
    _question, _reduce_fraction,
)

D = "math"



_AIME = re.compile(r"\bAIME\b|integers? (?:are )?(?:between|from) 0+ (?:and|to) 9\{?9\}?9"
                   r"|answers? are integers? from 0+ to 999"
                   r"|find \$?\s*(?:m|p)\s*\+\s*(?:n|q)\b\$?", re.I)


def _bounds_answer(text, arg, ctx):
    """The committed answer violates a hard constraint the problem states."""
    q, a = str(ctx.get("question", "")), str(arg).strip().strip("$")
    if _AIME.search(q):
        if re.fullmatch(r"-?\d+", a) and not 0 <= int(a) <= 999:
            return (f"the final answer {a} is outside the AIME range 0..999 the problem requires — "
                    f"re-read what quantity the problem asks for (often m+n or a remainder), and compute that")
        if re.fullmatch(r"-?\d+\.\d+|-?\d+/\d+", a):
            return (f"the final answer {a} is not an integer, but the problem asks for one in 0..999 — "
                    f"if the answer is a fraction m/n in lowest terms, the requested value is usually m+n")
    if re.search(r"\bpositive\b", q, re.I) and re.match(r"-", a):
        return f"the final answer {a} is negative, but the problem requires a positive value"
    if re.search(r"probabilit", q, re.I) and not _AIME.search(q) \
            and re.fullmatch(r"-?\d*\.\d+|-?\d+/\d+|\\frac\{[^}]+\}\{[^}]+\}", a):
        # Only a value SHAPED like a probability (decimal/fraction) is judged
        # against [0,1]. An integer on a "find m+n" problem is a count -- the
        # old check told the model its correct-typed 351 was "a probability
        # outside [0,1]", which is disinformation, and the control beat it.
        try:
            v = float(a.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")"))
            if not 0 <= v <= 1:
                return f"the final answer {a} is a probability outside [0, 1]"
        except (ValueError, TypeError):
            pass
    from skills.pf_template import steps as _steps
    for st in _steps(text):
        r = check_bounds(st)
        if r is not None:
            return r.verdict
    return None


def _eq_sub_answer(text, arg, ctx):
    # pass the whole record through: the checker may attach a solved value
    return check_equation_substitution(str(ctx.get("question", "")), str(arg))


# ── the nine verify families (seed triggers + seed hints + HASP checkers) ─

@pf_skill("arithmetic_slip", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="the reasoning contains an arithmetic chain "
                                "(three or more operators)"),
          summary="Recompute the arithmetic the reasoning states — sums, products, "
                  "fraction reductions, modular results — and flag any wrong value.")
class ArithmeticSlip:
    HINT = ("the final arithmetic chain (sums, products, fraction reductions, modular "
            "results) is numerically correct.")

    def should_activate(self, ctx, action, arg) -> bool:
        # An arithmetic CHAIN, not scattered symbols: "= 3" bullet points and
        # hyphens tripped the old count on all 12 of 12 audit rollouts. Require
        # digit-operator-digit at least twice, or a self-written compute step.
        r = ctx.reasoning + " " + arg
        return (len(re.findall(r"\d\s*[+\-*/]\s*\d", r)) >= 2
                or "compute[" in r.lower())

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_compute_observation)
        if f:
            return verdict(ctx, f, redo=True)
        if stalled(ctx):
            return continuation(ctx)
        # No reminder fallback: on 160 real rollouts it fired on 92% of wrong
        # AND 92% of correct ones -- zero discrimination. The concrete-verdict
        # path above is what carries this family.
        return abstain()


@pf_skill("algebraic_sign_error", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="the reasoning contains a negative sign, a radical or "
                                "an inequality"),
          summary="Check signs: distributing a negative, flipping an inequality, the ± "
                  "branch of a root, sign slips in subtraction.")
class AlgebraicSignError:
    HINT = ("algebraic signs: distributing a negative, flipping inequalities when "
            "multiplying by a negative, the ± branch of √, sign slips in subtraction.")

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"[-−]|sqrt|√|<|>|\\leq|\\geq|\\neq", ctx.reasoning))

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_interval_sign)
        if f:
            return verdict(ctx, f)
        if stalled(ctx):
            return continuation(ctx)
        # generic reminder dropped: -5% real lift from firing on 78% of correct rollouts; the interval-sign
        # verdict above is the value.
        return abstain()


@pf_skill("case_incompleteness", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="the reasoning splits into cases, or uses |x| / WLOG"),
          summary="Check that a case analysis is exhaustive and mutually exclusive — no "
                  "missing boundary, absolute-value or parity case.")
class CaseIncompleteness:
    HINT = ("the case analysis is exhaustive and mutually exclusive (no missing boundary "
            "/ |x| / even-odd cases).")

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"case\b|\|x\||absolute\s+value|WLOG|without\s+loss",
                              ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        r = ctx.reasoning
        # The one case defect that is checkable without solving the problem:
        # an absolute value / ± in play, a case split opened, and only one
        # branch ever written. Everything vaguer was a generic reminder that
        # fired on 35% of CORRECT real rollouts (lift -20%) and repaired
        # nothing in the efficacy test -- that reminder is gone.
        has_split_source = bool(re.search(r"\|[^|]{1,20}\||absolute\s+value|\\pm|±", r))
        n_cases = len(re.findall(r"\bcase\s*\d|\bcase\s+(?:one|two|1|2)\b", r, re.I))
        one_branch = (n_cases == 1
                      or (has_split_source and re.search(r"\bassume\b|\bWLOG\b|only consider", r, re.I)
                          and n_cases == 0))
        if has_split_source and one_branch:
            return inject(f"[{ctx.skill_id} {ctx.anchor.tag()}] the problem involves an "
                          f"absolute value or a ± branch, but the reasoning works only one "
                          f"branch. Work the other branch (the negative case) explicitly, "
                          f"then combine before finalizing.",
                          reason="single-branch case split")
        if stalled(ctx):
            return continuation(ctx)
        return abstain()
@pf_skill("boundary_violation", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the problem states a domain constraint, or the answer "
                                "is non-integer / negative"),
          summary="Check the committed answer against the problem's hard constraints — "
                  "range, integrality, positivity, domain.")
class BoundaryViolation:
    HINT = ("the answer respects all domain constraints (positive integer? AIME 0..999? "
            "log/sqrt domain positive? denominator nonzero?).")

    def should_activate(self, ctx, action, arg) -> bool:
        # _AIME already knows the LaTeX spellings ("Find $m+n$"); the local
        # pattern here didn't, which kept Detect off the audited failure. A
        # fraction committed on any constrained problem is also worth a look.
        if _AIME.search(ctx.question) or re.search(
                r"positive integer|\\boxed\{[^}]*\}", ctx.question + arg, re.I):
            return True
        return bool(re.search(r"\d+\.\d+|-\d+|\d+\s*/\s*\d+", arg))

    def intervene(self, ctx, action, arg):
        # "Find m+n" answered with the fraction m/n is the one boundary
        # violation whose repair is computable from the answer itself: reduce,
        # add, and the AIME convention (m, n coprime, sum in 0..999) checks it.
        # This is exactly the audited failure -- committed 5/16, gold 21.
        is_mn = re.search(r"(?:find|compute)\s+\$?\s*(?:m|p)\s*\+\s*(?:n|q)\b"
                          r"|\bm\s*\+\s*n\b", ctx.question, re.I)
        frac = re.fullmatch(r"(\d+)\s*/\s*(\d+)", (arg or "").strip().strip("$"))
        if frac and is_mn:
            from math import gcd
            m_, n_ = int(frac.group(1)), int(frac.group(2))
            g = gcd(m_, n_) or 1
            total = m_ // g + n_ // g
            if 0 <= total <= 999:
                return redirect(action, str(total),
                                because=(f"the problem asks for m+n, not the fraction: "
                                         f"{m_//g}/{n_//g} gives m+n = {total}"))
        # The other spelling of the same slip: the answer IS a sum m+n, but of
        # an unreduced fraction. Proof it happened is in the reasoning -- the
        # last "m = A and n = B" (or last fraction) summing exactly to the
        # committed value while gcd(A, B) > 1.
        if is_mn and re.fullmatch(r"\d+", (arg or "").strip()):
            from math import gcd
            pair = None
            mm = list(re.finditer(r"m\s*=\s*(\d+)\b.{0,20}?n\s*=\s*(\d+)", ctx.reasoning))
            if mm:
                pair = (int(mm[-1].group(1)), int(mm[-1].group(2)))
            else:
                fr = list(re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", ctx.reasoning))
                if fr:
                    pair = (int(fr[-1].group(1)), int(fr[-1].group(2)))
            if pair and sum(pair) == int(arg) and gcd(*pair) > 1:
                g = gcd(*pair)
                total = pair[0] // g + pair[1] // g
                if 0 <= total <= 999 and total != int(arg):
                    return redirect(action, str(total),
                                    because=(f"m/n must be in lowest terms: "
                                             f"{pair[0]}/{pair[1]} reduces to "
                                             f"{pair[0]//g}/{pair[1]//g}, so m+n = {total}"))
        f = answer_finding(ctx, arg, _bounds_answer)
        if f:
            return verdict(ctx, f)
        if stalled(ctx):
            return continuation(ctx)
        # reminder fallback dropped: -8% real lift; the concrete range and
        # positivity verdicts above are what this skill is for.
        return abstain()


@pf_skill("substitution_invalid", domain=D, needs_helper=True,
          anchor=Anchor(level="step", evidence="reminder",
                        trigger="the reasoning introduces a substitution "
                                "(let u = …, change of variable)"),
          summary="Check that a substitution preserved the domain, handled branches, and "
                  "was mapped back to the original variable.")
class SubstitutionInvalid:
    HINT = ("any substitution (u-sub, trig-sub, y=x²) preserved the domain, handled "
            "branches, and was mapped back to the original variable.")

    def should_activate(self, ctx, action, arg) -> bool:
        # "Substituting back to verify" is the practice this family exists to
        # encourage -- the audit's only fire was on a correct rollout doing
        # exactly that. Fire on an INTRODUCED substitution only.
        r = ctx.reasoning
        intro = re.search(r"\blet\s+\w\s*=|change\s+of\s+variable|\bsub(?:stitut\w*)?\s+u\b|\bu\s*=",
                          r, re.I)
        verify_only = (re.search(r"substitut", r, re.I)
                       and re.search(r"(?:substitut\w*\s+(?:back|.{0,20}back into)|"
                                     r"back.{0,12}substitut|verify|check)", r, re.I)
                       and not intro)
        return bool(intro) and not verify_only

    def intervene(self, ctx, action, arg):
        iv = helper_verdict(ctx, self.HINT)
        if iv:
            return iv
        if stalled(ctx):
            return continuation(ctx)
        return reminder(ctx, self.HINT) or abstain()


@pf_skill("overgeneralization", domain=D, needs_helper=True,
          anchor=Anchor(level="step", evidence="reminder",
                        trigger="the reasoning names a theorem (AM-GM, Cauchy-Schwarz, "
                                "L'Hôpital, geometric series, …)"),
          summary="Check that every named theorem's preconditions actually hold before "
                  "it is applied.")
class Overgeneralization:
    HINT = ("every named theorem's preconditions hold (AM-GM: non-negatives; geometric "
            "series: |r|<1; L'Hôpital: 0/0 or ∞/∞; Pythagoras: right angle).")
    NAMED = ("am-gm", "cauchy-schwarz", "jensen", "pythagoras", "pythagorean", "l'hopital",
             "l'hôpital", "binomial theorem", "geometric series", "arithmetic series",
             "telescoping", "mean value")

    def should_activate(self, ctx, action, arg) -> bool:
        low = ctx.reasoning.lower()
        return any(k in low for k in self.NAMED)

    def intervene(self, ctx, action, arg):
        iv = helper_verdict(ctx, self.HINT)
        if iv:
            return iv
        if stalled(ctx):
            return continuation(ctx)
        return reminder(ctx, self.HINT) or abstain()


@pf_skill("units_dimension_mismatch", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the problem asks for a probability, count, area, volume "
                                "or perimeter"),
          summary="Check that units and dimensions are consistent — a probability in "
                  "[0,1], an integer count, length not mixed with area.")
class UnitsDimensionMismatch:
    HINT = ("units/dimensions are consistent (probability ∈ [0,1], integer counts, "
            "area/volume not mixed with length).")

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"probability|probab|expected|count|area|volume|perimeter",
                              ctx.question, re.I))

    def intervene(self, ctx, action, arg):
        f = answer_finding(ctx, arg, _bounds_answer)
        if f:
            return verdict(ctx, f)
        if stalled(ctx):
            return continuation(ctx)
        # generic reminder dropped: -2% real lift, 1/6 vs ctrl 2/6 in the
        # efficacy test -- worse than saying nothing.
        return abstain()


@pf_skill("proof_step_gap", domain=D, needs_helper=True,
          anchor=Anchor(level="step", evidence="reminder",
                        trigger="the reasoning leans on 'clearly' / 'obviously' / "
                                "'it follows'"),
          summary="Check that no essential algebraic step was skipped behind a word like "
                  "'clearly' or 'therefore'.")
class ProofStepGap:
    HINT = ("no essential algebraic step was skipped behind 'clearly'/'obviously'/"
            "multi-step leaps that could hide an error.")

    def should_activate(self, ctx, action, arg) -> bool:
        # "therefore/hence" is how mathematics is written -- counting them
        # put this reminder on half the CORRECT real rollouts. Only the words
        # that actually paper over a leap, and at least two of them.
        return 2 <= len(re.findall(r"\b(?:clearly|obviously|trivially|it is easy)\b",
                                   ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        iv = helper_verdict(ctx, self.HINT)
        if iv:
            return iv
        if stalled(ctx):
            return continuation(ctx)
        # Reminder dropped: replayed on its own scenarios, the pf arm and a
        # generic nudge repaired the identical set -- the text carries no
        # information the model lacks. Detection stays (stall rescue).
        return abstain()


@pf_skill("verification_missing", domain=D,
          anchor=Anchor(level="final", evidence="reminder",
                        trigger="the reasoning never checks its answer back against the "
                                "problem"),
          summary="Flag an answer that was never substituted back into the original "
                  "equation or constraints to confirm it satisfies them.")
class VerificationMissing:
    HINT = ("the answer was plugged BACK into the original equation / constraints to "
            "confirm it actually satisfies them.")

    def should_activate(self, ctx, action, arg) -> bool:
        return not re.search(r"\b(check|verify|substitute|plug.*back|plug.*in)\b",
                             ctx.reasoning, re.I)

    def intervene(self, ctx, action, arg):
        if stalled(ctx):
            return continuation(ctx)
        return reminder(ctx, self.HINT) or abstain()


# ── the hand-written evidence skills (mined from the wrong cases) ────────

@pf_skill("compute_observation_verify", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="a `compute[...]` action whose Observation the model "
                                "wrote itself"),
          summary="Re-evaluate every self-written compute[...] Observation with sympy and "
                  "give the true value when it is wrong.")
class ComputeObservationVerify:
    """Works because the model wrote both the expression and its result, so the
    claim is self-contained and machine-checkable."""

    def should_activate(self, ctx, action, arg) -> bool:
        return "compute[" in ctx.reasoning.lower()

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_compute_observation)
        if f:
            # When the committed answer IS the miscopied value, the recomputed
            # truth is the answer -- no downstream reasoning to re-run.
            w, t = f.data.get("written"), f.data.get("truth")
            if w and t and (arg or "").strip().strip("$") in (w, f"\\boxed{{{w}}}"):
                return redirect(action, t, because=f.verdict[:160])
            return verdict(ctx, f, redo=True)
        if stalled(ctx):
            return continuation(ctx)
        return abstain()


@pf_skill("unsupported_final_answer", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="a guess phrase right before the answer is committed"),
          summary="Flag an answer committed on a guess — 'I will go with', 'known "
                  "result', 'given the time' — rather than on a derivation.")
class UnsupportedFinalAnswer:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\b(I(?:'ll| will) go with|known result|given the time|"
                              r"probably|I think|my best guess)\b", ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_unsupported_final)
        if f:
            f.verdict += (" — do not keep the guess: derive the value from the "
                          "work above, computing the key quantity explicitly")
            return verdict(ctx, f)
        if stalled(ctx):
            return continuation(ctx)
        return abstain()


@pf_skill("interval_sign_check", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="a sign claim about an expression over an interval"),
          summary="Evaluate each sign claim at a test point inside the stated interval; "
                  "a wrong sign is provable evidence.")
class IntervalSignCheck:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"[<>]\s*0\b|\bpositive on\b|\bnegative on\b",
                              ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_interval_sign)
        if f:
            return verdict(ctx, f, redo=True)
        if stalled(ctx):
            return continuation(ctx)
        return abstain()


@pf_skill("counting_small_case_check", domain=D,
          anchor=Anchor(level="step", evidence="executed",
                        trigger="a counting or combinatorial claim with a small parameter"),
          summary="Re-count a combinatorial claim by brute force on a small case and "
                  "compare against the number the reasoning states.")
class CountingSmallCaseCheck:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\b(choose|binom|combination|permutation|there are \d+|"
                              r"number of ways)\b|\bC\(\s*\d+\s*,", ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_stated_count)
        if f:
            # The committed answer IS the miscounted value: the recomputed
            # count is the answer, nothing downstream left to redo.
            stated = re.search(r"`[^`]*=\s*(\S+?)`", f.verdict or "")
            if f.fix and stated and (arg or "").strip().strip("$") == stated.group(1):
                return redirect(action, str(f.fix), because=f.verdict[:160])
            return verdict(ctx, f, redo=True)
        if stalled(ctx):
            return continuation(ctx)
        return abstain()


@pf_skill("claimed_unique_solution_search", domain=D,
          anchor=Anchor(level="step", evidence="executed",
                        trigger="a uniqueness claim — 'the only', 'unique', 'exactly one'"),
          summary="Search a bounded range for solutions a uniqueness claim missed.")
class ClaimedUniqueSolutionSearch:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\b(the only|unique|exactly one)\b", ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        # Either the uniqueness claim can be falsified by actually solving, or
        # this skill has nothing to say: the trigger-only verdict repaired 0/6
        # while a generic nudge repaired 0/6 too -- announcing "a uniqueness
        # claim was detected" carries no information the model lacks.
        r = ctx.reasoning
        m = re.search(r"(?:the only|unique|exactly one)[^.\n]{0,80}", r, re.I)
        # the CLAIM lives in the reasoning; the EQUATION lives in the question
        eq = None
        if m:
            for cand_eq in re.finditer(r"\$([^=$\n]{1,40})=([^=$\n]{1,40})\$", ctx.question):
                if set(re.findall(r"[a-z]", cand_eq.group(1) + cand_eq.group(2))) - {"e"}:
                    eq = cand_eq
                    break
        if eq:
            try:
                import sympy
                from sympy.parsing.sympy_parser import (standard_transformations,
                                                        implicit_multiplication_application)
                _TR = standard_transformations + (implicit_multiplication_application,)
                lhs, rhs = eq.group(1), eq.group(2)
                syms = sorted(set(re.findall(r"[a-z]", lhs + rhs)) - {"e"})
                if len(syms) == 1:
                    v = sympy.Symbol(syms[0])
                    L = sympy.parse_expr(lhs.replace("^", "**"), transformations=_TR,
                                         local_dict={syms[0]: v})
                    Rr = sympy.parse_expr(rhs.replace("^", "**"), transformations=_TR,
                                          local_dict={syms[0]: v})
                    sols = [x for x in sympy.solve(L - Rr, v) if x.is_real]
                    if len(sols) > 1:
                        # The problem's own constraints often pick the root: if
                        # filtering by them leaves exactly ONE, that root IS
                        # the answer -- committed directly, since sympy already
                        # verified it satisfies the equation.
                        q = ctx.question
                        keep = sols
                        if re.search(r"\bpositive\b", q, re.I):
                            keep = [x for x in keep if x > 0]
                        if re.search(r"\binteger\b", q, re.I):
                            keep = [x for x in keep if x == sympy.floor(x)]
                        if re.search(r"\bnegative\b", q, re.I):
                            keep = [x for x in keep if x < 0]
                        if len(keep) == 1 and len(keep) < len(sols):
                            val = (str(keep[0]) if keep[0] == sympy.nsimplify(keep[0])
                                   else f"{float(keep[0].evalf()):.6g}")
                            if val != (arg or "").strip():
                                return redirect(action, val,
                                                because=(f"{lhs.strip()} = {rhs.strip()} has "
                                                         f"{len(sols)} real roots; the problem's "
                                                         f"constraint keeps only {val}"))
                        return inject(f"[{ctx.skill_id} {ctx.anchor.tag()}] the reasoning "
                                      f"claims a unique solution, but {lhs.strip()} = "
                                      f"{rhs.strip()} has {len(sols)} real solutions: "
                                      f"{', '.join(str(x) for x in sols[:4])}. Check every "
                                      f"one against the problem's constraints.",
                                      reason="uniqueness claim falsified by solving")
            except Exception:
                pass
        if stalled(ctx):
            return continuation(ctx)
        return abstain()
@pf_skill("unsupported_known_result", domain=D, needs_helper=True,
          anchor=Anchor(level="step", evidence="helper",
                        trigger="the reasoning cites a 'known result' or 'well-known' fact"),
          summary="Audit a cited 'known result' that the reasoning never derives.")
class UnsupportedKnownResult:
    HINT = ("the cited 'known result' actually says what it is being used for, with the "
            "same hypotheses.")

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\b(known result|well[- ]known|standard result|it is known)\b",
                              ctx.reasoning, re.I))

    def intervene(self, ctx, action, arg):
        iv = helper_verdict(ctx, self.HINT)
        if iv:
            return iv
        if stalled(ctx):
            return continuation(ctx)
        # generic reminder dropped: -5% real lift, 5/14 of its injections landed on correct rollouts.
        return abstain()


# ── recorded negatives: kept with their status, gated by Detect ──────────

@pf_skill("equation_substitution_check", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the problem states an equation in one unknown and asks "
                                "for it"),
          summary="Substitute the committed answer back into the equation the problem "
                  "states and flag it if it does not satisfy it.")
class EquationSubstitutionCheck:
    """Targets a real pattern, but the trigger is narrow: it needs the problem to
    state an equation in one unknown and to ask for that unknown."""

    def should_activate(self, ctx, action, arg) -> bool:
        return "=" in ctx.question and bool(arg.strip())

    def intervene(self, ctx, action, arg):
        f = answer_finding(ctx, arg, _eq_sub_answer)
        # sympy solved the same single-variable equation the answer failed,
        # and the solution substituted back clean -- committing it is backed
        # by the same arithmetic that rejected the original.
        if f and f.data.get("solved"):
            return redirect(action, str(f.data["solved"]),
                            because=f.verdict[:160])
        return verdict(ctx, f) if f else abstain()


@pf_skill("runaway_enumeration_breaker", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="a step where most lines repeat the same pattern"),
          summary="Stop an enumeration that has degenerated into repeating the same line.")
class RunawayEnumerationBreaker:
    """Aimed at rollouts that degenerate into repeating one line until the budget
    runs out. Models that stall at `Action:` instead never reach this state."""

    def should_activate(self, ctx, action, arg) -> bool:
        return len(ctx.reasoning.splitlines()) > 30

    def intervene(self, ctx, action, arg):
        f = first_finding(ctx, check_runaway_enumeration)
        return verdict(ctx, f) if f else abstain()



# ── answer-shape skills: they compute a better value and write it in ─────
# All four rewrite the committed FINAL, which is what the measured library did.
# The replacement is derived from the answer itself -- reduce the fraction, wrap
# it in \boxed{}, pull the value out of a \boxed{} it was already in, cut a
# repetition -- so there is no judgement to check and nothing for the policy to
# weigh. Stating those as evidence instead costs a turn and adds a chance the
# model does not comply, and buys nothing: none of these four can be wrong in a
# way an extra turn would catch. `redirect` keeps the action type unchanged and
# replaces only its argument.


@pf_skill("simplification_incomplete", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the committed answer is a fraction that reduces further"),
          summary="Check that a fractional answer is in lowest terms, and give the "
                  "reduced form when it is not.")
class SimplificationIncomplete:
    NOTE = ("the committed answer is not in lowest terms; the problem expects a fully "
            "reduced fraction.")

    def should_activate(self, ctx, action, arg) -> bool:
        # The ANSWER must be a fraction, not merely contain one: "n^2+7n-1/2"
        # carries a /2 that is part of an expression, and "reducing" it
        # rewrote a symbolic answer into nonsense on two real rollouts.
        a = (arg or "").strip().strip("$")
        if not re.fullmatch(r"-?\d+\s*/\s*\d+|\\frac\{-?\d+\}\{-?\d+\}", a):
            return False
        return _reduce_fraction(a) is not None

    def intervene(self, ctx, action, arg):
        reduced = _reduce_fraction(arg)
        if reduced and reduced != arg:
            return redirect(action, reduced, because=self.NOTE)
        return abstain()


@pf_skill("final_format_error", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the committed answer is not in the dataset's expected "
                                "form (skipped for expression answers and Game-of-24)"),
          summary="Check the answer's shape against the dataset convention — a bare "
                  "integer for AIME, \\boxed{} elsewhere, no units.")
class FinalFormatError:
    NOTE = ("the committed answer is not in the form this dataset expects (a bare value, "
            "no units, no surrounding sentence).")
    _UNITS = re.compile(r"\s*(units?|cm|kg|sec(?:onds?)?|degrees?|meters?|m)\b\.?$", re.I)

    def should_activate(self, ctx, action, arg) -> bool:
        a = (arg or "").strip()
        if not a:
            return False
        q = _question(ctx.raw)
        if _is_gameof24(q) or _is_expression(a):
            return False
        if _AIME.search(q):
            # only a wrapped-or-decorated integer needs unwrapping
            return not re.fullmatch(r"-?\d{1,3}", a)
        # On 160 real rollouts the boxing branch fired on 78% of CORRECT
        # answers (a bare "27" is how the model normally commits) and on
        # almost no wrong ones -- the judge strips \boxed anyway, so wrapping
        # bare values is pure noise. Only an answer carrying actual format
        # debris (units, prose, "The answer is ...") needs normalising.
        if "\\boxed{" in a or "$" not in q:
            return False
        return bool(re.search(r"[A-Za-z]{2,}", a) or " " in a.strip())

    def intervene(self, ctx, action, arg):
        # Two format repairs, both provable from the answer alone; anything
        # beyond that is a correctness problem, not a format one, and belongs
        # to boundary_violation's inject. The old branch grabbed the first
        # 1-3-digit run out of a non-integer answer ("5/16" -> "5"), which is
        # inventing an answer, not repairing a format.
        ans, q = arg.strip(), _question(ctx.raw)
        if _AIME.search(q):
            inner = (_explicit_answer(ans) or ans).strip()
            if re.fullmatch(r"\\boxed\{\s*(-?\d{1,3})\s*\}", inner):
                inner = re.fullmatch(r"\\boxed\{\s*(-?\d{1,3})\s*\}", inner).group(1)
            if re.fullmatch(r"-?\d{1,3}", inner) and 0 <= int(inner) <= 999:
                inner = str(int(inner))          # \boxed{042} -> 42, no invented digits
                if inner != ans:
                    return redirect(action, inner, because=self.NOTE)
            return abstain()
        stripped = self._UNITS.sub("", ans).strip()
        if "\\boxed{" in stripped:
            new = stripped
        else:
            inner = _explicit_answer(stripped) or stripped
            new = (f"\\boxed{{{inner}}}"
                   if inner and len(inner) <= 40 and " " not in inner
                   and not _is_expression(inner) else stripped)
        return redirect(action, new, because=self.NOTE) if new and new != ans else abstain()
@pf_skill("boxed_extraction", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="a verbose answer carrying an explicit \\boxed{} or "
                                "'answer is' marker"),
          summary="Pull the graded value out of an answer that wrapped it in a sentence.")
class BoxedExtraction:
    NOTE = ("the committed answer is a sentence rather than a value; the graded answer is "
            "whatever sits inside it.")

    def should_activate(self, ctx, action, arg) -> bool:
        if not _is_verbose_answer(arg):
            return False
        # Only an EXPLICIT marker counts. The last-number fallback would fire on
        # any answer that happens to end in a digit.
        ext = _explicit_answer(arg)
        return ext is not None and not _is_expression(ext)

    def intervene(self, ctx, action, arg):
        ext = _explicit_answer(arg)
        if ext and ext.strip() and ext.strip() != arg.strip():
            return redirect(action, ext.strip(), because=self.NOTE)
        return abstain()


@pf_skill("repetition_circuit_breaker", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the committed answer is a degenerate repetition"),
          summary="Collapse an answer that degenerated into a repeated line down to its "
                  "single value.")
class RepetitionCircuitBreaker:
    NOTE = "the committed answer is a degenerate repetition rather than a single value."

    @staticmethod
    def _repeats(a: str) -> bool:
        a = a.strip()
        lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
        if len(lines) >= 4 and len(set(lines)) <= max(1, len(lines) // 3):
            return True
        toks = a.split()
        if len(toks) >= 8 and len(set(toks)) <= 3:
            return True
        # A single token that is a short pattern cycled ("121212121212",
        # "3.3333333333"): the classic period test -- s occurs inside s+s
        # before its own length -- with the period at most a third of it.
        if len(toks) == 1 and len(a) >= 9:
            period = (a + a).find(a, 1)
            return 0 < period <= len(a) // 3
        return False

    def should_activate(self, ctx, action, arg) -> bool:
        return self._repeats(arg)

    def intervene(self, ctx, action, arg):
        a = arg.strip()
        first = next((ln.strip() for ln in a.splitlines() if ln.strip()), "")
        # Single-line degenerations (which is what the base model actually
        # emits) reduce the same way they were detected: a repeated token
        # collapses to the token, a cycled digit string to one period.
        if first == a:
            toks = a.split()
            if len(toks) >= 8 and len(set(toks)) <= 3:
                first = toks[0]
            else:
                period = (a + a).find(a, 1)
                if 0 < period <= len(a) // 3:
                    first = a[:period]
        explicit = _explicit_answer(a)
        new = (explicit.strip() if explicit and not _is_expression(explicit)
               and not _is_expression(first) else first)
        return redirect(action, new, because=self.NOTE) if new and new != a else abstain()
