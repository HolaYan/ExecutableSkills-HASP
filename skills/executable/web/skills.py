"""web PF skills — every skill writes out both modules.

    Detect   should_activate(state)  — does this state and proposed action
                                       match the failure pattern?
    Repair   intervene(state)        — redirect, inject corrective context, or
                                       abstain.

The four evidence skills mined from the web error cases. `answer_grounding_check`
is the one that separates wrong from correct answers usefully; the other three
are dormant or unmeasured and say so in their own docstrings.

**Web is not measured end to end**: it needs live search, so there is no offline
corpus to score these against. Treat their fire rates as the only evidence, and
read `question_entity_coverage` / `comparison_evidence_completeness` as
cautionary tales rather than tools.

Roughly half of wrong web rollouts fail because the evidence was never
retrieved, not because it was misread. That half is a *retrieval* failure,
invisible at inference without the gold — which is why the two containment-based
skills below are dormant.
"""
from __future__ import annotations

import importlib.util as _iu
import re
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[3]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills.pf_template import (  # noqa: E402
    as_action, as_evidence, call_base_intervene, correction,
    Anchor, Finding, Ctx, abstain, helper_verdict, pf_skill, redirect, verdict,
)

_spec = _iu.spec_from_file_location(
    "_hasp_web_checkers", str(Path(__file__).resolve().parent / "checkers.py"))
_C = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_C)

_IMPL_SPEC = _iu.spec_from_file_location(
    "_hasp_web_impls", str(Path(__file__).resolve().parent / "implementations.py"))
_IMPL = _iu.module_from_spec(_IMPL_SPEC)
_IMPL_SPEC.loader.exec_module(_IMPL)

D = "web"



_CMP_WORDS = ("which", "who", "whose", "more", "most", "less", "least", "first",
              "earlier", "earliest", "older", "oldest", "younger", "later", "latest",
              "larger", "longer", "higher", "lower", "before", "after")


def _run(ctx: Ctx, arg: str, checker) -> Finding | None:
    """Web checkers take `(step_context, answer)` and return a verdict string."""
    try:
        v = checker(ctx.raw, arg)
    except Exception:
        return None
    if isinstance(v, dict):        # {"verdict": ..., "fix": ...} -- see answer_finding
        return (Finding(verdict=v.get("verdict", ""),
                        fix=v.get("fix") or v.get("search"), data=v)
                if v.get("verdict") else None)
    return Finding(verdict=v) if v else None


def _search_for(ctx: Ctx, f, *, budget: int = 2):
    """Rewrite the answer into the search that would close the gap `f` names.

    Only when the checker named a target. A skill that knows *which* entity has
    no evidence knows what to search for, and taking the search is strictly
    better than asking for it -- the policy cannot decline. A skill that only
    knows the answer is unsupported has nothing better to search than what was
    searched already, so it states the gap instead.

    Guarded on budget: with fewer than `budget` steps left the rollout cannot
    afford a retrieval round-trip, and forcing one turns a weak answer into no
    answer at all.
    """
    if not (f and f.fix):
        return None
    if int(ctx.step_count or 0) >= int(ctx.max_steps or 10) - budget:
        return None
    return redirect("SEARCH", str(f.fix), because=f.verdict)


def _has_observations(ctx: Ctx) -> bool:
    return bool(str(ctx.all_read_contents).strip()
                or str(ctx.last_search_results_text).strip())


# ── the one with measured separation ─────────────────────────────────────

@pf_skill("answer_grounding_check", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="a committed answer that is not a yes/no or bare number, "
                                "with at least one observation to check it against"),
          summary="Check that every part of the committed answer actually appears in "
                  "something that was searched or read; name the part that does not.")
class AnswerGroundingCheck:
    """Yes/no and numeric answers are skipped by the checker — containment says
    nothing useful about them, and including them is what turns a grounding
    check into noise."""

    def should_activate(self, ctx, action, arg) -> bool:
        if not arg.strip() or not _has_observations(ctx):
            return False
        a = arg.strip().lower().rstrip(".")
        if a in ("yes", "no") or re.fullmatch(r"[\d.,%$ ]+", a):
            return False
        return True

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.answer_grounding)
        # Nothing was ever searched, so the question itself is a target that has
        # not been tried; once it has, re-running it returns the same page.
        if f and not int(ctx.search_count or 0):
            f.fix = ctx.question
        return _search_for(ctx, f) or (verdict(ctx, f) if f else abstain())


# ── helper-scoped: the evidence may already hold a different answer ──────

@pf_skill("evidence_answer_consistency", domain=D, needs_helper=True,
          anchor=Anchor(level="final", evidence="helper",
                        trigger="documents were read and an answer was committed"),
          summary="Check whether a passage that was read explicitly states a different "
                  "answer to the question, and quote it.")
class EvidenceAnswerConsistency:
    """Targets the evidence_present family — 42% of wrong web rollouts, where the
    gold is sitting inside an observation the model then answered past. Needs a
    helper: containment cannot tell "states a different answer" from "mentions
    the same words". Unmeasured end to end."""

    SCOPE = ("A passage already answers the question differently — quote it and give "
             "the answer it states.")

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(arg.strip()) and len(str(ctx.all_read_contents)) > 100

    def intervene(self, ctx, action, arg):
        iv = helper_verdict(ctx, self.SCOPE)
        if iv is None:
            return abstain()
        # The helper quotes the passage's answer in prose. Extracting a value
        # from prose is a guess -- unless the value appears VERBATIM in what
        # was actually read, which is this domain's example gate: an answer
        # sitting in the evidence cannot be an artifact of the helper's
        # phrasing. Only then is the rewrite taken; otherwise the quote is
        # injected and the policy decides.
        text = iv.context_text or ""
        evid = str(ctx.all_read_contents).lower()
        for tup in re.findall(r'"([^"]{2,60})"'
                              r"|\banswer(?:\s+it\s+states)?\s+is[:\s]+([^.;\n\x22]{2,60})",
                              text):
            cand = next(g for g in tup if g).strip().strip(".")
            if (cand and cand.lower() != str(arg).strip().lower()
                    and cand.lower() in evid):
                return redirect("FINAL", cand, because=text[:160])
        return iv


# ── recorded negatives: kept with their status, gated by Detect ──────────

@pf_skill("question_entity_coverage", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the question names an entity and searches were issued"),
          summary="Flag a named entity from the question that no search or observation "
                  "ever covered.")
class QuestionEntityCoverage:
    """Dormant. The failure it aims at is "the entity *was* searched, the fact
    was not retrieved", which entity containment cannot see. A relation probe —
    checking whether the relation the question asks about is missing — was tried
    and separates no better. Do not re-propose this shape without a different
    anchor."""

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", ctx.question)) \
            and _has_observations(ctx)

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.question_entity_coverage)
        return _search_for(ctx, f) or (verdict(ctx, f) if f else abstain())


@pf_skill("comparison_evidence_completeness", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="a comparison question naming two or more entities"),
          summary="Flag a comparison answered with evidence about only one of the things "
                  "being compared, and say which side is missing.")
class ComparisonEvidenceCompleteness:
    """Dormant, same root cause as `question_entity_coverage`: what is missing is
    a relation, not an entity name."""

    def should_activate(self, ctx, action, arg) -> bool:
        q = ctx.question.lower()
        if not any(w in q for w in _CMP_WORDS):
            return False
        entities = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", ctx.question))
        return len(entities) >= 2 and _has_observations(ctx)

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.comparison_evidence_completeness)
        return _search_for(ctx, f) or (verdict(ctx, f) if f else abstain())


@pf_skill("insufficient_exploration", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='an answer committed with no search or no read, and budget left'),
          summary='This answer is being given without enough evidence gathered.')
class InsufficientExploration:
    """Detect and Repair both delegate to `InsufficientExplorationPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.InsufficientExplorationPF()
    NOTE = ('this answer is being given without enough evidence gathered.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("hallucination", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a committed answer whose proper nouns or years appear in no gathered text'),
          summary='A number or entity in this answer does not appear in anything '
                  'that was searched or read.')
class Hallucination:
    """Detect and Repair both delegate to `HallucinationPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.HallucinationPF()
    NOTE = ('a number or entity in this answer does not appear in anything '
            'that was searched or read.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("temporal_confusion", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a year in the answer that appears in no document read'),
          summary='A year in the answer that appears in no document read.')
class TemporalConfusion:
    """Detect and Repair both delegate to `TemporalConfusionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.TemporalConfusionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("numerical_reasoning_error", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a number in the answer that appears in no document read'),
          summary='A number in the answer that appears in no document read.')
class NumericalReasoningError:
    """Detect and Repair both delegate to `NumericalReasoningPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.NumericalReasoningPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("negation_oversight", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the question carries a negation the reasoning never echoes'),
          summary='The question carries a negation the reasoning never echoes.')
class NegationOversight:
    """Detect and Repair both delegate to `NegationOversightPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.NegationOversightPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("citation_mismatch", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a proper-noun entity in a short answer, absent from the read '
                                'content'),
          summary='A proper-noun entity in a short answer, absent from the read '
                  'content.')
class CitationMismatch:
    """Detect and Repair both delegate to `CitationMismatchPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.CitationMismatchPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("outdated_information", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a recency question answered from documents whose newest year '
                                'is old'),
          summary='A recency question answered from documents whose newest year '
                  'is old.')
class OutdatedInformation:
    """Detect and Repair both delegate to `OutdatedInformationPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.OutdatedInformationPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("format_extraction_error", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the answer carries a prefix, suffix or markdown wrapper around '
                                'the value'),
          summary='This answer carries formatting artifacts around the value that '
                  'is actually being asked for.')
class FormatExtractionError:
    """Detect and Repair both delegate to `FormatExtractionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.FormatExtractionPF()
    NOTE = ('this answer carries formatting artifacts around the value that '
            'is actually being asked for.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("multi_hop_reasoning_failure", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a multi-hop question answered with nothing read, or with an answer no evidence contains'),
          summary='This answer skips a hop the question requires.')
class MultiHopReasoningFailure:
    """Detect and Repair both delegate to `MultiHopReasoningPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.MultiHopReasoningPF()
    NOTE = ('this answer skips a hop the question requires.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("answer_completeness", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a multi-part question answered in fewer than three words'),
          summary='A multi-part question answered in fewer than three words.')
class AnswerCompleteness:
    """Detect and Repair both delegate to `AnswerCompletenessPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.AnswerCompletenessPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("reasoning_error", domain=D,
          anchor=Anchor(level="final", evidence="helper",
                        trigger='a long reasoning trace containing a self-contradiction marker'),
          summary='A long reasoning trace containing a self-contradiction marker.')
class ReasoningError:
    """Detect and Repair both delegate to `ReasoningErrorPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ReasoningErrorPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    # as_action, not as_evidence: the implementation's WRONG branch already
    # gates its FINAL rewrite on the corrected value appearing verbatim in the
    # read evidence -- restating a gated rewrite as advice throws the gate away.
    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("evidence_synthesis", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a question entity that never appears in the read content'),
          summary='This answer does not combine the evidence that was collected.')
class EvidenceSynthesis:
    """Detect and Repair both delegate to `EvidenceSynthesisPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.EvidenceSynthesisPF()
    NOTE = ('this answer does not combine the evidence that was collected.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("comparison_analyzer", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a comparison question naming two or more distinct entities'),
          summary='This answer does not resolve the comparison the question asks '
                  'for.')
class ComparisonAnalyzer:
    """Detect and Repair both delegate to `ComparisonAnalyzerPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ComparisonAnalyzerPF()
    NOTE = ('this answer does not resolve the comparison the question asks '
            'for.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    # Demoted from executing its rewrite: the audit caught it turning a FINAL
    # into `SEARCH: Premier League` -- a query more generic than the three the
    # rollout had already tried -- with a set()-ordered pick of which side to
    # search. A repair that cannot name its target deterministically is stated,
    # not taken.
    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("search_depth_controller", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a complex question answered after too few searches, with '
                                'budget left'),
          summary='The search stopped before reaching the evidence this question '
                  'needs.')
class SearchDepthController:
    """Detect and Repair both delegate to `SearchDepthControllerPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.SearchDepthControllerPF()
    NOTE = ('the search stopped before reaching the evidence this question '
            'needs.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("claim_triangulation", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='specific facts asserted after reading exactly one document'),
          summary='Specific facts asserted after reading exactly one document.')
class ClaimTriangulation:
    """Detect and Repair both delegate to `ClaimTriangulationPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ClaimTriangulationPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        # The committed answer names the fact to verify, so the second-source
        # search writes itself: the answer plus the question's subject. Taking
        # that search replaces nothing irreversible -- the answer can be
        # recommitted one step later, now with corroboration -- so this is the
        # recoverable kind of rewrite. Below two steps of budget the round
        # trip cannot complete, and the hint is all that fits.
        subject = (_C._question_entities(ctx.question) or [""])[0]
        # (A "thick evidence" gate that counted the ANSWER's occurrences was
        # tried and retracted: a wrong answer repeated in the evidence is not
        # corroboration, and the gate blocked repairs -- 3/12 vs 5.)
        if (arg or "").strip() and subject and \
                int(ctx.step_count or 0) < int(ctx.max_steps or 10) - 2:
            query = " ".join((str(arg).strip() + " " + subject).split())[:120]
            return redirect("SEARCH", query,
                            because="single-source answer; fetching a second source")
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)

@pf_skill("answer_confidence_guard", domain=D,
          anchor=Anchor(level="final", evidence="helper",
                        trigger='the committed answer differs from one stated earlier after a '
                                'read'),
          summary='An earlier step, after reading a source, stated a different '
                  'answer, and nothing read since contradicts it.')
class AnswerConfidenceGuard:
    """Detect and Repair both delegate to `AnswerConfidenceGuardPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.AnswerConfidenceGuardPF()
    NOTE = ('an earlier step, after reading a source, stated a different '
            'answer, and nothing read since contradicts it.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("entity_constraint_check", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a multi-hop or comparison question, checked once per episode'),
          summary='A multi-hop or comparison question, checked once per episode.')
class EntityConstraintCheck:
    """Detect and Repair both delegate to `EntityConstraintCheckPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.EntityConstraintCheckPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("retrieval_failure", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a search query longer than 15 words, capped per episode'),
          summary='The evidence retrieved does not support this answer.')
class RetrievalFailure:
    """Detect and Repair both delegate to `RetrievalFailurePF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.RetrievalFailurePF()
    NOTE = ('the evidence retrieved does not support this answer.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("adversarial_distraction", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='search results carrying three or more conflict words'),
          summary='Search results returned for a multi-hop question, capped per '
                  'episode.')
class AdversarialDistraction:
    """Detect and Repair both delegate to `AdversarialDistractionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.AdversarialDistractionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("wrong_entity_confusion", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger="search results whose names do not match the question's, once "
                                'per episode'),
          summary="Search results whose names do not match the question's, once "
                  'per episode.')
class WrongEntityConfusion:
    """Detect and Repair both delegate to `WrongEntityConfusionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.WrongEntityConfusionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("decompose_complex_question", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='an early search restating a long multi-hop question'),
          summary='An early search on a question with two or more possessives.')
class DecomposeComplexQuestion:
    """Detect and Repair both delegate to `DecomposeComplexQuestionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.DecomposeComplexQuestionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        # A search that restates the whole multi-hop question retrieves pages
        # about none of its hops. The innermost entity IS the first hop -- in
        # "the capital of the country where X was born", nothing can be
        # answered before X -- so the rewrite is that entity, plus the leading
        # relation word for context. SEARCH->SEARCH: a wasted step at worst,
        # never a lost answer. Only when the query really is the question
        # restated; a query the model already narrowed is left alone.
        ents = _C._question_entities(ctx.question)
        if (action == "SEARCH" and ents and len((arg or "").split()) >= 10
                and _C._contains(arg, ents[-1])):
            hop = re.sub(r"[\u2019']s?$", "", ents[-1]).strip()
            return redirect("SEARCH", hop,
                            because="whole-question search on a multi-hop question; "
                                    "resolving the innermost entity first")
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)
@pf_skill("query_decomposition", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a search query carrying two or more question words'),
          summary='This question has parts that were never searched separately.')
class QueryDecomposition:
    """Detect and Repair both delegate to `QueryDecompositionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.QueryDecompositionPF()
    NOTE = ('this question has parts that were never searched separately.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("constraint_search", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a search on a question with four or more constraints'),
          summary='A constraint stated in the question was never used in any '
                  'search.')
class ConstraintSearch:
    """Detect and Repair both delegate to `ConstraintSearchPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ConstraintSearchPF()
    NOTE = ('a constraint stated in the question was never used in any '
            'search.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("iterative_refinement", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a search more than 50% similar to a recent one'),
          summary='The evidence gathered so far does not settle the question.')
class IterativeRefinement:
    """Detect and Repair both delegate to `IterativeRefinementPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.IterativeRefinementPF()
    NOTE = ('the evidence gathered so far does not settle the question.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("search_stall_reformulate", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a repeated search, or empty results after two attempts'),
          summary='The searches so far keep returning the same unhelpful results.')
class SearchStallReformulate:
    """Detect and Repair both delegate to `SearchStallReformulatePF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.SearchStallReformulatePF()
    NOTE = ('the searches so far keep returning the same unhelpful results.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("reading_comprehension_error", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='a READ of dense content — ten or more numbers or fifteen or more names'),
          summary='.')
class ReadingComprehensionError:
    """Detect and Repair both delegate to `ReadingComprehensionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ReadingComprehensionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("language_barrier", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='read content with more than 50 non-ASCII characters'),
          summary='Read content with more than 50 non-ascii characters.')
class LanguageBarrier:
    """Detect and Repair both delegate to `LanguageBarrierPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.LanguageBarrierPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("misinformation_detector", domain=D,
          anchor=Anchor(level="step", evidence="deterministic",
                        trigger='two or more documents read that disagree on dates or facts'),
          summary='Two or more documents read that disagree on dates or facts.')
class MisinformationDetector:
    """Detect and Repair both delegate to `MisinformationDetectorPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.MisinformationDetectorPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)
