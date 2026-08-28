"""
Program Functions — direct intervention hooks for skills.

Each ProgramFunction is checked on EVERY agent step. When activated, it
produces an Intervention that modifies the agent's behavior:
- ModifyAction: change action_type and/or arg
- InjectContext: append text to the next observation
- NoOp: log only, no change

Program functions are deterministic (no LLM calls) and designed to be
fast enough to run every step without measurable overhead.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from .quota import note_api_error

logger = logging.getLogger(__name__)


# ============================================================================
# Intervention types
# ============================================================================

class InterventionType(Enum):
    NOOP = "noop"
    MODIFY_ACTION = "modify_action"
    INJECT_CONTEXT = "inject_context"


@dataclass
class Intervention:
    """Result of a program function activation."""
    type: InterventionType = InterventionType.NOOP
    # For MODIFY_ACTION
    new_action_type: Optional[str] = None
    new_action_arg: Optional[str] = None
    # For INJECT_CONTEXT
    context_text: str = ""
    # Metadata
    reason: str = ""
    skill_id: str = ""

    def to_dict(self) -> dict:
        d = {"type": self.type.value, "skill_id": self.skill_id, "reason": self.reason}
        if self.type == InterventionType.MODIFY_ACTION:
            d["new_action_type"] = self.new_action_type
            d["new_action_arg"] = self.new_action_arg
        elif self.type == InterventionType.INJECT_CONTEXT:
            d["context_text"] = self.context_text[:200]
        return d


@dataclass
class PFRecord:
    """Record of a program function check (for metrics)."""
    skill_id: str
    step: int
    activated: bool
    intervention_type: str = "noop"
    reason: str = ""
    # Pre/post intervention values (populated when activated=True)
    original_action: Optional[str] = None
    original_arg: Optional[str] = None
    new_action_type: Optional[str] = None
    new_action_arg: Optional[str] = None
    context_text: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "skill_id": self.skill_id,
            "step": self.step,
            "activated": self.activated,
            "intervention_type": self.intervention_type,
            "reason": self.reason,
        }
        if self.original_action is not None:
            d["original_action"] = self.original_action
            d["original_arg"] = self.original_arg
        if self.new_action_type is not None:
            d["new_action_type"] = self.new_action_type
            d["new_action_arg"] = self.new_action_arg
        if self.context_text is not None:
            d["context_text"] = self.context_text
        return d


# ============================================================================
# Base class
# ============================================================================

class ProgramFunction:
    """Base class for skill program functions.

    PFs are checked on every agent step. When activated, they produce an
    Intervention (modify action, inject context, or noop).

    PFs that set ``needs_helper = True`` will receive a PF helper in
    ``intervene()`` when one is available. They MUST still work (with a
    degraded code-only fallback) when ``teacher`` is None.
    """
    skill_id: str = ""
    needs_helper: bool = False  # Override to True to receive PF helper

    def should_activate(self, step_context: Dict[str, Any], action_type: str, arg: str) -> bool:
        """Check if this function should activate on the current step."""
        raise NotImplementedError

    def intervene(self, step_context: Dict[str, Any], action_type: str, arg: str,
                  teacher=None) -> Intervention:
        """Produce an intervention. Only called if should_activate() returns True.

        Args:
            step_context: Current step context dict.
            action_type: Current action type.
            arg: Current action argument.
            teacher: Optional PF helper (APIModelWrapper) for LLM-assisted
                     intervention. Only passed when ``needs_helper=True`` and
                     a PF helper is configured. PFs MUST handle ``teacher=None``.
        """
        raise NotImplementedError


# ============================================================================
# Registry
# ============================================================================

_PF_REGISTRY: Dict[str, ProgramFunction] = {}


def register_pf(skill_id: str):
    """Decorator to register a program function class."""
    def decorator(cls):
        cls.skill_id = skill_id
        _PF_REGISTRY[skill_id] = cls()
        return cls
    return decorator


def get_all_program_functions() -> Dict[str, ProgramFunction]:
    return dict(_PF_REGISTRY)


def get_program_function(skill_id: str) -> Optional[ProgramFunction]:
    return _PF_REGISTRY.get(skill_id)


# ============================================================================
# Executor — runs all active PFs for a step
# ============================================================================

def execute_program_functions(
    active_skill_ids: List[str],
    step_context: Dict[str, Any],
    action_type: str,
    arg: str,
    reasoning: str = "",
    disabled_pfs: Optional[set] = None,
    teacher_model=None,
) -> Tuple[str, str, List[PFRecord], List[str]]:
    """Execute all active program functions for the current step.

    Args:
        active_skill_ids: Skill IDs active in this episode.
        step_context: Current step context dict.
        action_type: Current action type (SEARCH/READ/FINAL/SUMMARY).
        arg: Current action argument.
        reasoning: Agent's thought/reasoning text.
        teacher_model: Optional PF helper for PFs with needs_helper=True.

    Returns:
        (final_action_type, final_arg, records, context_injections)
        - records: list of PFRecord for metrics
        - context_injections: list of text strings to inject after observation
    """
    records = []
    context_injections = []
    current_action = action_type
    current_arg = arg

    # Add reasoning to step_context for PFs that need it
    step_context_with_reasoning = dict(step_context)
    step_context_with_reasoning["thought"] = reasoning

    # Per-PF fire counts (persisted across steps via step_context)
    pf_fire_counts = step_context.get("_pf_fire_counts", {})
    step_context["_pf_fire_counts"] = pf_fire_counts
    _MAX_MODIFY_FIRES = 2  # Max MODIFY_ACTION per PF per episode

    for skill_id in active_skill_ids:
        if disabled_pfs and skill_id in disabled_pfs:
            continue
        pf = _PF_REGISTRY.get(skill_id)
        if pf is None:
            continue

        step = step_context.get("step_count", 0)

        try:
            if pf.should_activate(step_context_with_reasoning, current_action, current_arg):
                # Pass PF helper only to PFs that declared they need it
                if pf.needs_helper and teacher_model is not None:
                    intervention = pf.intervene(
                        step_context_with_reasoning, current_action, current_arg,
                        helper=teacher_model,
                    )
                else:
                    intervention = pf.intervene(
                        step_context_with_reasoning, current_action, current_arg,
                    )
                intervention.skill_id = skill_id

                # Rate-limit MODIFY_ACTION to prevent infinite loops
                if intervention.type == InterventionType.MODIFY_ACTION:
                    fires = pf_fire_counts.get(skill_id, 0)
                    if fires >= _MAX_MODIFY_FIRES:
                        logger.info(f"[PF:{skill_id}] Rate-limited (fired {fires} times)")
                        records.append(PFRecord(
                            skill_id=skill_id, step=step, activated=False,
                            reason=f"rate-limited ({fires} fires)",
                        ))
                        continue
                    pf_fire_counts[skill_id] = fires + 1

                records.append(PFRecord(
                    skill_id=skill_id,
                    step=step,
                    activated=True,
                    intervention_type=intervention.type.value,
                    reason=intervention.reason,
                    original_action=current_action,
                    original_arg=current_arg,
                    new_action_type=(
                        intervention.new_action_type
                        if intervention.type == InterventionType.MODIFY_ACTION else None
                    ),
                    new_action_arg=(
                        intervention.new_action_arg
                        if intervention.type == InterventionType.MODIFY_ACTION else None
                    ),
                    context_text=(
                        (intervention.context_text or "")[:200]
                        if intervention.type == InterventionType.INJECT_CONTEXT else None
                    ),
                ))

                if intervention.type == InterventionType.MODIFY_ACTION:
                    logger.info(
                        f"[PF:{skill_id}] Modifying action: "
                        f"{current_action}→{intervention.new_action_type} | {intervention.reason}"
                    )
                    if intervention.new_action_type:
                        current_action = intervention.new_action_type
                    if intervention.new_action_arg is not None:
                        current_arg = intervention.new_action_arg
                    # Carry feedback text along: when MODIFY_ACTION rewrites
                    # FINAL → RETRY (or similar), runners pull this from
                    # context_injections to render as the next observation.
                    if intervention.context_text:
                        context_injections.append(intervention.context_text)
                elif intervention.type == InterventionType.INJECT_CONTEXT:
                    logger.info(f"[PF:{skill_id}] Injecting context | {intervention.reason}")
                    context_injections.append(intervention.context_text)
            else:
                records.append(PFRecord(
                    skill_id=skill_id, step=step, activated=False,
                ))
        except Exception as e:
            logger.warning(f"[PF:{skill_id}] Error: {e}")
            records.append(PFRecord(
                skill_id=skill_id, step=step, activated=False,
                reason=f"error: {e}",
            ))

    return current_action, current_arg, records, context_injections


# ============================================================================
# Implementations (16 skills)
# ============================================================================

@register_pf("insufficient_exploration")
class InsufficientExplorationPF(ProgramFunction):
    """Block premature FINAL when agent hasn't explored enough."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        # Don't block near budget exhaustion
        if step >= max_steps - 2:
            return False
        search_count = ctx.get("search_count", 0)
        read_count = ctx.get("read_count", 0)
        # Block if no exploration at all
        if search_count == 0 and read_count == 0:
            return True
        # Block if searched but never read (and not near end)
        if search_count > 0 and read_count == 0 and step < max_steps - 3:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        search_count = ctx.get("search_count", 0)
        if search_count == 0:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="No search performed yet; forcing search",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="READ",
            new_action_arg="doc_0",
            reason="Searched but never read; forcing read",
        )


@register_pf("retrieval_failure")
class RetrievalFailurePF(ProgramFunction):
    """Reformulate overly long search queries using PF helper or smart truncation.

    When PF helper is available: asks PF helper to extract a concise search query
    from the model's verbose output. Falls back to smart keyword extraction.
    """
    needs_helper = True

    # Rate limit: max 3 fires per episode to avoid over-intervention
    _MAX_FIRES = 3

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        words = arg.split()
        # Only activate for genuinely long queries (>15 words)
        if len(words) > 15:
            fires = ctx.get("_retrieval_failure_fires", 0)
            if fires >= self._MAX_FIRES:
                return False
            return True
        return False

    def intervene(self, ctx, action_type, arg, helper=None):
        words = arg.split()
        question = ctx.get("question", "")

        # Track fires
        ctx["_retrieval_failure_fires"] = ctx.get("_retrieval_failure_fires", 0) + 1

        # Helper-backed: ask PF helper to extract a good search query
        if helper is not None:
            try:
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A search agent is trying to answer this question:\n"
                        f"Question: {question}\n\n"
                        f"The agent generated this overly long search query:\n"
                        f"\"{arg[:500]}\"\n\n"
                        f"Extract a concise, effective search query (5-12 words) that "
                        f"captures the key search intent. Reply with ONLY the query, "
                        f"no explanation."
                    )}],
                    max_tokens=50,
                    temperature=0.0,
                )
                if result and result.strip():
                    clean = result.strip().strip('"').strip("'").strip()
                    # Accept any reasonable length (1-20 words)
                    if 1 < len(clean.split()) <= 20:
                        logger.info(
                            f"[PF:retrieval_failure] PF helper reformulated: "
                            f"'{arg[:50]}...' → '{clean}'"
                        )
                        return Intervention(
                            type=InterventionType.MODIFY_ACTION,
                            new_action_type="SEARCH",
                            new_action_arg=clean,
                            reason=f"PF helper reformulated query ({len(words)} words → {len(clean.split())})",
                        )
                    else:
                        logger.warning(
                            f"[PF:retrieval_failure] PF helper returned bad length "
                            f"({len(clean.split())} words): '{clean[:80]}'"
                        )
                else:
                    logger.warning(f"[PF:retrieval_failure] PF helper returned empty result")
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:retrieval_failure] PF helper call failed: {e}")

        # Smart fallback: extract key noun phrases rather than blind truncation
        # Try to find the core question keywords
        shortened = self._smart_shorten(arg, question)
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=shortened,
            reason=f"Smart-shortened query ({len(words)} words → {len(shortened.split())})",
        )

    @staticmethod
    def _smart_shorten(query: str, question: str) -> str:
        """Extract the model's actual search intent from its verbose output.

        The Qwen model typically outputs: "actual search terms")\\n\\nreasoning..."
        The real search query is at the very beginning, often in quotes.
        We should extract THAT, not replace with the original question.
        """
        # Strategy 1: Extract quoted string at the start of the query
        # Pattern: "search terms") or "search terms"
        # This is the model's actual intended search query
        quote_match = re.match(r'^["\u201c](.+?)["\u201d]\s*\)?', query)
        if quote_match:
            extracted = quote_match.group(1).strip()
            if 1 < len(extracted.split()) <= 15:
                return extracted

        # Strategy 2: Extract content before first ")
        # Pattern: search terms")\n\nreasoning...
        paren_match = re.match(r'^(.+?)["\u201d]\s*\)', query)
        if paren_match:
            extracted = paren_match.group(1).strip().strip('"').strip()
            if 1 < len(extracted.split()) <= 15:
                return extracted

        # Strategy 3: Take the first line if it's short enough
        first_line = query.split('\n')[0].strip().strip('"').strip("'").rstrip(')')
        if 1 < len(first_line.split()) <= 12:
            return first_line

        # Strategy 4: Extract quoted strings anywhere in the query
        quoted = re.findall(r'"([^"]{3,60})"', query)
        if quoted:
            # Take the first quoted string that looks like a search query
            for q in quoted:
                q = q.strip()
                if 1 < len(q.split()) <= 12:
                    return q

        # Strategy 5: Fall back to original question only as last resort
        if question:
            q_words = question.split()
            if len(q_words) <= 12:
                return question
            return " ".join(q_words[:10])

        # Strategy 6: first 10 words
        return " ".join(query.split()[:10])


@register_pf("hallucination")
class HallucinationPF(ProgramFunction):
    """Block FINAL if answer makes claims without any read content."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        if step >= max_steps - 1:
            return False
        all_read = ctx.get("all_read_contents", "")
        # No documents read at all
        if not all_read or len(all_read.strip()) < 50:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        search_count = ctx.get("search_count", 0)
        if search_count == 0:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="Answering without any source; forcing search",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="READ",
            new_action_arg="doc_0",
            reason="Answering without reading any doc; forcing read",
        )


@register_pf("adversarial_distraction")
class AdversarialDistractionPF(ProgramFunction):
    """Inject warning when search results show genuinely conflicting information.

    Only activates for questions with multi-hop complexity (2+ hop signals)
    to avoid injecting unnecessary warnings on simple factoid questions where
    "however"/"actually" are common in natural text without real conflict.
    """

    _CONFLICT_WORDS = {
        "however", "contrary", "incorrect", "not true",
        "actually", "in fact", "disputed", "false", "disagree",
        "contradicts", "conflicting",
    }

    # Rate limit: max 1 fire per episode to avoid warning fatigue
    _MAX_FIRES = 1

    def should_activate(self, ctx, action_type, arg):
        # Only after SEARCH results arrive
        if action_type != "SEARCH":
            return False
        results_text = ctx.get("last_search_results_text", "")
        if not results_text:
            return False

        # Rate limit
        fires = ctx.get("_adversarial_distraction_fires", 0)
        if fires >= self._MAX_FIRES:
            return False

        # Skip for simple questions — conflict words appear naturally in text
        # without indicating real adversarial distraction
        question = ctx.get("question", "")
        q_lower = question.lower()
        hop_signals = (
            question.count("'s")
            + sum(1 for w in ["whose", "which", "that"] if f" {w} " in f" {q_lower} ")
            + q_lower.count(" of the ")
        )
        if hop_signals < 1 and len(question.split()) < 15:
            return False  # Simple question, skip

        text_lower = results_text.lower()
        conflict_count = sum(1 for w in self._CONFLICT_WORDS if w in text_lower)
        # Require higher threshold (3+) to reduce false positives
        return conflict_count >= 3

    def intervene(self, ctx, action_type, arg, helper=None):
        ctx["_adversarial_distraction_fires"] = ctx.get("_adversarial_distraction_fires", 0) + 1
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[Note: Conflicting Sources] Search results contain "
                "contradictory claims. Cross-reference before answering."
            ),
            reason="Detected conflicting information in search results",
        )


@register_pf("temporal_confusion")
class TemporalConfusionPF(ProgramFunction):
    """On FINAL, warn if years in the answer are not found in read docs.

    Only injects a warning — never forces SEARCH, since the model may
    have correctly inferred or computed a year not literally in the text.
    """

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        answer_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', arg))
        if not answer_years:
            return False
        doc_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', all_read))
        unsupported = answer_years - doc_years
        return len(unsupported) > 0

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        answer_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', arg))
        doc_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', all_read))
        unsupported = answer_years - doc_years
        # Only inject a gentle warning — do NOT force SEARCH
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[Note] Year(s) {unsupported} in your answer are not "
                f"directly found in the documents you read. Verify if correct."
            ),
            reason=f"Unsupported year(s) {unsupported} in answer",
        )


@register_pf("numerical_reasoning_error")
class NumericalReasoningPF(ProgramFunction):
    """On FINAL, check if numbers in the answer are supported by read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        answer_nums = set(re.findall(r'\b(\d{2,})\b', arg))
        # Exclude years
        year_re = re.compile(r'^(1[0-9]{3}|20[0-9]{2})$')
        answer_nums = {n for n in answer_nums if not year_re.match(n)}
        if not answer_nums:
            return False
        doc_nums = set(re.findall(r'\b(\d{2,})\b', all_read))
        unsupported = answer_nums - doc_nums
        return len(unsupported) > 0

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        answer_nums = set(re.findall(r'\b(\d{2,})\b', arg))
        year_re = re.compile(r'^(1[0-9]{3}|20[0-9]{2})$')
        answer_nums = {n for n in answer_nums if not year_re.match(n)}
        doc_nums = set(re.findall(r'\b(\d{2,})\b', all_read))
        unsupported = answer_nums - doc_nums
        # Only inject a gentle warning — do NOT force SEARCH
        # The model may have correctly computed/derived the number
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[Note] Number(s) {unsupported} in your answer are not "
                f"directly found in the documents. Verify if correct."
            ),
            reason=f"Unsupported number(s) {unsupported} in answer",
        )


@register_pf("negation_oversight")
class NegationOversightPF(ProgramFunction):
    """Inject reminder when question has negation but reasoning doesn't address it."""

    _NEGATION_WORDS = {"not", "never", "none", "neither", "except", "without",
                       "other than", "besides", "excluding"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        if not any(neg in q_lower for neg in self._NEGATION_WORDS):
            return False
        thought = ctx.get("thought", "")
        t_lower = thought.lower()
        return not any(neg in t_lower for neg in self._NEGATION_WORDS)

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[NEGATION REMINDER] The question contains a negation "
                "(not/never/except/...) but your reasoning doesn't address it. "
                "Re-read the question carefully."
            ),
            reason="Negation in question not reflected in reasoning",
        )


@register_pf("citation_mismatch")
class CitationMismatchPF(ProgramFunction):
    """On FINAL, check if the CORE entity in the answer exists in read documents.

    Only triggers when the answer contains a proper noun that is clearly the
    main answer AND that name is nowhere in any read document. Avoids false
    positives from peripheral mentions.
    """

    _STARTERS = {"The", "This", "That", "These", "Those", "Some", "Each",
                  "Every", "Many", "Most", "Both", "All", "Any", "In", "On",
                  "At", "By", "For", "To", "It", "He", "She", "We", "They",
                  "A", "An", "As", "So", "Or", "If", "My", "His", "Her"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read or len(all_read) < 100:
            return False
        # Only check short answers (likely a name/entity answer)
        if len(arg.split()) > 10:
            return False
        # Extract proper nouns from answer
        entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', arg)
        if not entities:
            return False
        cleaned = []
        for phrase in entities:
            words = phrase.split()
            while words and words[0] in self._STARTERS:
                words = words[1:]
            if len(words) >= 2:
                cleaned.append(" ".join(words))
        if not cleaned:
            return False
        all_read_lower = all_read.lower()
        # Also check in search results text
        search_text = ctx.get("last_search_results_text", "").lower()
        combined = all_read_lower + " " + search_text
        unsupported = [e for e in set(cleaned) if e.lower() not in combined]
        # Only trigger if the MAJORITY of entities are unsupported
        return len(unsupported) > 0 and len(unsupported) >= len(set(cleaned))

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        search_text = ctx.get("last_search_results_text", "").lower()
        combined = all_read.lower() + " " + search_text
        entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', arg)
        cleaned = []
        for phrase in entities:
            words = phrase.split()
            while words and words[0] in self._STARTERS:
                words = words[1:]
            if len(words) >= 2:
                cleaned.append(" ".join(words))
        unsupported = [e for e in set(cleaned) if e.lower() not in combined]

        # Only inject a warning — do NOT force SEARCH
        # The entity may be a valid inference or from search snippets
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[Note] Entity '{unsupported[0]}' in your answer was "
                f"not found in the documents you read. Verify if correct."
            ),
            reason=f"Unsupported entity '{unsupported[0]}' in answer",
        )


@register_pf("outdated_information")
class OutdatedInformationPF(ProgramFunction):
    """Warn when all source documents are older than 5 years."""

    _RECENCY_KEYWORDS = {"current", "currently", "now", "today", "latest", "recent",
                         "present", "2023", "2024", "2025", "2026"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        # Only trigger if the question asks about current/recent info
        question = ctx.get("question", "").lower()
        if not any(kw in question for kw in self._RECENCY_KEYWORDS):
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-9]\d)\b', all_read)]
        if not doc_years:
            return False
        return max(doc_years) < 2021

    def intervene(self, ctx, action_type, arg):
        all_read = ctx.get("all_read_contents", "")
        doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-9]\d)\b', all_read)]
        max_year = max(doc_years)
        # Only inject a warning — do NOT force SEARCH
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[Note] Your newest source is from {max_year}. "
                f"The question asks about current info — consider searching for more recent data."
            ),
            reason=f"All sources older than 2021 (newest: {max_year})",
        )


@register_pf("format_extraction_error")
class FormatExtractionPF(ProgramFunction):
    """Clean up FINAL answer formatting issues.

    When PF helper is available, uses it to extract the precise answer
    from messy output. Falls back to regex-based cleanup otherwise.
    """
    needs_helper = True  # Use PF helper for precise answer extraction

    # Patterns to strip from the answer (prefix patterns)
    _PREFIX_PATTERNS = [
        r'^The answer is:?\s*',
        r'^Based on [\w\s]+,\s*',
        r'^Based on my research,?\s*',
        r'^After reviewing[\w\s]*,\s*',
        r'^From the[\w\s]+,\s*',
        r'^According to [\w\s]+,\s*',
        r'^In summary,?\s*',
    ]
    # Trailing reference/note patterns
    _SUFFIX_PATTERNS = [
        r'\s*\[\d+\]\s*$',
        r'\s*\(note:.*?\)\s*$',
        r'\s*\[note:.*?\]\s*$',
        r'\s*\(Source:.*?\)\s*$',
    ]
    # Markdown formatting patterns
    _MARKDOWN_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
    _MARKDOWN_HEADER_RE = re.compile(r'^#+\s*')
    # Quoted answer extraction
    _QUOTED_ANSWER_RE = re.compile(r'^["\'](.+?)["\'](.*)$', re.DOTALL)

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        if not arg:
            return False
        for pattern in self._PREFIX_PATTERNS:
            if re.search(pattern, arg, re.IGNORECASE):
                return True
        for pattern in self._SUFFIX_PATTERNS:
            if re.search(pattern, arg, re.IGNORECASE):
                return True
        # Check for markdown formatting artifacts
        if self._MARKDOWN_BOLD_RE.search(arg):
            return True
        if self._MARKDOWN_HEADER_RE.search(arg):
            return True
        # Check for quoted answer with trailing meta-text
        quote_match = self._QUOTED_ANSWER_RE.match(arg)
        if quote_match:
            rest = quote_match.group(2).strip()
            if rest and re.match(r'^[\s,.]*(which|note|based|according|source|however)', rest, re.IGNORECASE):
                return True
        return False

    def _clean_regex(self, arg: str) -> str:
        """Regex-based answer cleanup (code-only fallback)."""
        cleaned = arg
        for pattern in self._PREFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        for pattern in self._SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = self._MARKDOWN_BOLD_RE.sub(r'\1', cleaned)
        cleaned = self._MARKDOWN_HEADER_RE.sub('', cleaned).strip()
        quote_match = self._QUOTED_ANSWER_RE.match(cleaned)
        if quote_match:
            quoted_part = quote_match.group(1).strip()
            rest = quote_match.group(2).strip()
            if rest and re.match(r'^[\s,.]*(which|note|based|according|source|however)', rest, re.IGNORECASE):
                cleaned = quoted_part
        return cleaned

    def intervene(self, ctx, action_type, arg, helper=None):
        # Try PF helper-based extraction first (more accurate for complex cases)
        if helper is not None and len(arg) > 30:
            try:
                question = ctx.get("question", "")
                extracted = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"Extract the precise, concise answer from this raw output. "
                        f"Return ONLY the answer entity/value, nothing else.\n\n"
                        f"Question: {question}\nRaw answer: {arg}"
                    )}],
                    max_tokens=100,
                    temperature=0.0,
                )
                if extracted and extracted.strip() and len(extracted.strip()) < len(arg):
                    cleaned = extracted.strip()
                    # Sanity: PF helper answer should be shorter and non-empty
                    if 0 < len(cleaned) < len(arg) * 0.9:
                        logger.info(f"[PF:format_extraction_error] PF helper extracted: {cleaned[:80]}")
                        return Intervention(
                            type=InterventionType.MODIFY_ACTION,
                            new_action_type="FINAL",
                            new_action_arg=cleaned,
                            reason="Teacher model extracted clean answer",
                        )
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:format_extraction_error] PF helper call failed: {e}")

        # Fallback: regex-based cleanup
        cleaned = self._clean_regex(arg)
        if cleaned and cleaned != arg:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="FINAL",
                new_action_arg=cleaned,
                reason="Cleaned formatting artifacts from answer",
            )
        return Intervention(type=InterventionType.NOOP, reason="No cleanup needed")


@register_pf("wrong_entity_confusion")
class WrongEntityConfusionPF(ProgramFunction):
    """After search, inject warning if results contain similarly-named entities."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        results_text = ctx.get("last_search_results_text", "")
        if not results_text:
            return False
        # Rate limit: at most once per episode (use ctx, not class attribute)
        if ctx.get("_wrong_entity_fired", False):
            return False
        # Only activate if the question mentions a named entity (multi-word proper noun)
        question = ctx.get("question", "")
        q_names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        if not q_names:
            return False
        # Look for multiple names that share a word with the question entity
        names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', results_text)
        if len(names) < 4:
            return False
        # Check for similar names (shared word with question entity)
        q_words = set()
        for qn in q_names:
            q_words.update(w.lower() for w in qn.split())
        similar_count = 0
        seen_names = set()
        for name in names:
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)
            name_words = set(name_lower.split())
            if name_words & q_words and name_words != q_words:
                similar_count += 1
        return similar_count >= 2

    def intervene(self, ctx, action_type, arg):
        ctx["_wrong_entity_fired"] = True
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[ENTITY WARNING] Search results contain similarly-named entities. "
                "READ documents carefully to identify the correct one. "
                "Pay attention to disambiguating details (dates, locations, roles)."
            ),
            reason="Similar entity names detected in search results",
        )


@register_pf("reading_comprehension_error")
class ReadingComprehensionPF(ProgramFunction):
    """After READ, inject a reminder to extract carefully if content is dense."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Only fire once (first READ with dense content)
        read_count = ctx.get("read_count", 0)
        if read_count > 1:
            return False
        # Dense content: VERY high entity/number density
        recent = all_read[-3000:]
        nums = len(re.findall(r'\b\d{2,}\b', recent))
        names = len(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', recent))
        return nums >= 10 or names >= 15

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[READ CAREFULLY] This document contains many entities/numbers. "
                "Extract the specific information relevant to the question. "
                "Don't confuse similar entities or misread numbers."
            ),
            reason="Dense document with many entities/numbers",
        )


@register_pf("multi_hop_reasoning_failure")
class MultiHopReasoningPF(ProgramFunction):
    """On FINAL, check if a multi-hop question was answered with single source."""

    _MULTI_HOP_INDICATORS = [
        "who", "what", "where", "when",  # Basic question words
    ]

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        # Multi-hop: question has multiple clauses or relative pronouns
        q_lower = question.lower()
        has_relative = any(w in q_lower for w in ["which", "whose", "that", "who"])
        # Check for compound questions
        has_compound = " and " in q_lower or "?" in question[:-1]
        if not (has_relative or has_compound):
            return False
        # No documents read at all for a multi-hop question
        read_count = ctx.get("read_count", 0)
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        return read_count == 0 and step < max_steps - 2

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=ctx.get("question", arg),
            reason="Multi-hop question answered with <=1 doc read; forcing more exploration",
        )


@register_pf("answer_completeness")
class AnswerCompletenessPF(ProgramFunction):
    """On FINAL, check if multi-part questions have all parts addressed."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Multi-part indicators
        has_and = " and " in q_lower
        has_multi_q = q_lower.count("?") > 1
        has_list = any(w in q_lower for w in ["both", "each", "all of", "list"])
        if not (has_and or has_multi_q or has_list):
            return False
        # Answer is suspiciously short for a multi-part question
        if len(arg.split()) < 3:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[COMPLETENESS WARNING] The question appears to have multiple parts, "
                "but your answer may be incomplete. Make sure you address all parts."
            ),
            reason="Multi-part question with short answer",
        )


@register_pf("reasoning_error")
class ReasoningErrorPF(ProgramFunction):
    """On FINAL, check for logical inconsistencies in reasoning.

    Only activates when the reasoning shows STRONG contradiction signals
    (multiple conflicting claims), not just "but/however" which are common
    in normal reasoning. Uses PF helper to verify before intervening.

    When PF helper available: asks PF helper to judge if there's a real flaw
    and whether the answer is correct. Only intervenes if PF helper finds
    a genuine error.
    Code-only fallback: only fires on very strong contradiction patterns.
    """
    needs_helper = True

    # Rate limit: max 1 fire per episode
    _MAX_FIRES = 1

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        thought = ctx.get("thought", "")
        if not thought or len(thought) < 150:
            return False

        # Rate limit
        if ctx.get("_reasoning_error_fired", False):
            return False

        t_lower = thought.lower()

        # Require STRONG contradiction: "but actually" or "however ... not"
        # Simple "but" or "however" alone is normal reasoning
        strong_patterns = [
            "but actually" in t_lower,
            "however, " in t_lower and " not " in t_lower.split("however")[1][:100] if "however" in t_lower else False,
            "i was wrong" in t_lower,
            "correction:" in t_lower,
            "wait, " in t_lower and ("no" in t_lower.split("wait")[1][:50] if "wait" in t_lower else False),
        ]
        if sum(bool(p) for p in strong_patterns) < 1:
            return False

        # Skip for simple questions — "but/however" is common in normal text
        question = ctx.get("question", "")
        if len(question.split()) < 12 and question.count("'s") == 0:
            return False

        return True

    def intervene(self, ctx, action_type, arg, helper=None):
        ctx["_reasoning_error_fired"] = True
        thought = ctx.get("thought", "")
        question = ctx.get("question", "")

        if helper is not None:
            try:
                all_read = ctx.get("all_read_contents", "")
                evidence = all_read[-1500:] if all_read else "(no documents)"
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A web search agent is answering a question. Its reasoning "
                        f"may contain contradictions. Check if the proposed answer is "
                        f"actually correct despite the messy reasoning.\n\n"
                        f"Question: {question}\n"
                        f"Agent's reasoning (last part): {thought[-400:]}\n"
                        f"Agent's proposed answer: {arg}\n"
                        f"Evidence from docs: ...{evidence}\n\n"
                        f"Is the answer '{arg}' correct based on the evidence? "
                        f"Reply with:\n"
                        f"CORRECT: if the answer is right (even if reasoning is messy)\n"
                        f"WRONG: <brief explanation of the flaw and what the answer should be>"
                    )}],
                    max_tokens=100,
                    temperature=0.0,
                )
                if result:
                    result_upper = result.strip().upper()
                    if result_upper.startswith("CORRECT"):
                        logger.info(f"[PF:reasoning_error] PF helper confirmed answer is correct")
                        return Intervention(type=InterventionType.NOOP,
                                            reason="Teacher confirmed answer correct despite messy reasoning")
                    elif "WRONG" in result_upper:
                        correction = result.strip()
                        logger.info(f"[PF:reasoning_error] PF helper found error: {correction[:100]}")
                        return Intervention(
                            type=InterventionType.INJECT_CONTEXT,
                            context_text=f"\n[REASONING CHECK] {correction}",
                            reason="Teacher identified genuine reasoning error",
                        )
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:reasoning_error] PF helper call failed: {e}")

        # Code-only fallback: only inject a mild warning, never block
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[Note] Double-check that your answer matches the strongest evidence."
            ),
            reason="Contradictory reasoning pattern detected (code-only)",
        )


@register_pf("language_barrier")
class LanguageBarrierPF(ProgramFunction):
    """After READ, inject warning if document contains significant non-ASCII text."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Check last chunk only
        recent = all_read[-3000:]
        non_ascii = sum(1 for c in recent if ord(c) > 127)
        return non_ascii > 50

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[LANGUAGE NOTE] This document contains non-English text. "
                "Be careful with transliteration and name variants. "
                "The same entity may appear in different scripts."
            ),
            reason="Significant non-ASCII content in document",
        )


# ============================================================================
# NEW SKILLS: Reasoning / Information Distillation (3)
# ============================================================================

@register_pf("decompose_complex_question")
class DecomposeComplexQuestionPF(ProgramFunction):
    """At episode start, detect multi-hop questions and inject decomposition hints."""

    _RELATIVE_PRONOUNS = {"which", "whose", "that", "who", "whom", "where"}

    def should_activate(self, ctx, action_type, arg):
        # Only inject on early SEARCH steps (step 0-1) for multi-hop questions
        if action_type != "SEARCH":
            return False
        step = ctx.get("step_count", 0)
        if step > 1:
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Detect multi-hop patterns
        # Pattern 1: possessive chain ("X's Y's Z")
        possessive_count = question.count("'s")
        if possessive_count >= 2:
            return True
        # Pattern 2: relative clause chains
        relative_count = sum(1 for w in self._RELATIVE_PRONOUNS if f" {w} " in f" {q_lower} ")
        if relative_count >= 1 and len(question.split()) > 10:
            return True
        # Pattern 3: "of the" chains indicating nested references
        of_the_count = q_lower.count(" of the ")
        if of_the_count >= 2:
            return True
        return False

    def intervene(self, ctx, action_type, arg):
        # Always inject guidance hint — do NOT modify the search query here
        # (retrieval_failure PF handles query shortening if needed)
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[DECOMPOSITION HINT] This is a multi-hop question. "
                "Search for each piece of information separately. "
                "Find intermediate entities first, then search for the final answer."
            ),
            reason="Multi-hop question detected; injecting decomposition guidance",
        )


@register_pf("evidence_synthesis")
class EvidenceSynthesisPF(ProgramFunction):
    """On FINAL, verify that key entities from question are found in read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        if not all_read or len(all_read) < 100:
            return False
        # Extract key entities from the question (multi-word proper nouns only)
        q_entities = [e for e in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
                      if len(e) > 3]
        if not q_entities:
            return False
        # Check if key question entities appear in read docs
        all_read_lower = all_read.lower()
        missing = [e for e in q_entities if e.lower() not in all_read_lower]
        # Only fire if ALL question entities are missing (very poor evidence)
        if len(missing) == len(q_entities) and len(missing) >= 2:
            step = ctx.get("step_count", 0)
            max_steps = ctx.get("max_steps", 10)
            return step < max_steps // 2
        return False

    def intervene(self, ctx, action_type, arg):
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        q_entities = [e for e in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
                      if len(e) > 3]
        all_read_lower = all_read.lower()
        missing = [e for e in q_entities if e.lower() not in all_read_lower]
        search_target = missing[0] if missing else ctx.get("question", arg)
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=search_target,
            reason=f"Key entity '{search_target}' from question not found in read docs",
        )


@register_pf("comparison_analyzer")
class ComparisonAnalyzerPF(ProgramFunction):
    """For comparison questions, ensure both entities are researched before FINAL."""

    _COMPARISON_WORDS = {
        "first", "earlier", "later", "before", "after", "older", "younger",
        "more", "less", "bigger", "smaller", "longer", "shorter",
        "which", "compare", "difference", "vs", "versus",
    }

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Check if this is a comparison question
        is_comparison = any(w in q_lower for w in self._COMPARISON_WORDS)
        if not is_comparison:
            return False
        # Check if question mentions 2+ named entities
        q_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        unique_entities = set(e.lower() for e in q_entities)
        if len(unique_entities) < 2:
            return False
        # Check if both entities appear in read content
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return True  # No docs read, definitely incomplete
        all_read_lower = all_read.lower()
        entities_found = sum(1 for e in unique_entities if e in all_read_lower)
        # Block if not all entities are found in read docs (only in first half)
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        return entities_found < len(unique_entities) and step < max_steps // 2

    def intervene(self, ctx, action_type, arg):
        question = ctx.get("question", "")
        all_read = ctx.get("all_read_contents", "")
        q_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        unique_entities = list(set(e for e in q_entities))
        all_read_lower = (all_read or "").lower()
        missing = [e for e in unique_entities if e.lower() not in all_read_lower]
        search_target = missing[0] if missing else unique_entities[0]
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=search_target,
            reason=f"Comparison question: entity '{search_target}' not yet researched",
        )


# ============================================================================
# NEW SKILLS: Search Optimization (3)
# ============================================================================

@register_pf("query_decomposition")
class QueryDecompositionPF(ProgramFunction):
    """Break overly complex search queries into focused sub-queries.

    NOTE: Defers to retrieval_failure PF for long queries (>15 words).
    This PF only handles the case of multiple question words in short queries.
    """

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        words = arg.split()
        # Do NOT activate for long queries — retrieval_failure handles those
        if len(words) > 15:
            return False
        # Only activate: multiple question words in 8-15 word queries
        q_words = {"who", "what", "where", "when", "which", "how"}
        q_count = sum(1 for w in words if w.lower() in q_words)
        return q_count >= 2 and len(words) >= 8

    def intervene(self, ctx, action_type, arg):
        # Multiple question words: use the original question if available
        question = ctx.get("question", "")
        if question and len(question.split()) <= 12:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=question,
                reason="Query has multiple questions; using original question",
            )
        words = arg.split()
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=" ".join(words[:8]),
            reason="Query has multiple questions; taking first part",
        )


@register_pf("iterative_refinement")
class IterativeRefinementPF(ProgramFunction):
    """Detect repeated similar searches and inject hint to change strategy."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "SEARCH":
            return False
        action_history = ctx.get("action_history", [])
        # Check if we've done 2+ searches with similar queries
        recent_searches = [
            a["arg"] for a in action_history[-4:]
            if a.get("action_type") == "SEARCH" and a.get("arg")
        ]
        if len(recent_searches) < 2:
            return False
        # Check similarity: overlapping words
        current_words = set(arg.lower().split())
        for prev in recent_searches:
            prev_words = set(prev.lower().split())
            overlap = len(current_words & prev_words)
            total = max(len(current_words | prev_words), 1)
            if overlap / total > 0.5:
                return True
        return False

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[SEARCH REFINEMENT] Your recent searches are very similar. "
                "Try a different approach: use specific names you've found, "
                "try synonyms, or search from a different angle."
            ),
            reason="Repeated similar searches detected",
        )


@register_pf("search_depth_controller")
class SearchDepthControllerPF(ProgramFunction):
    """Enforce minimum search/read depth based on question complexity."""

    _COMPLEX_INDICATORS = {"whose", "which", "that", "'s", "of the", "born", "died",
                           "director", "author", "spouse", "father", "mother"}

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        if step >= max_steps - 1:
            return False
        question = ctx.get("question", "")
        q_lower = question.lower()
        # Determine minimum depth
        complexity = sum(1 for ind in self._COMPLEX_INDICATORS if ind in q_lower)
        search_count = ctx.get("search_count", 0)
        read_count = ctx.get("read_count", 0)
        if complexity >= 3:
            # Complex: need at least 2 searches + 1 read
            return search_count < 2 or read_count < 1
        elif complexity >= 1:
            # Medium: need at least 1 search + 1 read
            return search_count < 1 or read_count < 1
        return False

    def intervene(self, ctx, action_type, arg):
        read_count = ctx.get("read_count", 0)
        search_count = ctx.get("search_count", 0)
        if search_count < 1:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="SEARCH",
                new_action_arg=ctx.get("question", arg),
                reason="Complex question needs more searches before answering",
            )
        if read_count < 1:
            return Intervention(
                type=InterventionType.MODIFY_ACTION,
                new_action_type="READ",
                new_action_arg="doc_0",
                reason="Complex question needs at least 1 document read",
            )
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg=ctx.get("question", arg),
            reason="Complex question needs more exploration",
        )


# ============================================================================
# NEW SKILLS: Adversarial Defense (2)
# ============================================================================

@register_pf("claim_triangulation")
class ClaimTriangulationPF(ProgramFunction):
    """On FINAL, prefer multi-source confirmation; warn if single-source answer."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "FINAL":
            return False
        read_count = ctx.get("read_count", 0)
        step = ctx.get("step_count", 0)
        max_steps = ctx.get("max_steps", 10)
        # Only trigger when we've read exactly 1 doc and have budget
        if read_count != 1 or step >= max_steps - 2:
            return False
        # Only for questions that might have adversarial content
        # or when the answer contains specific facts (names, numbers, dates)
        has_specific_facts = bool(
            re.findall(r'\b\d{2,}\b', arg) or
            re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', arg)
        )
        return has_specific_facts

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[VERIFICATION HINT] Your answer is based on a single source. "
                "Consider reading one more document to cross-verify the key facts "
                "before providing your final answer."
            ),
            reason="Single-source answer with specific facts; suggest cross-verification",
        )


@register_pf("misinformation_detector")
class MisinformationDetectorPF(ProgramFunction):
    """After READ, detect if newly read doc contradicts previously read docs."""

    def should_activate(self, ctx, action_type, arg):
        if action_type != "READ":
            return False
        read_count = ctx.get("read_count", 0)
        if read_count < 2:
            return False  # Need 2+ docs to compare
        all_read = ctx.get("all_read_contents", "")
        if not all_read:
            return False
        # Check for contradiction indicators in the combined read content
        text_lower = all_read.lower()
        # Look for conflicting years/numbers for the same entity
        years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', all_read)
        if len(set(years)) >= 3:
            # Many different years — potential for confusion
            return True
        # Look for explicit contradiction language
        contradiction_phrases = [
            "however", "in contrast", "on the other hand",
            "contradicts", "incorrect", "actually",
        ]
        contra_count = sum(1 for p in contradiction_phrases if p in text_lower)
        return contra_count >= 2

    def intervene(self, ctx, action_type, arg):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                "\n[CROSS-CHECK WARNING] Your read documents may contain conflicting "
                "information. Before answering, carefully compare the key facts across "
                "sources. Prefer information that is consistent across multiple documents."
            ),
            reason="Potential contradictions detected across read documents",
        )


@register_pf("constraint_search")
class ConstraintSearchPF(ProgramFunction):
    """Decompose multi-constraint questions (BrowseComp-style) into focused
    constraint-based search strategies.

    BrowseComp questions describe an entity through 5-10 independent constraints
    (date ranges, attributes, relationships). Standard search fails because
    cramming all constraints into one query returns nothing useful.

    Strategy:
    - Step 0-1: Detect multi-constraint question, extract the most searchable
      constraint, and modify the SEARCH query to focus on it.
    - Later steps: Inject context reminding agent to use "search → get candidates
      → verify constraints" strategy.
    - Never modify FINAL — only guide search behavior.
    """

    # Patterns that indicate constraint-heavy questions
    _DATE_RANGE_RE = re.compile(
        r'between\s+\d{4}\s+and\s+\d{4}|'
        r'\d{4}\s*[-–]\s*\d{4}|'
        r'\d{4}\s+(?:and|to)\s+\d{4}\s+inclusive',
        re.IGNORECASE,
    )
    # Constraint signal phrases
    _CONSTRAINT_PHRASES = [
        "born between", "born in", "formed between", "founded between",
        "published between", "released between", "played between",
        "joined", "scored", "attended", "located in", "based in",
        "starred", "directed by", "written by", "created by",
        "more than", "less than", "fewer than", "at least",
        "who was", "who is", "whose", "that was", "that is",
        "an actor", "a player", "a person", "a particular",
    ]

    def _count_constraints(self, question: str) -> int:
        """Estimate the number of independent constraints in a question."""
        q_lower = question.lower()
        count = 0
        # Date ranges
        count += len(self._DATE_RANGE_RE.findall(question))
        # Constraint phrases
        count += sum(1 for p in self._CONSTRAINT_PHRASES if p in q_lower)
        # Commas separating constraint clauses
        comma_count = question.count(",")
        if comma_count >= 3:
            count += comma_count - 2
        # Sentence count (each sentence often = one constraint)
        sentences = [s.strip() for s in re.split(r'[.!?]', question) if s.strip()]
        if len(sentences) >= 3:
            count += len(sentences) - 2
        return count

    # Words to strip from beginning of extracted queries
    _STRIP_PREFIXES = {"the", "a", "an", "of", "in", "for", "by", "on", "at",
                       "its", "their", "his", "her", "this", "that"}

    def _extract_searchable_constraint(self, question: str) -> Optional[str]:
        """Pick the most distinctive/searchable constraint from the question.

        Prefers constraints with named entities (places, events, titles) over
        generic date-range constraints. Returns a focused 4-8 word query.
        """
        sentences = [s.strip() for s in re.split(r'[.!?]', question) if len(s.strip()) > 10]

        best_sentence = None
        best_score = -1

        # Entity-type keywords that make a sentence highly searchable
        _ENTITY_KW = {"cup", "championship", "award", "academy", "series",
                      "university", "school", "college", "film", "movie", "manga",
                      "team", "league", "competition", "wembley", "olympics",
                      "nobel", "grammy", "oscar", "emmy", "pulitzer",
                      "institute", "museum", "cathedral", "festival"}

        for sent in sentences:
            score = 0
            s_lower = sent.lower()
            words = sent.split()

            # Proper nouns (capitalized words, skip sentence-initial)
            proper_nouns = [w.strip(",.;:()") for w in words[1:]
                          if w[0].isupper() and len(w) > 2]
            score += len(proper_nouns) * 3

            # Entity-type keyword boost
            for kw in _ENTITY_KW:
                if kw in s_lower:
                    score += 3

            # Penalize generic constraint-only sentences
            date_ranges = len(self._DATE_RANGE_RE.findall(sent))
            if date_ranges > 0 and len(proper_nouns) == 0:
                score -= 3

            # Penalize very short sentences (less informative)
            if len(words) < 5:
                score -= 1

            if score > best_score:
                best_score = score
                best_sentence = sent

        if not best_sentence or best_score <= 0:
            return None

        # Extract a focused 4-8 word query from the best sentence
        words = best_sentence.split()

        # Find key content words (proper nouns + entity keywords)
        key_indices = []
        for i, w in enumerate(words):
            w_clean = w.strip(",.;:()'\"")
            if not w_clean:
                continue
            if w_clean[0].isupper() and len(w_clean) > 2 and i > 0:
                key_indices.append(i)
            elif w_clean.lower() in _ENTITY_KW:
                key_indices.append(i)

        if key_indices:
            # Build query around key words with +-1 context
            start = max(0, min(key_indices) - 1)
            end = min(len(words), max(key_indices) + 2)
            query_words = words[start:end]
        else:
            # Fallback: first 6 content words
            query_words = words[:6]

        # Clean: strip leading articles/prepositions, limit to 8 words
        while query_words and query_words[0].lower().strip(",.") in self._STRIP_PREFIXES:
            query_words = query_words[1:]

        query = " ".join(query_words[:8]).strip(",.;: ")
        return query if len(query) > 5 else None

    def should_activate(self, ctx, action_type, arg):
        # Only on SEARCH actions
        if action_type != "SEARCH":
            return False

        question = ctx.get("question", "")
        constraint_count = self._count_constraints(question)

        # Need at least 4 constraints to be a BrowseComp-style question
        if constraint_count < 4:
            return False

        step = ctx.get("step_count", 0)

        # Step 0-1: Modify search query to focus on best constraint
        if step <= 1:
            # Check if query is too long (cramming all constraints)
            if len(arg.split()) > 8:
                return True

        # Steps 2-5: Inject strategy guidance if agent seems stuck
        if 2 <= step <= 5:
            action_history = ctx.get("action_history", [])
            search_count = sum(1 for a in action_history if a.get("action_type") == "SEARCH")
            read_count = ctx.get("read_count", 0)
            # Agent has searched multiple times but hasn't read — still fishing
            if search_count >= 2 and read_count == 0:
                return True

        return False

    def intervene(self, ctx, action_type, arg):
        step = ctx.get("step_count", 0)
        question = ctx.get("question", "")

        if step <= 1:
            # Modify first search to focus on best constraint
            focused_query = self._extract_searchable_constraint(question)
            if focused_query:
                return Intervention(
                    type=InterventionType.MODIFY_ACTION,
                    new_action_type="SEARCH",
                    new_action_arg=focused_query,
                    reason=f"Multi-constraint question ({self._count_constraints(question)} constraints); "
                           f"focusing search on most distinctive constraint",
                )

        # Inject strategy guidance
        constraint_count = self._count_constraints(question)
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[CONSTRAINT SEARCH STRATEGY] This question has ~{constraint_count} constraints. "
                "Do NOT try to search for all constraints at once. Instead:\n"
                "1. Search for the most unique/specific constraint (a named event, place, or title)\n"
                "2. READ a result to find candidate entities\n"
                "3. Verify candidates against the other constraints\n"
                "4. Narrow down until only one entity matches ALL constraints"
            ),
            reason=f"Multi-constraint question; guiding search strategy (step {step})",
        )


# ============================================================================
# Answer Confidence Guard — prevent over-refinement regressions
# ============================================================================

@register_pf("answer_confidence_guard")
class AnswerConfidenceGuardPF(ProgramFunction):
    """Track candidate answers across steps and warn against over-refinement.

    Prevents the common regression pattern where the agent correctly identifies
    an answer early on, but then second-guesses itself after additional (often
    irrelevant) search/read steps, changing to a worse answer.

    When PF helper available: asks PF helper to judge which answer is better
    supported by the evidence. May revert to the original answer.
    """
    needs_helper = True

    # Patterns that indicate the model has settled on a candidate answer
    _ANSWER_PATTERNS = [
        re.compile(r'the answer (?:is|should be|would be)\s+["\']?(.+?)["\']?\s*[.,;]', re.IGNORECASE),
        re.compile(r'["\']?(.+?)["\']?\s+is the answer', re.IGNORECASE),
        re.compile(r'so the answer should be\s+["\']?(.+?)["\']?\s*[.,;]', re.IGNORECASE),
        re.compile(r'therefore,?\s+["\']?(.+?)["\']?\s*[.,;$]', re.IGNORECASE),
        re.compile(r'I (?:can|will) conclude (?:that |the answer is )\s*["\']?(.+?)["\']?\s*[.,;$]', re.IGNORECASE),
    ]

    def _extract_candidate(self, thought: str) -> Optional[str]:
        """Try to extract a candidate answer from reasoning text."""
        for pattern in self._ANSWER_PATTERNS:
            m = pattern.search(thought)
            if m:
                candidate = m.group(1).strip()
                # Filter out very long or very short extractions
                if 1 < len(candidate) < 100:
                    return candidate
        return None

    def should_activate(self, ctx, action_type, arg):
        thought = ctx.get("thought", "")
        candidates = ctx.get("_candidate_answers", [])

        if action_type == "FINAL":
            # Activate if there's a previous candidate and the current answer differs
            if candidates and arg:
                last_candidate = candidates[-1]
                old_answer = last_candidate.get("answer", "")
                # Only warn if answers are meaningfully different
                if old_answer and old_answer.lower().strip() != arg.lower().strip():
                    # Check if the old answer was from a step with read evidence
                    if last_candidate.get("has_read", False):
                        return True
            return False

        # On non-FINAL actions: check if thought contains a confident answer statement
        if thought and action_type in ("SEARCH", "READ", "SUMMARY"):
            candidate = self._extract_candidate(thought)
            if candidate:
                # Store the candidate (side-effect in should_activate, but
                # this runs deterministically and is the simplest approach)
                has_read = ctx.get("has_read", False)
                step = ctx.get("step_count", 0)
                if "_candidate_answers" not in ctx:
                    ctx["_candidate_answers"] = []
                ctx["_candidate_answers"].append({
                    "answer": candidate,
                    "step": step,
                    "has_read": has_read,
                })
                return False  # Don't intervene on non-FINAL, just track

        return False

    def intervene(self, ctx, action_type, arg, helper=None):
        candidates = ctx.get("_candidate_answers", [])
        if not candidates:
            return Intervention(type=InterventionType.NOOP, reason="No prior candidates")

        last_candidate = candidates[-1]
        old_answer = last_candidate.get("answer", "")
        question = ctx.get("question", "")

        # Helper-backed: ask which answer is better supported
        if helper is not None:
            try:
                all_read = ctx.get("all_read_contents", "")
                # Truncate evidence to fit in context
                evidence_snippet = all_read[-2000:] if all_read else "(no documents read)"
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"A search agent is answering a question. It first concluded "
                        f"'{old_answer}' after reading sources, but now wants to change "
                        f"to '{arg}'. Which answer is better supported by the evidence?\n\n"
                        f"Question: {question}\n"
                        f"Evidence (recent): ...{evidence_snippet}\n\n"
                        f"Reply with ONLY one of:\n"
                        f"KEEP_OLD: {old_answer}\n"
                        f"USE_NEW: {arg}\n"
                    )}],
                    max_tokens=50,
                    temperature=0.0,
                )
                if result and "KEEP_OLD" in result.upper():
                    logger.info(
                        f"[PF:answer_confidence_guard] PF helper chose old answer: "
                        f"'{old_answer}' over '{arg}'"
                    )
                    return Intervention(
                        type=InterventionType.MODIFY_ACTION,
                        new_action_type="FINAL",
                        new_action_arg=old_answer,
                        reason=f"PF helper judged '{old_answer}' better supported than '{arg}'",
                    )
                else:
                    logger.info(f"[PF:answer_confidence_guard] PF helper accepted new answer: '{arg}'")
                    return Intervention(type=InterventionType.NOOP, reason="Teacher approved answer change")
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[PF:answer_confidence_guard] PF helper call failed: {e}")

        # Fallback: warning injection
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text=(
                f"\n[CONFIDENCE WARNING] You previously concluded '{old_answer}' "
                f"based on source evidence (step {last_candidate.get('step', '?')}). "
                f"Only change your answer if you have strong new evidence."
            ),
            reason=f"Answer changed from '{old_answer}' to '{arg}'; prior answer had read support",
        )


# ============================================================================
# Observation Transformers — post-tool programmatic observation modification
# ============================================================================
# Unlike PFs (pre-dispatch), these run AFTER tool execution and transform the
# observation text before the model sees it. They reduce noise instead of
# adding instructions, avoiding attention dilution.

class ObservationTransformer:
    """Base class for post-tool observation transformers."""
    skill_id: str = ""
    needs_helper: bool = False  # Override to True to receive PF helper

    def should_activate(self, action_type: str, step_context: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def transform(self, obs_text: str, step_context: Dict[str, Any], helper=None) -> str:
        """Transform observation text. Must return modified obs_text."""
        raise NotImplementedError


_OT_REGISTRY: Dict[str, ObservationTransformer] = {}


def register_ot(skill_id: str):
    """Decorator to register an observation transformer."""
    def decorator(cls):
        cls.skill_id = skill_id
        _OT_REGISTRY[skill_id] = cls()
        return cls
    return decorator


def get_all_observation_transformers() -> Dict[str, ObservationTransformer]:
    return dict(_OT_REGISTRY)


def execute_observation_transformers(
    active_pf_ids: List[str],
    obs_text: str,
    action_type: str,
    step_context: Dict[str, Any],
    teacher_model=None,
) -> str:
    """Execute active observation transformers on the observation text.

    Args:
        active_pf_ids: PF IDs selected for this episode (includes OT IDs).
        obs_text: Raw observation text from tool execution.
        action_type: "SEARCH", "READ", or "SUMMARY".
        step_context: Current step context dict.
        teacher_model: Optional PF helper for OTs with needs_helper=True.

    Returns:
        Transformed observation text.
    """
    for skill_id, ot in _OT_REGISTRY.items():
        if skill_id not in active_pf_ids:
            continue
        try:
            if ot.should_activate(action_type, step_context):
                if ot.needs_helper and teacher_model is not None:
                    obs_text = ot.transform(obs_text, step_context, helper=teacher_model)
                else:
                    obs_text = ot.transform(obs_text, step_context)
                logger.info(f"[OT:{skill_id}] Transformed observation")
        except Exception as e:
            logger.warning(f"[OT:{skill_id}] Error: {e}")
    return obs_text


# ---- Stop words shared by OTs ----
_OT_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "of", "in", "on", "at",
    "to", "for", "by", "with", "and", "or", "but", "not", "that", "this",
    "which", "who", "what", "where", "when", "how", "do", "does", "did",
    "have", "has", "had", "be", "been", "being", "it", "its", "he", "she",
    "they", "we", "you", "if", "so", "as", "from", "about", "into", "than",
    "me", "my", "your", "his", "her", "their", "our", "can", "will", "would",
})


def _extract_keywords(text: str) -> List[str]:
    """Extract content keywords from text (no stop words, len > 2)."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [w for w in words if w not in _OT_STOP_WORDS and len(w) > 2]


def _extract_entities(text: str) -> List[str]:
    """Extract multi-word proper nouns from text."""
    return re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)


@register_ot("search_result_reranker")
class SearchResultRerankerOT(ObservationTransformer):
    """Rerank search results by relevance to question keywords.

    Scores each search result by keyword overlap with the question,
    reorders so the most relevant result appears first, and marks it
    with a ★ prefix. Zero extra text added — pure data transformation.
    """

    def should_activate(self, action_type, step_context):
        return action_type == "SEARCH"

    def transform(self, obs_text, step_context):
        if "No results found" in obs_text:
            return obs_text

        question = step_context.get("question", "")
        q_keywords = _extract_keywords(question)
        q_entities = _extract_entities(question)
        if not q_keywords:
            return obs_text

        lines = obs_text.split("\n")
        header_lines = []
        result_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- [doc_") or stripped.startswith("★ [doc_"):
                result_lines.append(line)
            else:
                header_lines.append(line)

        if len(result_lines) < 2:
            return obs_text

        # Score each result line
        scored = []
        for line in result_lines:
            line_lower = line.lower()
            score = sum(1 for kw in q_keywords if kw in line_lower)
            # Bonus for multi-word entity matches
            for entity in q_entities:
                if entity.lower() in line_lower:
                    score += 3
            scored.append((score, line))

        scored.sort(reverse=True)

        # Mark top result with ★
        reranked = []
        for i, (score, line) in enumerate(scored):
            if i == 0 and score > 0:
                reranked.append(line.replace("- [", "★ [", 1))
            else:
                reranked.append(line)

        return "\n".join(header_lines) + "\n" + "\n".join(reranked)


@register_ot("relevant_content_extractor")
class RelevantContentExtractorOT(ObservationTransformer):
    """Extract question-relevant paragraphs from READ content.

    Instead of showing the raw first 3000 chars of a web page (which may
    include navigation, ads, irrelevant sections), this transformer extracts
    the most relevant content.

    When PF helper is available: asks GPT-4o to extract the key facts from
    the document that are relevant to the question. This is far more accurate
    than keyword matching for complex questions.

    Fallback: scores paragraphs by keyword overlap and keeps top ones.
    """
    needs_helper = True

    def should_activate(self, action_type, step_context):
        return action_type == "READ"

    def transform(self, obs_text, step_context, helper=None):
        question = step_context.get("question", "")

        # Only process long content
        if len(obs_text) < 1500:
            return obs_text

        prefix = "Observation: "
        has_prefix = obs_text.startswith(prefix)
        content = obs_text[len(prefix):] if has_prefix else obs_text

        # Helper-backed extraction
        if helper is not None and len(content) > 2000:
            try:
                # Truncate content for PF helper context window
                content_for_teacher = content[:4000]
                result = helper.generate(
                    messages=[{"role": "user", "content": (
                        f"Extract the key facts from this document that are relevant "
                        f"to answering the question. Return ONLY the relevant facts "
                        f"as concise bullet points. If no relevant info found, reply "
                        f"with 'No relevant information found.'\n\n"
                        f"Question: {question}\n\n"
                        f"Document:\n{content_for_teacher}"
                    )}],
                    max_tokens=400,
                    temperature=0.0,
                )
                if result and result.strip() and "no relevant" not in result.lower():
                    extracted = result.strip()
                    if len(extracted) > 50:
                        logger.info(f"[OT:relevant_content_extractor] PF helper extracted {len(extracted)} chars")
                        out = (prefix if has_prefix else "") + f"[Key facts for: {question[:60]}]\n{extracted}"
                        return out
            except Exception as e:
                if note_api_error(e): raise
                logger.warning(f"[OT:relevant_content_extractor] PF helper call failed: {e}")

        # Fallback: keyword-based extraction
        q_keywords = _extract_keywords(question)
        q_entities = _extract_entities(question)
        if not q_keywords:
            return obs_text

        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if len(p.strip()) >= 30]

        if len(paragraphs) <= 2:
            return obs_text

        # Score each paragraph
        scored = []
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for kw in q_keywords if kw in para_lower)
            for entity in q_entities:
                if entity.lower() in para_lower:
                    score += 5
            scored.append((score, para))

        scored.sort(reverse=True)

        # Select top-scoring paragraphs up to ~2500 chars
        selected = []
        total = 0
        for score, para in scored:
            if score == 0:
                break
            if total + len(para) > 2500 and selected:
                break
            selected.append(para)
            total += len(para)

        if not selected or total < 100:
            return obs_text

        return (prefix if has_prefix else "") + "[Key excerpts]\n" + "\n\n".join(selected)


# ============================================================================
# Register MATH-domain PFs (auto-registered via @register_pf decorators)
# ============================================================================

from . import math_program_functions  # noqa: F401, E402 — side-effect: registers math PFs
