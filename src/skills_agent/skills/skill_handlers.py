"""
Skill Handlers — code-level handlers for skills.

Two types:
1. Pure-code handlers (6): deterministic, zero latency, no external calls
2. LLM-assisted handlers (8): use a PF helper (GPT/Claude/etc.) for evaluation

All handlers follow the same interface:
    handler(context, teacher_model=None) -> Optional[str]

Returns:
    - None: no problem detected (or handler skipped)
    - str: intervention text to inject into the conversation

Handler registry maps handler_id -> (handler_fn, requires_api).
"""

import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Callable, Tuple, List

logger = logging.getLogger(__name__)

# Type alias for handler functions
HandlerFn = Callable[[Dict[str, Any], Any], Optional[str]]

# Registry: handler_id -> (handler_fn, requires_api)
_HANDLER_REGISTRY: Dict[str, Tuple[HandlerFn, bool]] = {}


# ============================================================================
# Handler activity record — captures every handler invocation for logging
# ============================================================================

@dataclass
class HandlerRecord:
    """Record of a single handler invocation."""
    handler_id: str
    skill_id: str = ""                   # Originating skill
    handler_type: str = ""               # "pure_code" | "llm_assisted"
    result: str = ""                     # "triggered" | "clean" | "skipped" | "error"
    intervention_text: Optional[str] = None  # Output text (if triggered)
    # PF helper fields (only for LLM-assisted)
    teacher_prompt: Optional[str] = None     # Full prompt sent to PF helper
    teacher_response: Optional[str] = None   # Raw response from PF helper
    teacher_model_name: Optional[str] = None # e.g. "claude-sonnet-4-20250514"
    latency_ms: float = 0.0                  # Wall-clock time in milliseconds
    error_message: Optional[str] = None      # Error message (if failed)

    def to_dict(self) -> dict:
        """Serialize to dict, omitting None fields for compactness."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != "" and v != 0.0}


def register_handler(handler_id: str, requires_api: bool = False):
    """Decorator to register a skill handler."""
    def decorator(fn: HandlerFn) -> HandlerFn:
        _HANDLER_REGISTRY[handler_id] = (fn, requires_api)
        return fn
    return decorator


def get_handler(handler_id: str) -> Optional[Tuple[HandlerFn, bool]]:
    """Look up a handler by ID. Returns (handler_fn, requires_api) or None."""
    return _HANDLER_REGISTRY.get(handler_id)


def get_all_handlers() -> Dict[str, Tuple[HandlerFn, bool]]:
    """Return the full handler registry."""
    return dict(_HANDLER_REGISTRY)


def execute_handler(
    handler_id: str,
    context: Dict[str, Any],
    teacher_model=None,
    skill_id: str = "",
) -> Optional[str]:
    """Execute a handler by ID.

    Args:
        handler_id: Registered handler ID.
        context: Step context dict with keys like question, action_arg,
                 all_read_contents, thought, etc.
        teacher_model: Optional APIModelWrapper for LLM-assisted handlers.
        skill_id: Originating skill ID (for record keeping).

    Returns:
        Intervention text or None.

    Side effect:
        Appends a HandlerRecord to context["_handler_records"] if that key
        exists (list). Callers can pre-create this list to collect records.
    """
    entry = _HANDLER_REGISTRY.get(handler_id)
    if entry is None:
        logger.debug(f"Handler '{handler_id}' not found in registry")
        return None

    handler_fn, requires_api = entry
    handler_type = "llm_assisted" if requires_api else "pure_code"

    if requires_api and teacher_model is None:
        logger.debug(f"Handler '{handler_id}' requires API but no teacher_model provided, skipping")
        _append_record(context, HandlerRecord(
            handler_id=handler_id, skill_id=skill_id,
            handler_type=handler_type, result="skipped",
        ))
        return None

    t0 = time.monotonic()
    try:
        result_text = handler_fn(context, teacher_model)
        latency = (time.monotonic() - t0) * 1000

        record = HandlerRecord(
            handler_id=handler_id,
            skill_id=skill_id,
            handler_type=handler_type,
            result="triggered" if result_text else "clean",
            intervention_text=result_text,
            latency_ms=round(latency, 1),
        )

        # Capture PF helper call details if LLM-assisted
        if requires_api:
            record.teacher_model_name = getattr(teacher_model, "model_name", None)
            # _last_teacher_call is set by _call_teacher
            last_call = context.pop("_last_teacher_call", None)
            if last_call:
                record.teacher_prompt = last_call.get("prompt")
                record.teacher_response = last_call.get("response")

        _append_record(context, record)
        return result_text

    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning(f"Handler '{handler_id}' failed: {e}")
        _append_record(context, HandlerRecord(
            handler_id=handler_id, skill_id=skill_id,
            handler_type=handler_type, result="error",
            error_message=str(e), latency_ms=round(latency, 1),
        ))
        return None


def _append_record(context: Dict[str, Any], record: HandlerRecord):
    """Append a record to context['_handler_records'] if the list exists."""
    records = context.get("_handler_records")
    if records is not None and isinstance(records, list):
        records.append(record)


# ============================================================================
# Helper: truncate evidence for LLM prompts
# ============================================================================

def _truncate(text: str, max_chars: int = 3000) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _call_teacher(
    teacher_model, messages: list, context: Dict[str, Any] = None, max_tokens: int = 200,
) -> Optional[str]:
    """Call PF helper with error handling and activity capture.

    Stores the prompt/response in context["_last_teacher_call"] so that
    execute_handler can attach it to the HandlerRecord.

    Returns response text or None.
    """
    if teacher_model is None:
        return None
    from .quota import guard, note_api_error
    guard()  # raise immediately if a prior call tripped the breaker
    try:
        # Build a compact prompt string for logging (user message only)
        prompt_text = messages[-1]["content"] if messages else ""
        response = teacher_model.generate_from_messages(
            messages, max_tokens=max_tokens, temperature=0.0
        )
        # Store for record capture
        if context is not None:
            context["_last_teacher_call"] = {
                "prompt": prompt_text,
                "response": response,
            }
        return response
    except Exception as e:
        if note_api_error(e):
            # Quota / auth error — don't mask with warning; abort the run.
            raise
        logger.warning(f"PF helper model call failed: {e}")
        return None


# ============================================================================
# Pure-Code Handlers (6)
# ============================================================================

@register_handler("verify_temporal_claims", requires_api=False)
def verify_temporal_claims(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Cross-check years in the answer against source documents."""
    answer = context.get("action_arg", "") or ""
    all_read = context.get("all_read_contents", "") or ""

    if not answer or not all_read:
        return None

    answer_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', answer))
    if not answer_years:
        return None

    doc_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', all_read))
    unsupported = answer_years - doc_years

    if not unsupported:
        return None

    return (
        f"[TEMPORAL CHECK] Year(s) {unsupported} in your answer are NOT found in any "
        f"document you read. Please verify these dates or search for confirmation."
    )


@register_handler("verify_numerical_claims", requires_api=False)
def verify_numerical_claims(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Cross-check numbers in the answer against source documents."""
    answer = context.get("action_arg", "") or ""
    all_read = context.get("all_read_contents", "") or ""

    if not answer or not all_read:
        return None

    answer_nums = set(re.findall(r'\b(\d{2,})\b', answer))
    if not answer_nums:
        return None

    # Exclude years from numeric check (handled by temporal handler)
    year_pattern = re.compile(r'^(1[0-9]{3}|20[0-9]{2})$')
    answer_nums = {n for n in answer_nums if not year_pattern.match(n)}
    if not answer_nums:
        return None

    doc_nums = set(re.findall(r'\b(\d{2,})\b', all_read))
    unsupported = answer_nums - doc_nums

    if not unsupported:
        return None

    return (
        f"[NUMERICAL CHECK] Number(s) {unsupported} in your answer are NOT found in any "
        f"document you read. Please verify or search for the correct values."
    )


@register_handler("block_premature_final", requires_api=False)
def block_premature_final(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Block FINAL only when the agent has done zero exploration.

    Triggers when search_count == 0 AND read_count == 0 (completely unexplored).
    Budget-aware: never blocks when step_count >= max_steps - 2.
    """
    step_count = context.get("step_count", 0)
    max_steps = context.get("max_steps", 10)

    # Never block near budget exhaustion
    if step_count >= max_steps - 2:
        return None

    search_count = context.get("search_count", 0)
    read_count = context.get("read_count", 0)

    # Only block if absolutely no exploration was done
    if search_count > 0 or read_count > 0:
        return None

    return (
        "[PREMATURE ANSWER BLOCKED] You have not searched or read any documents yet. "
        "Please SEARCH for relevant information before giving your final answer."
    )


@register_handler("check_source_freshness", requires_api=False)
def check_source_freshness(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Flag if the newest source year is more than 5 years old."""
    all_read = context.get("all_read_contents", "") or ""
    if not all_read:
        return None

    doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-9]\d)\b', all_read)]
    if not doc_years:
        return None

    max_year = max(doc_years)
    # Use 2026 as reference (could also get dynamically)
    if max_year >= 2021:
        return None

    return (
        f"[FRESHNESS WARNING] Your newest source is from {max_year}, which is over 5 years old. "
        f"Consider searching for more recent information."
    )


@register_handler("verify_citations", requires_api=False)
def verify_citations(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Cross-check proper nouns in the answer against source documents."""
    answer = context.get("action_arg", "") or ""
    all_read = context.get("all_read_contents", "") or ""

    if not answer or not all_read:
        return None

    # Extract proper nouns (capitalized multi-word phrases)
    answer_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', answer)
    if not answer_entities:
        return None

    # Strip common sentence starters
    starters = {"The", "This", "That", "These", "Those", "Some", "Each",
                 "Every", "Many", "Most", "Both", "All", "Any"}
    cleaned = []
    for phrase in answer_entities:
        words = phrase.split()
        while words and words[0] in starters:
            words = words[1:]
        if len(words) >= 2:
            cleaned.append(" ".join(words))

    if not cleaned:
        return None

    all_read_lower = all_read.lower()
    unsupported = [e for e in set(cleaned) if e.lower() not in all_read_lower]

    if not unsupported:
        return None

    return (
        f"[CITATION CHECK] Entity/entities {unsupported[:3]} in your answer are NOT found "
        f"in any document you read. Please verify these names."
    )


@register_handler("check_negation", requires_api=False)
def check_negation(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Verify that negation in the question is reflected in reasoning."""
    question = context.get("question", "") or ""
    thought = context.get("thought", "") or ""

    if not question or not thought:
        return None

    negation_words = {"not", "never", "none", "neither", "except", "without",
                      "other than", "besides", "excluding"}
    q_lower = question.lower()
    has_negation = any(neg in q_lower for neg in negation_words)

    if not has_negation:
        return None

    thought_lower = thought.lower()
    negation_in_reasoning = any(neg in thought_lower for neg in negation_words)

    if negation_in_reasoning:
        return None

    return (
        "[NEGATION CHECK] The question contains a negation (not/never/except/...) "
        "but your reasoning does not address it. Please re-read the question carefully."
    )


# ============================================================================
# LLM-Assisted Handlers (8)
# Removed: resolve_question_ambiguity, verify_source_authority,
#          verify_general_quality, check_scope_alignment, check_answer_completeness
# Added:   verify_answer_relevance (merges scope + completeness)
# ============================================================================

@register_handler("verify_reasoning_chain", requires_api=True)
def verify_reasoning_chain(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Verify multi-hop reasoning logic using PF helper."""
    question = context.get("question", "") or ""
    thought = context.get("thought", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    if not question or not thought:
        return None

    messages = [
        {"role": "system", "content": (
            "You check reasoning chains for CLEAR logical errors that would change the answer. "
            "Minor omissions, stylistic issues, or incomplete explanations are NOT errors. "
            "Only flag if the conclusion does NOT follow from the evidence."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Reasoning: {thought}\n"
            f"Answer: {answer}\n"
            f"Evidence: {evidence}\n\n"
            "Is there a CLEAR logical error in the reasoning that leads to a wrong answer? "
            "Reply VALID if the answer is reasonably supported, or INVALID: <describe the specific error>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("INVALID"):
        return f"[REASONING CHAIN CHECK] {response}"
    return None


@register_handler("verify_answer_relevance", requires_api=True)
def verify_answer_relevance(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Check if the answer addresses the question (merged scope + completeness).

    Only flags CLEAR mismatches — ignores minor verbosity or style issues.
    """
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""

    if not question or not answer:
        return None

    messages = [
        {"role": "system", "content": (
            "You verify that an answer addresses the question asked. "
            "Only flag CLEAR, DEFINITIVE problems. Minor style differences, "
            "extra context, or slight verbosity are NOT problems."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Does this answer attempt to address the question? "
            "Reply RELEVANT if the answer is on-topic, or "
            "IRRELEVANT: <brief explanation> ONLY if it answers a completely different question."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("IRRELEVANT"):
        return f"[RELEVANCE CHECK] {response}"
    return None


@register_handler("verify_adversarial_distraction", requires_api=True)
def verify_adversarial_distraction(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Cross-check answer against potentially conflicting sources."""
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    if not evidence or not question:
        return None

    messages = [
        {"role": "system", "content": (
            "You check if source documents DIRECTLY CONTRADICT the proposed answer. "
            "Only flag when sources explicitly state a different fact. "
            "Missing information or incomplete sources are NOT contradictions."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n"
            f"Source documents: {evidence}\n\n"
            "Do any sources explicitly state a DIFFERENT answer to this question? "
            "Reply VERIFIED if no contradiction, or SUSPICIOUS: <quote the conflicting fact>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("SUSPICIOUS"):
        return f"[ADVERSARIAL CHECK] {response}"
    return None


@register_handler("verify_hallucination_grounding", requires_api=True)
def verify_hallucination_grounding(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Check if the CORE answer claim is grounded in source documents."""
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    if not evidence or not question:
        return None

    messages = [
        {"role": "system", "content": (
            "You check if the CORE factual claim in the answer is supported by sources. "
            "Focus on the main answer, not peripheral details. "
            "Common knowledge that supplements source facts is acceptable."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Source documents: {evidence}\n\n"
            "Is the CORE answer claim supported by the source documents? "
            "Reply GROUNDED if the main answer is supported, or "
            "UNGROUNDED: <what specific claim is fabricated>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("UNGROUNDED"):
        return f"[GROUNDING CHECK] {response}"
    return None


## resolve_question_ambiguity — REMOVED (100% false positive rate on factoid QA)


@register_handler("verify_reading_comprehension", requires_api=True)
def verify_reading_comprehension(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Verify that facts were correctly extracted from documents."""
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    if not evidence:
        return None

    messages = [
        {"role": "system", "content": "You verify extraction accuracy from source documents."},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Source documents: {evidence}\n\n"
            "Was the answer correctly extracted from the source documents? Check for "
            "misread numbers, swapped entities, or misinterpreted passages. "
            "Reply CORRECT if accurately extracted, or MISREAD: <describe what was misread>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("MISREAD"):
        return f"[READING CHECK] {response}"
    return None


@register_handler("verify_reasoning_steps", requires_api=True)
def verify_reasoning_steps(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Check logic and calculations in reasoning against source evidence."""
    question = context.get("question", "") or ""
    thought = context.get("thought", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    messages = [
        {"role": "system", "content": (
            "You verify logical reasoning and calculations against source evidence. "
            "Only flag CLEAR errors where the conclusion contradicts the evidence or "
            "contains a demonstrable calculation mistake. Minor gaps in explanation are NOT errors."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Reasoning: {thought}\n"
            f"Answer: {answer}\n"
            f"Evidence from sources: {evidence}\n\n"
            "Is there a CLEAR logical or calculation error that leads to a wrong answer? "
            "Reply SOUND if the reasoning is correct, or FLAWED: <describe the specific error>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("FLAWED"):
        return f"[REASONING CHECK] {response}"
    return None


@register_handler("verify_entity_disambiguation", requires_api=True)
def verify_entity_disambiguation(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Verify that the correct entity was identified."""
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "")

    if not evidence:
        return None

    messages = [
        {"role": "system", "content": "You verify entity identity in QA tasks."},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Source documents: {evidence}\n\n"
            "Is the answer about the correct entity? Check for confusion between "
            "similarly-named people, places, or things. "
            "Reply CORRECT_ENTITY if right, or WRONG_ENTITY: <describe the confusion>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("WRONG_ENTITY"):
        return f"[ENTITY CHECK] {response}"
    return None


## verify_source_authority — REMOVED (86% false positive rate, web search inherently mixed-quality)


@register_handler("check_language_handling", requires_api=True)
def check_language_handling(context: Dict[str, Any], teacher_model=None) -> Optional[str]:
    """Check for transliteration or foreign name handling issues."""
    question = context.get("question", "") or ""
    answer = context.get("action_arg", "") or ""
    evidence = _truncate(context.get("all_read_contents", "") or "", max_chars=2000)

    if not evidence:
        return None

    # Quick check: any non-ASCII in sources?
    non_ascii_count = sum(1 for c in evidence if ord(c) > 127)
    if non_ascii_count < 5:
        return None

    messages = [
        {"role": "system", "content": "You check for language/transliteration issues in QA."},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Source documents: {evidence}\n\n"
            "Are there transliteration, romanization, or foreign language issues that "
            "could cause the answer to be incorrect? "
            "Reply HANDLED if no issues, or CONFUSION: <describe the language issue>."
        )},
    ]

    response = _call_teacher(teacher_model, messages, context)
    if response is None:
        return None

    response = response.strip()
    if response.upper().startswith("CONFUSION"):
        return f"[LANGUAGE CHECK] {response}"
    return None


## verify_general_quality — REMOVED (94.6% false positive rate, overlaps with other handlers)


# ============================================================================
# Multi-PF helper Deliberation (Plan 3)
# ============================================================================

@dataclass
class DeliberationRecord:
    """Record of multi-PF helper deliberation for one handler."""
    handler_id: str
    skill_id: str = ""
    teacher_opinions: List[dict] = field(default_factory=list)  # [{model, verdict, response}]
    consensus: str = ""       # "triggered" | "clean" | "split"
    strategy: str = "majority"
    final_result: str = ""    # "triggered" | "clean"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "handler_id": self.handler_id,
            "teacher_opinions": self.teacher_opinions,
            "consensus": self.consensus,
            "strategy": self.strategy,
            "final_result": self.final_result,
        }
        if self.skill_id:
            d["skill_id"] = self.skill_id
        if self.latency_ms > 0:
            d["latency_ms"] = round(self.latency_ms, 1)
        return d


def execute_handler_multi_teacher(
    handler_id: str,
    context: Dict[str, Any],
    teacher_models: list,
    strategy: str = "majority",
    skill_id: str = "",
) -> Tuple[Optional[str], Optional[DeliberationRecord]]:
    """Execute an LLM-assisted handler with multiple PF helper models.

    Each PF helper independently evaluates. Consensus determines the final verdict.

    Args:
        handler_id: Registered handler ID.
        context: Step context dict.
        teacher_models: List of APIModelWrapper instances.
        strategy: "majority" | "unanimous" | "any"
        skill_id: Originating skill ID.

    Returns:
        (intervention_text_or_None, DeliberationRecord)
    """
    entry = _HANDLER_REGISTRY.get(handler_id)
    if entry is None:
        return None, None

    handler_fn, requires_api = entry
    if not requires_api:
        # Pure-code handler — no deliberation needed, run once
        result = execute_handler(handler_id, context, None, skill_id=skill_id)
        return result, None

    if not teacher_models:
        return None, None

    t0 = time.monotonic()
    opinions = []
    intervention_texts = []

    for teacher in teacher_models:
        model_name = getattr(teacher, "model_name", "unknown")
        # Create isolated context copy so _last_teacher_call doesn't collide
        ctx_copy = dict(context)
        ctx_copy.pop("_last_teacher_call", None)

        try:
            result_text = handler_fn(ctx_copy, teacher)
            last_call = ctx_copy.pop("_last_teacher_call", None)
            response_text = last_call.get("response", "") if last_call else ""

            triggered = result_text is not None
            opinions.append({
                "model": model_name,
                "verdict": "triggered" if triggered else "clean",
                "response": response_text[:300] if response_text else "",
            })
            if triggered:
                intervention_texts.append(result_text)
        except Exception as e:
            logger.warning(f"[deliberation] {handler_id}/{model_name} error: {e}")
            opinions.append({
                "model": model_name,
                "verdict": "error",
                "response": str(e)[:200],
            })

    latency = (time.monotonic() - t0) * 1000

    # Compute consensus
    triggered_count = sum(1 for o in opinions if o["verdict"] == "triggered")
    total_valid = sum(1 for o in opinions if o["verdict"] != "error")

    if total_valid == 0:
        consensus = "clean"
        final_result = "clean"
    elif strategy == "unanimous":
        consensus = "triggered" if triggered_count == total_valid else (
            "clean" if triggered_count == 0 else "split"
        )
        final_result = "triggered" if triggered_count == total_valid else "clean"
    elif strategy == "any":
        consensus = "triggered" if triggered_count > 0 else "clean"
        final_result = "triggered" if triggered_count > 0 else "clean"
    else:  # majority
        consensus = "triggered" if triggered_count > total_valid / 2 else (
            "clean" if triggered_count == 0 else "split"
        )
        final_result = "triggered" if triggered_count > total_valid / 2 else "clean"

    record = DeliberationRecord(
        handler_id=handler_id,
        skill_id=skill_id,
        teacher_opinions=opinions,
        consensus=consensus,
        strategy=strategy,
        final_result=final_result,
        latency_ms=latency,
    )

    # Also append to handler_records for backward compatibility
    handler_record = HandlerRecord(
        handler_id=handler_id,
        skill_id=skill_id,
        handler_type="llm_assisted_multi",
        result=final_result,
        intervention_text=intervention_texts[0] if intervention_texts else None,
        teacher_model_name=",".join(o["model"] for o in opinions),
        latency_ms=round(latency, 1),
    )
    _append_record(context, handler_record)

    intervention = intervention_texts[0] if (final_result == "triggered" and intervention_texts) else None

    if final_result == "triggered":
        logger.info(
            f"[deliberation] {handler_id}: TRIGGERED ({triggered_count}/{total_valid} "
            f"helpers agree, strategy={strategy})"
        )
    elif consensus == "split":
        logger.info(
            f"[deliberation] {handler_id}: SPLIT ({triggered_count}/{total_valid} "
            f"helpers triggered, strategy={strategy} → clean)"
        )

    return intervention, record
