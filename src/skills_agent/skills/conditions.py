"""
Condition evaluator for phase-gated skill injection.

Lightweight, stateless, thread-safe module that evaluates conditions
against observation context. All condition functions are pure and
registered in a global registry.
"""

import re
from typing import List, Dict, Any, Callable

# Registry: condition_name -> pure function(context) -> bool
_CONDITION_REGISTRY: Dict[str, Callable[[Dict[str, Any]], bool]] = {}


def register_condition(name: str):
    """Decorator to register a condition function."""
    def decorator(fn: Callable[[Dict[str, Any]], bool]):
        _CONDITION_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================================
# Condition functions (all pure, stateless, thread-safe)
# ============================================================================

@register_condition("always")
def _always(context: Dict[str, Any]) -> bool:
    """Unconditional — always fires."""
    return True


@register_condition("search_empty")
def _search_empty(context: Dict[str, Any]) -> bool:
    """Last search returned no results."""
    obs = context.get("observation_text", "")
    return "No results found" in obs or context.get("empty_results", False)


@register_condition("search_has_conflicts")
def _search_has_conflicts(context: Dict[str, Any]) -> bool:
    """Search results contain conflicting factual claims.

    Improved over the old _detect_contradictions: checks for contradiction
    indicators in the observation text, plus the explicit flag.
    """
    if context.get("contradictory_sources", False):
        return True

    obs = context.get("observation_text", "")
    if not obs:
        return False

    obs_lower = obs.lower()
    # Check for patterns indicating conflicting information
    conflict_indicators = [
        "however", "contrary", "incorrect", "not true",
        "actually", "in fact", "disputed", "false",
        "conflicting", "disagree", "contradicts",
    ]
    return any(indicator in obs_lower for indicator in conflict_indicators)


@register_condition("search_has_similar_entities")
def _search_has_similar_entities(context: Dict[str, Any]) -> bool:
    """Multiple results with similar-but-different entity names."""
    if context.get("similar_entity_results", False):
        return True

    obs = context.get("observation_text", "")
    if not obs:
        return False

    # Look for multiple doc entries with similar titles
    titles = re.findall(r'\[doc_\d+\]\s*(.+?):', obs)
    if len(titles) < 2:
        return False

    # Check if any pair of titles share significant word overlap
    title_words = [set(t.lower().split()) for t in titles]
    for i in range(len(title_words)):
        for j in range(i + 1, len(title_words)):
            common = title_words[i] & title_words[j]
            # Filter common stopwords and short words
            common -= {"the", "a", "an", "of", "in", "and", "or", "to", "is", "-", "–", "for", "on", "at"}
            if len(common) >= 1:
                return True
    return False


@register_condition("read_has_numbers")
def _read_has_numbers(context: Dict[str, Any]) -> bool:
    """Document contains numerical data requiring extraction."""
    obs = context.get("observation_text", "")
    if not obs or context.get("action_type") not in ("READ", "SUMMARY"):
        return False
    # Check for numbers that look like data (years, quantities, etc.)
    numbers = re.findall(r'\b\d{2,}\b', obs)
    return len(numbers) >= 2


@register_condition("read_has_multiple_entities")
def _read_has_multiple_entities(context: Dict[str, Any]) -> bool:
    """Document discusses multiple people/places that could be confused."""
    obs = context.get("observation_text", "")
    if not obs or context.get("action_type") not in ("READ", "SUMMARY"):
        return False
    # Rough heuristic: count capitalized multi-word phrases (proper nouns)
    proper_nouns = re.findall(r'(?<!\. )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', obs)
    unique_entities = set(proper_nouns)
    return len(unique_entities) >= 3


@register_condition("no_read_yet")
def _no_read_yet(context: Dict[str, Any]) -> bool:
    """Agent hasn't READ any document yet."""
    return not context.get("has_read", False)


@register_condition("step_count_high")
def _step_count_high(context: Dict[str, Any]) -> bool:
    """Approaching budget limit (>= 70% of max_steps)."""
    step_count = context.get("step_count", 0)
    max_steps = context.get("max_steps", 10)
    return step_count >= max_steps * 0.7


# ============================================================================
# New conditions for expanded skill set
# ============================================================================

@register_condition("question_has_temporal")
def _question_has_temporal(context: Dict[str, Any]) -> bool:
    """Question contains temporal keywords or years."""
    question = context.get("question", "")
    if not question:
        return False
    # Check for year patterns
    if re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', question):
        return True
    # Check for temporal keywords
    temporal_kw = {"when", "year", "date", "century", "decade", "era", "period",
                   "before", "after", "during", "since", "until"}
    question_lower = question.lower()
    return any(kw in question_lower.split() for kw in temporal_kw)


@register_condition("question_has_numbers")
def _question_has_numbers(context: Dict[str, Any]) -> bool:
    """Question contains numerical keywords or digits."""
    question = context.get("question", "")
    if not question:
        return False
    # Check for digits (2+)
    if re.search(r'\b\d{2,}\b', question):
        return True
    # Check for numerical keywords
    num_kw = {"how many", "how much", "number", "count", "total", "population",
              "percentage", "amount", "quantity", "size", "length", "height",
              "weight", "distance", "area"}
    question_lower = question.lower()
    return any(kw in question_lower for kw in num_kw)


@register_condition("question_is_multi_part")
def _question_is_multi_part(context: Dict[str, Any]) -> bool:
    """Question has multiple sub-questions (multiple '?' or 'and' structure)."""
    question = context.get("question", "")
    if not question:
        return False
    # Multiple question marks
    if question.count("?") >= 2:
        return True
    # "and" joining distinct aspects
    q_lower = question.lower()
    if " and " in q_lower and len(question.split()) > 10:
        return True
    return False


@register_condition("question_has_negation")
def _question_has_negation(context: Dict[str, Any]) -> bool:
    """Question contains negation words (not, never, except...)."""
    question = context.get("question", "")
    if not question:
        return False
    negation_words = {"not", "never", "none", "neither", "nor", "except",
                      "without", "other than", "besides", "excluding"}
    q_lower = question.lower()
    return any(neg in q_lower for neg in negation_words)


@register_condition("only_one_source_read")
def _only_one_source_read(context: Dict[str, Any]) -> bool:
    """Agent has read exactly 1 document."""
    return context.get("read_count", 0) == 1


@register_condition("answer_is_verbose")
def _answer_is_verbose(context: Dict[str, Any]) -> bool:
    """Pending answer is more than 10 words."""
    answer = context.get("_pending_original_arg", "")
    if not answer:
        return False
    return len(answer.split()) > 10


@register_condition("question_is_multi_hop")
def _question_is_multi_hop(context: Dict[str, Any]) -> bool:
    """Question requires multi-hop reasoning (long or chained structure)."""
    question = context.get("question", "")
    if not question:
        return False
    # Long questions are more likely multi-hop
    if len(question.split()) > 20:
        return True
    # Multi-hop indicator phrases
    multi_hop_phrases = [
        "the .+ of the .+ that",
        "the .+ who .+ the",
        "born in the .+ where",
        "directed by the .+ who",
        "written by the .+ of",
    ]
    q_lower = question.lower()
    return any(re.search(p, q_lower) for p in multi_hop_phrases)


class ConditionEvaluator:
    """Evaluates conditions against observation context.

    Thread-safe: all condition functions are pure and stateless.
    """

    @staticmethod
    def evaluate(conditions: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate all conditions (AND logic).

        Args:
            conditions: List of condition names to check.
            context: Dictionary with observation context.

        Returns:
            True if ALL conditions match. Empty list → always True.
        """
        if not conditions:
            return True

        for cond_name in conditions:
            fn = _CONDITION_REGISTRY.get(cond_name)
            if fn is None:
                # Unknown condition → skip (fail-open for forward compat)
                continue
            if not fn(context):
                return False
        return True

    @staticmethod
    def available_conditions() -> List[str]:
        """Return list of registered condition names."""
        return sorted(_CONDITION_REGISTRY.keys())
