"""
Answer post-processor for factoid QA.

Pure rule-based pipeline that cleans model-generated answers:
1. Strip verbose prefixes ("The answer is ...")
2. Remove parenthetical supplements
3. Strip trailing punctuation
4. Extract core entity for factoid questions

All functions are pure, stateless, and thread-safe.
"""

import re
from typing import Optional


# Prefix patterns to strip (case-insensitive)
_PREFIX_RE = re.compile(
    r"^(?:the answer is|based on (?:the |my )?(?:search|research|findings|reading|information|sources?|documents?|evidence),?|"
    r"according to (?:the |my )?(?:search|research|findings|reading|information|sources?|documents?|evidence),?|"
    r"i (?:found|believe|think) (?:that (?:it (?:is|was) )?|the answer (?:is|to be) )?|"
    r"it (?:is|was|appears to be|seems to be) |"
    r"the (?:answer|result) (?:is|appears to be|seems to be) )"
    r"\s*",
    re.IGNORECASE,
)

# Factoid question starters (case-insensitive)
_FACTOID_RE = re.compile(
    r"^(?:who|what|when|where|which|how many|how much|how old|how long|"
    r"how tall|how far|name the|name a|in what|at what|on what)\b",
    re.IGNORECASE,
)


def postprocess_answer(raw_answer: str, question: str = "") -> str:
    """Clean a model-generated answer through a rule-based pipeline.

    Args:
        raw_answer: The raw answer string from the model.
        question: The original question (used for factoid detection).

    Returns:
        Cleaned answer string.
    """
    if not raw_answer:
        return raw_answer

    answer = raw_answer.strip()

    # Step 1: Strip verbose prefixes
    answer = _strip_prefix(answer)

    # Step 2: Remove parenthetical supplements
    answer = _strip_parenthetical(answer)

    # Step 3: Strip trailing punctuation
    answer = _strip_trailing_punctuation(answer)

    # Step 4: Extract core entity (only for factoid questions with longer answers)
    if question and _is_factoid(question):
        answer = _extract_core_entity(answer)

    # Final trim
    answer = answer.strip()

    # Safety: never return empty string if we had input
    if not answer and raw_answer.strip():
        return raw_answer.strip()

    return answer


def _strip_prefix(answer: str) -> str:
    """Remove verbose answer prefixes."""
    return _PREFIX_RE.sub("", answer).strip()


def _strip_parenthetical(answer: str) -> str:
    """Remove trailing parenthetical supplements like '(capital of France)'."""
    # Only strip parenthetical at the end of the answer
    stripped = re.sub(r"\s*\([^)]*\)\s*$", "", answer)
    return stripped.strip() if stripped.strip() else answer


def _strip_trailing_punctuation(answer: str) -> str:
    """Remove trailing periods, commas, semicolons (but not question/exclamation marks)."""
    return answer.rstrip(".,;: ")


def _is_factoid(question: str) -> bool:
    """Check if a question expects a short factoid answer."""
    return bool(_FACTOID_RE.match(question.strip()))


def _extract_core_entity(answer: str) -> str:
    """Extract the core entity from a factoid answer.

    Rules:
    - Comma rule always applies: "Paris, France" → "Paris"
    - Short answers (<=4 words, no comma): return as-is
    - Contains ' is ' or ' was ': take the complement (part after is/was)
    """
    # Comma rule: "Paris, France" → "Paris" (always applies, even for short answers)
    if "," in answer:
        parts = answer.split(",", 1)
        candidate = parts[0].strip()
        if candidate:
            return candidate

    words = answer.split()
    if len(words) <= 4:
        return answer

    # is/was rule: "The capital is Paris" → "Paris"
    for sep in (" is ", " was "):
        if sep in answer.lower():
            idx = answer.lower().index(sep)
            candidate = answer[idx + len(sep):].strip()
            # Strip trailing punctuation from extracted part
            candidate = candidate.rstrip(".,;: ")
            if candidate:
                return candidate

    return answer
