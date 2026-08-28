"""Process-wide circuit breaker for unrecoverable LLM API errors.

When OpenAI returns `insufficient_quota` (or any 401 auth failure), every
subsequent PF helper call in the same process is DOA. The previous behaviour
was to log-and-retry, which meant a 2h SLURM job could stay alive burning
GPU on nothing but 429 spinlocks.

Call sites wrap their PF helper call with:

    from .quota import guard, note_api_error
    guard()                                  # fail fast if already tripped
    try:
        response = teacher_model.generate(...)
    except Exception as e:
        if note_api_error(e):
            raise                            # propagate so SLURM exits cleanly
        logger.warning(...)                  # transient — keep going
        return None
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

# Substrings in str(exception) that mean "no point retrying, ever"
_FATAL_PATTERNS: List[str] = [
    "insufficient_quota",
    "invalid_api_key",
    "Incorrect API key",
    "authentication",
    "Unauthorized",
]

_tripped: bool = False
_reason: str = ""


class APIQuotaExhausted(RuntimeError):
    """Raised to abort the run when an unrecoverable API error is detected."""


def is_fatal(e: Exception) -> bool:
    msg = str(e)
    return any(p in msg for p in _FATAL_PATTERNS)


def note_api_error(e: Exception) -> bool:
    """Inspect `e`. If it matches a fatal pattern, trip the breaker and
    return True (caller should raise). Otherwise return False."""
    global _tripped, _reason
    if is_fatal(e):
        if not _tripped:
            _tripped = True
            _reason = str(e)[:300]
            logger.error("LLM circuit breaker TRIPPED: %s", _reason)
        return True
    return False


def guard() -> None:
    """Raise immediately if the breaker has been tripped earlier in this run."""
    if _tripped:
        raise APIQuotaExhausted(
            f"Aborting further PF helper calls; first fatal error was: {_reason}"
        )


def reset() -> None:
    """Only for tests."""
    global _tripped, _reason
    _tripped = False
    _reason = ""


def is_tripped() -> bool:
    return _tripped
