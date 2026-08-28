"""
Loop detector for agent action histories.

Lightweight, stateless, thread-safe — same design pattern as conditions.py.
Detects repeated searches, search-only loops, and oscillation patterns.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional


@dataclass
class LoopDetection:
    """Result of loop detection analysis."""
    loop_detected: bool = False
    loop_type: Optional[str] = None        # "repeated_search" | "search_only" | "oscillation"
    recommended_action: Optional[str] = None  # "force_read" | "force_final"
    details: str = ""


def detect_loop(
    action_history: List[Dict[str, Any]],
    step_context: Dict[str, Any],
    threshold: int = 3,
) -> LoopDetection:
    """Detect looping behavior in the agent's action history.

    All checks are pure and stateless — safe for concurrent use.

    Args:
        action_history: List of {"action_type": str, "arg": str, "step": int}.
        step_context: Dict with search_count, has_read, empty_results, last_search_results_text, etc.
        threshold: Minimum consecutive actions to trigger search_only detection.

    Returns:
        LoopDetection with diagnosis and recommended corrective action.
    """
    if len(action_history) < 2:
        return LoopDetection()

    search_actions = [a for a in action_history if a["action_type"] == "SEARCH"]

    # --- Rule 1: repeated_search ---
    # Latest query is very similar to 2+ prior queries (word-level overlap)
    if len(search_actions) >= 3:
        latest_query = search_actions[-1]["arg"]
        similar_count = _count_similar_queries(latest_query, search_actions[:-1])
        if similar_count >= 2:
            has_results = bool(step_context.get("last_search_results_text"))
            if has_results:
                return LoopDetection(
                    loop_detected=True,
                    loop_type="repeated_search",
                    recommended_action="force_read",
                    details=f"Query '{latest_query}' similar to {similar_count} prior queries",
                )
            else:
                return LoopDetection(
                    loop_detected=True,
                    loop_type="repeated_search",
                    recommended_action="force_final",
                    details=f"Query '{latest_query}' similar to {similar_count} prior queries (no results)",
                )

    # --- Rule 2: search_only ---
    # N+ consecutive SEARCH with no READ
    recent = action_history[-threshold:]
    if len(recent) >= threshold and all(a["action_type"] == "SEARCH" for a in recent):
        has_results = bool(step_context.get("last_search_results_text"))
        if has_results:
            return LoopDetection(
                loop_detected=True,
                loop_type="search_only",
                recommended_action="force_read",
                details=f"{threshold}+ consecutive SEARCH without READ (results available)",
            )
        else:
            return LoopDetection(
                loop_detected=True,
                loop_type="search_only",
                recommended_action="force_final",
                details=f"{threshold}+ consecutive SEARCH without READ (no results)",
            )

    # --- Rule 3: oscillation ---
    # SEARCH-READ-SEARCH-READ-SEARCH-READ pattern (6 steps) with similar queries
    if len(action_history) >= 6:
        last6 = action_history[-6:]
        pattern = [a["action_type"] for a in last6]
        expected = ["SEARCH", "READ", "SEARCH", "READ", "SEARCH", "READ"]
        if pattern == expected:
            queries = [a["arg"] for a in last6 if a["action_type"] == "SEARCH"]
            if len(queries) >= 2:
                # Check pairwise similarity using word overlap
                all_similar = True
                for i in range(len(queries) - 1):
                    if not _queries_similar(queries[i], queries[i + 1]):
                        all_similar = False
                        break
                if all_similar:
                    return LoopDetection(
                        loop_detected=True,
                        loop_type="oscillation",
                        recommended_action="force_final",
                        details="SEARCH-READ oscillation with similar queries",
                    )

    return LoopDetection()


# Stopwords to ignore in word-level similarity
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "was",
    "are", "were", "and", "or", "not", "it", "by", "with", "from", "as",
})


def _normalize_query_words(query: str) -> set:
    """Extract meaningful content words from a query."""
    words = set(query.lower().split())
    return words - _STOPWORDS


def _queries_similar(q1: str, q2: str) -> bool:
    """Check if two queries are semantically similar using word overlap.

    Uses Jaccard similarity on content words: overlap / union >= 0.4
    """
    w1 = _normalize_query_words(q1)
    w2 = _normalize_query_words(q2)
    if not w1 or not w2:
        return False
    intersection = w1 & w2
    union = w1 | w2
    return len(intersection) / len(union) >= 0.4


def _count_similar_queries(query: str, prior_actions: list) -> int:
    """Count how many prior search queries are similar to the given query."""
    count = 0
    for prev in prior_actions:
        if _queries_similar(query, prev["arg"]):
            count += 1
    return count
