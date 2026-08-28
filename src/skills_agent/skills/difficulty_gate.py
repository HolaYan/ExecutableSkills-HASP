"""
Difficulty Gate — estimates question difficulty to decide whether skills should activate.

Uses a lightweight LLM call (single turn, ~50 tokens) to classify question difficulty
on a 1-5 scale. Skills are only injected for questions at or above the threshold.

Design rationale:
- Skills help when baseline MBE is 30-50% (GAIA, weak models) but hurt when >70% (FRAMES)
- A single cheap difficulty estimation call (~$0.0001) avoids wasting prompt tokens on easy questions
- Can also use heuristic mode (no LLM) based on question surface features
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DIFFICULTY_PROMPT = """Rate this question's difficulty for a web search agent on a scale of 1-5:

1: Simple factoid — one search should find the answer directly
2: Easy lookup — needs 1-2 searches but answer is straightforward
3: Multi-hop — requires chaining 2-3 facts from different sources
4: Complex reasoning — requires multiple searches, reading, and careful reasoning
5: Extreme — requires many constraints, deep research, or specialized knowledge

Question: {question}

Respond with ONLY a single number (1-5)."""


class DifficultyGate:
    """Estimates question difficulty and gates skill activation."""

    def __init__(self, teacher_model=None, threshold: int = 3):
        """
        Args:
            teacher_model: APIModelWrapper for LLM-based estimation (optional).
                          If None, uses heuristic mode.
            threshold: Minimum difficulty score to enable skills (1-5).
        """
        self._teacher = teacher_model
        self._threshold = threshold

    def should_enable_skills(self, question: str) -> bool:
        """Estimate difficulty and return True if skills should be enabled."""
        score = self.estimate_difficulty(question)
        enabled = score >= self._threshold
        logger.info(
            f"[DifficultyGate] score={score}, threshold={self._threshold}, "
            f"skills_enabled={enabled}"
        )
        return enabled

    def estimate_difficulty(self, question: str) -> int:
        """Estimate question difficulty on a 1-5 scale.

        Uses LLM if teacher_model is available, otherwise falls back to
        heuristic estimation.
        """
        if self._teacher is not None:
            return self._estimate_llm(question)
        return self._estimate_heuristic(question)

    def _estimate_llm(self, question: str) -> int:
        """Use a lightweight LLM call to estimate difficulty."""
        try:
            prompt = _DIFFICULTY_PROMPT.format(question=question)
            response = self._teacher.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            # Extract digit from response
            digits = re.findall(r'[1-5]', response or "")
            if digits:
                return int(digits[0])
            logger.warning(f"[DifficultyGate] LLM returned no valid score: {response}")
            return self._estimate_heuristic(question)
        except Exception as e:
            logger.warning(f"[DifficultyGate] LLM estimation failed: {e}")
            return self._estimate_heuristic(question)

    def _estimate_heuristic(self, question: str) -> int:
        """Heuristic difficulty estimation based on question surface features.

        Signals:
        - Length: longer questions tend to be harder
        - Multi-hop indicators: possessives, relative clauses, "of the"
        - Constraint indicators: date ranges, "between X and Y"
        - Computation indicators: "how many", "calculate", "total"
        """
        score = 2  # Default: easy lookup
        q_lower = question.lower()
        words = question.split()

        # Length signal
        if len(words) > 40:
            score += 1
        if len(words) > 80:
            score += 1

        # Multi-hop indicators
        possessives = question.count("'s")
        relative_clauses = sum(1 for w in ["who", "which", "whose", "that", "where"]
                              if f" {w} " in f" {q_lower} ")
        of_the_count = q_lower.count(" of the ")
        hop_signal = possessives + relative_clauses + of_the_count
        if hop_signal >= 2:
            score += 1
        if hop_signal >= 4:
            score += 1

        # Constraint indicators (BrowseComp-style)
        date_ranges = len(re.findall(
            r'between\s+\d{4}\s+and\s+\d{4}|\d{4}\s*[-–]\s*\d{4}',
            question, re.IGNORECASE
        ))
        if date_ranges >= 2:
            score += 1

        # Computation indicators
        compute_words = ["how many", "calculate", "total", "sum", "difference",
                        "average", "percentage", "ratio"]
        if any(w in q_lower for w in compute_words):
            score += 1

        return min(score, 5)
