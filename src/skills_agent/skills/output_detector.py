"""
Output Problem Detector (Layer 0) — reactive skill triggering.

Analyzes the model's thought + action output *before* tool dispatch to detect
if the output exhibits problems described by any active skill. Pure, stateless,
thread-safe — same design pattern as LoopDetection and ConditionEvaluator.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, List

from .loop_detector import _queries_similar


@dataclass
class DetectedProblem:
    """A problem detected in the model's output."""
    skill_id: str           # which skill this problem maps to
    problem_type: str       # specific sub-type (e.g. "multiple_entities_no_disambiguation")
    confidence: float       # 0.0-1.0
    evidence: str           # human-readable detection evidence
    suggested_action: str   # "inject_text" | "programmatic_override" | "helper_review"


class OutputProblemDetector:
    """Detects problems in model output before tool dispatch.

    Pure and stateless — all detection methods are thread-safe for async
    batch inference. Constructor takes only a confidence threshold.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def detect(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Run all detectors and return problems above the confidence threshold.

        Args:
            thought: The model's reasoning/thought text.
            action_type: Parsed action type ("SEARCH", "READ", "FINAL", etc.)
            action_arg: Parsed action argument.
            step_context: Current step context dict.

        Returns:
            List of DetectedProblem instances with confidence >= threshold.
        """
        all_problems: List[DetectedProblem] = []

        all_problems.extend(self._detect_entity_confusion(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_insufficient_exploration(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_hallucination(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_retrieval_failure(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_format_issue(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_temporal_confusion(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_numerical_error(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_negation_oversight(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_premature_commitment(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_answer_completeness(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_citation_mismatch(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_outdated_information(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_multi_hop_failure(thought, action_type, action_arg, step_context))
        all_problems.extend(self._detect_language_barrier(thought, action_type, action_arg, step_context))
        # source_authority_error — REMOVED (86% false positive rate)

        # Filter by confidence threshold
        return [p for p in all_problems if p.confidence >= self.threshold]

    # ========================================================================
    # Detection methods (all pure, stateless, thread-safe)
    # ========================================================================

    def _detect_entity_confusion(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect entity confusion: model mentions 2+ entities from search results
        without comparing/distinguishing them.

        Key signals:
        - Entity word overlap between thought and search_result_entities in step_context
        - Absence of disambiguation markers ("however", "but", "unlike", etc.)
        """
        if not thought:
            return []

        # Extract entities from thought and from last search results
        thought_entities = _extract_entities_from_text(thought)
        results_text = step_context.get("last_search_results_text", "")
        result_entities = _extract_entities_from_text(results_text)

        if len(thought_entities) < 2 or len(result_entities) < 2:
            return []

        # Find entities from results that appear in thought
        overlapping = []
        for te in thought_entities:
            te_lower = te.lower()
            for re_ent in result_entities:
                if re_ent.lower() in te_lower or te_lower in re_ent.lower():
                    overlapping.append(te)
                    break

        if len(overlapping) < 2:
            return []

        # Check for disambiguation markers
        disambiguation_markers = [
            "however", "but", "unlike", "in contrast", "whereas",
            "on the other hand", "different from", "not to be confused",
            "distinguish", "as opposed to",
        ]
        thought_lower = thought.lower()
        has_disambiguation = any(m in thought_lower for m in disambiguation_markers)

        if has_disambiguation:
            return []

        # Confidence: higher when more entities overlap without disambiguation
        confidence = min(0.5 + 0.15 * len(overlapping), 0.95)

        return [DetectedProblem(
            skill_id="wrong_entity_confusion",
            problem_type="multiple_entities_no_disambiguation",
            confidence=confidence,
            evidence=f"Thought mentions entities {overlapping[:3]} from search results without disambiguation",
            suggested_action="inject_text",
        )]

    def _detect_insufficient_exploration(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect insufficient exploration: model attempts FINAL without adequate
        evidence gathering.

        Key signals:
        - action_type == "FINAL" and (has_read == False or search_count <= 1)
        - "I don't have enough" in thought followed by FINAL
        """
        if action_type != "FINAL":
            return []

        has_read = step_context.get("has_read", False)
        search_count = step_context.get("search_count", 0)

        problems = []

        # Case 1: FINAL without reading any document
        if not has_read:
            confidence = 0.9 if search_count >= 1 else 0.7
            problems.append(DetectedProblem(
                skill_id="insufficient_exploration",
                problem_type="final_without_read",
                confidence=confidence,
                evidence=f"Attempting FINAL without any READ (search_count={search_count})",
                suggested_action="programmatic_override",
            ))

        # Case 2: FINAL with only 1 search and no deep exploration
        elif search_count <= 1:
            confidence = 0.65
            problems.append(DetectedProblem(
                skill_id="insufficient_exploration",
                problem_type="final_with_minimal_search",
                confidence=confidence,
                evidence=f"Attempting FINAL with only {search_count} search(es)",
                suggested_action="inject_text",
            ))

        # Case 3: Thought indicates uncertainty but still answering
        if thought:
            uncertainty_phrases = [
                "i don't have enough", "not enough information",
                "i'm not sure", "i cannot determine",
                "unable to find", "no information",
            ]
            thought_lower = thought.lower()
            for phrase in uncertainty_phrases:
                if phrase in thought_lower:
                    confidence = 0.85
                    problems.append(DetectedProblem(
                        skill_id="insufficient_exploration",
                        problem_type="uncertain_final",
                        confidence=confidence,
                        evidence=f"Thought expresses uncertainty ('{phrase}') but still attempting FINAL",
                        suggested_action="programmatic_override",
                    ))
                    break

        return problems

    def _detect_hallucination(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect hallucination: model makes factual claims in FINAL that aren't
        supported by read documents.

        Key signals:
        - action_type == "FINAL"
        - Extract specific claims (numbers, years, proper nouns) from thought
        - Cross-check against all_read_contents in step_context
        """
        if action_type != "FINAL":
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read or not thought:
            return []

        # Extract specific claims from the answer and thought
        answer_claims = _extract_specific_claims(action_arg)
        thought_claims = _extract_specific_claims(thought)
        all_claims = list(set(answer_claims + thought_claims))

        if not all_claims:
            return []

        # Check how many claims appear in read content
        all_read_lower = all_read.lower()
        unsupported = []
        supported = []
        for claim in all_claims:
            if claim.lower() in all_read_lower:
                supported.append(claim)
            else:
                unsupported.append(claim)

        if not unsupported:
            return []

        total = len(all_claims)
        unsupported_ratio = len(unsupported) / total if total > 0 else 0

        # Only flag if a significant portion is unsupported
        if unsupported_ratio < 0.3:
            return []

        confidence = min(0.5 + 0.3 * unsupported_ratio, 0.9)

        return [DetectedProblem(
            skill_id="hallucination",
            problem_type="unsupported_claims",
            confidence=confidence,
            evidence=f"{len(unsupported)}/{total} claims not found in read documents: {unsupported[:3]}",
            suggested_action="programmatic_override",
        )]

    def _detect_retrieval_failure(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect retrieval failure: model uses poor search strategies.

        Key signals:
        - Query is too long (>15 words)
        - Query is verbatim copy of question
        - Consecutive similar queries (Jaccard >= 0.6 via _queries_similar)
        """
        if action_type != "SEARCH":
            return []

        problems = []
        query = action_arg

        # Check 1: Query too long
        word_count = len(query.split())
        if word_count > 15:
            confidence = min(0.6 + 0.02 * (word_count - 15), 0.9)
            problems.append(DetectedProblem(
                skill_id="retrieval_failure",
                problem_type="query_too_long",
                confidence=confidence,
                evidence=f"Search query has {word_count} words (>15)",
                suggested_action="programmatic_override",
            ))

        # Check 2: Query is verbatim copy of question
        question = step_context.get("question", "")
        if question and query.strip().lower() == question.strip().lower():
            problems.append(DetectedProblem(
                skill_id="retrieval_failure",
                problem_type="verbatim_question_as_query",
                confidence=0.75,
                evidence="Search query is verbatim copy of the question",
                suggested_action="programmatic_override",
            ))

        # Check 3: Similar to recent queries (reuse _queries_similar from loop_detector)
        action_history = step_context.get("action_history", [])
        recent_searches = [
            a["arg"] for a in action_history
            if a.get("action_type") == "SEARCH" and a.get("arg")
        ]
        if recent_searches:
            similar_count = sum(
                1 for prev in recent_searches[-3:]
                if _queries_similar(query, prev)
            )
            if similar_count >= 1:
                confidence = min(0.6 + 0.15 * similar_count, 0.9)
                problems.append(DetectedProblem(
                    skill_id="retrieval_failure",
                    problem_type="similar_to_recent_query",
                    confidence=confidence,
                    evidence=f"Query similar to {similar_count} recent search(es)",
                    suggested_action="programmatic_override",
                ))

        return problems

    def _detect_format_issue(
        self,
        thought: str,
        action_type: str,
        action_arg: str,
        step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect format extraction error: model's FINAL answer is a full sentence
        rather than a concise entity.

        Key signals:
        - action_type == "FINAL" and arg contains subject-verb patterns
        - Starts with "Yes/No" followed by explanation
        """
        if action_type != "FINAL":
            return []

        answer = action_arg.strip()
        if not answer:
            return []

        problems = []

        # Check 1: Answer starts with Yes/No followed by explanation
        if re.match(r'^(?:Yes|No)[,.]?\s+\w', answer, re.IGNORECASE):
            problems.append(DetectedProblem(
                skill_id="format_extraction_error",
                problem_type="yes_no_with_explanation",
                confidence=0.8,
                evidence="FINAL answer starts with Yes/No followed by explanation",
                suggested_action="programmatic_override",
            ))

        # Check 2: Answer is a full sentence (contains subject-verb patterns)
        # Heuristic: if answer has >8 words and contains common verb forms
        words = answer.split()
        if len(words) > 8:
            verb_patterns = [
                r'\b(?:is|was|are|were|has|have|had|does|did|do)\b',
                r'\b(?:because|since|therefore|however|although)\b',
            ]
            for pattern in verb_patterns:
                if re.search(pattern, answer, re.IGNORECASE):
                    problems.append(DetectedProblem(
                        skill_id="format_extraction_error",
                        problem_type="full_sentence_answer",
                        confidence=0.7,
                        evidence=f"FINAL answer is {len(words)} words with sentence structure",
                        suggested_action="programmatic_override",
                    ))
                    break

        return problems

    # ========================================================================
    # New detectors for expanded skill coverage
    # ========================================================================

    def _detect_temporal_confusion(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect years in answer not found in read documents."""
        if action_type != "FINAL":
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read or not action_arg:
            return []

        answer_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', action_arg))
        if not answer_years:
            return []

        doc_years = set(re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', all_read))
        unsupported = answer_years - doc_years

        if not unsupported:
            return []

        confidence = min(0.6 + 0.15 * len(unsupported), 0.9)
        return [DetectedProblem(
            skill_id="temporal_confusion",
            problem_type="unsupported_years",
            confidence=confidence,
            evidence=f"Year(s) {unsupported} in answer not found in read documents",
            suggested_action="inject_text",
        )]

    def _detect_numerical_error(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect numbers in answer not found in read documents."""
        if action_type != "FINAL":
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read or not action_arg:
            return []

        answer_nums = set(re.findall(r'\b(\d{2,})\b', action_arg))
        if not answer_nums:
            return []

        doc_nums = set(re.findall(r'\b(\d{2,})\b', all_read))
        unsupported = answer_nums - doc_nums

        if not unsupported:
            return []

        confidence = min(0.6 + 0.1 * len(unsupported), 0.85)
        return [DetectedProblem(
            skill_id="numerical_reasoning_error",
            problem_type="unsupported_numbers",
            confidence=confidence,
            evidence=f"Number(s) {unsupported} in answer not found in read documents",
            suggested_action="inject_text",
        )]

    def _detect_negation_oversight(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect negation in question but absent in reasoning."""
        if action_type != "FINAL" or not thought:
            return []

        question = step_context.get("question", "")
        if not question:
            return []

        negation_words = {"not", "never", "none", "neither", "except", "without",
                          "other than", "besides", "excluding"}
        q_lower = question.lower()
        has_negation = any(neg in q_lower for neg in negation_words)

        if not has_negation:
            return []

        # Check if reasoning addresses the negation
        thought_lower = thought.lower()
        negation_in_reasoning = any(neg in thought_lower for neg in negation_words)

        if negation_in_reasoning:
            return []

        return [DetectedProblem(
            skill_id="negation_oversight",
            problem_type="negation_not_addressed",
            confidence=0.7,
            evidence="Question has negation but reasoning doesn't reflect it",
            suggested_action="inject_text",
        )]

    def _detect_premature_commitment(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect FINAL with zero exploration (no searches AND no reads).

        Merged into insufficient_exploration skill.
        """
        if action_type != "FINAL":
            return []

        search_count = step_context.get("search_count", 0)
        read_count = step_context.get("read_count", 0)
        step_count = step_context.get("step_count", 0)
        max_steps = step_context.get("max_steps", 10)

        # Never flag near budget exhaustion
        if step_count >= max_steps - 2:
            return []

        # Only flag if completely unexplored
        if search_count > 0 or read_count > 0:
            return []

        return [DetectedProblem(
            skill_id="insufficient_exploration",
            problem_type="final_without_exploration",
            confidence=0.9,
            evidence="Attempting FINAL without any SEARCH or READ",
            suggested_action="programmatic_override",
        )]

    def _detect_answer_completeness(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect multi-part question with short answer (merged from partial_answer + scope_mismatch)."""
        if action_type != "FINAL" or not action_arg:
            return []

        question = step_context.get("question", "")
        if not question:
            return []

        # Check if multi-part question with short answer
        is_multi = question.count("?") >= 2 or (" and " in question.lower() and len(question.split()) > 10)
        if is_multi and len(action_arg.split()) <= 5:
            return [DetectedProblem(
                skill_id="answer_completeness",
                problem_type="short_answer_for_multi_part",
                confidence=0.7,
                evidence=f"Multi-part question but answer is only {len(action_arg.split())} words",
                suggested_action="inject_text",
            )]

        return []


# ============================================================================
# Helper functions
# ============================================================================

    def _detect_citation_mismatch(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect proper nouns in FINAL answer not present in read documents."""
        if action_type != "FINAL" or not action_arg:
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read:
            return []

        # Extract proper nouns from answer
        answer_entities = _extract_entities_from_text(action_arg)
        if not answer_entities:
            return []

        all_read_lower = all_read.lower()
        unsupported = [e for e in answer_entities if e.lower() not in all_read_lower]

        if not unsupported:
            return []

        confidence = min(0.6 + 0.1 * len(unsupported), 0.85)
        return [DetectedProblem(
            skill_id="citation_mismatch",
            problem_type="unsupported_entities",
            confidence=confidence,
            evidence=f"Entities {unsupported[:3]} in answer not found in read documents",
            suggested_action="inject_text",
        )]

    def _detect_outdated_information(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect when question asks about current state but source is old."""
        if action_type != "FINAL":
            return []

        question = step_context.get("question", "")
        if not question:
            return []

        current_phrases = ["current", "latest", "now", "today", "present", "as of"]
        q_lower = question.lower()
        asks_current = any(p in q_lower for p in current_phrases)
        if not asks_current:
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read:
            return []

        doc_years = [int(y) for y in re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', all_read)]
        if not doc_years:
            return []

        max_year = max(doc_years)
        if max_year >= 2022:
            return []

        return [DetectedProblem(
            skill_id="outdated_information",
            problem_type="stale_source_for_current_question",
            confidence=0.7,
            evidence=f"Question asks about current state but newest source is from {max_year}",
            suggested_action="inject_text",
        )]

    def _detect_multi_hop_failure(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect multi-hop question answered without intermediate verification."""
        if action_type != "FINAL" or not thought:
            return []

        question = step_context.get("question", "")
        if not question:
            return []

        # Heuristic: multi-hop indicators
        multi_hop_patterns = [
            r'\b\w+ of the \w+ (?:of|that|who|which)\b',
            r'\b(?:born in|located in|directed by|written by|founded by)\b.*\b(?:the|a)\b',
        ]
        is_multi_hop = any(re.search(p, question, re.IGNORECASE) for p in multi_hop_patterns)
        if not is_multi_hop:
            return []

        # Check if reasoning shows step-by-step intermediate facts
        step_markers = ["step 1", "step 2", "first,", "then,", "next,", "therefore"]
        thought_lower = thought.lower()
        has_steps = sum(1 for m in step_markers if m in thought_lower)

        if has_steps >= 2:
            return []

        read_count = step_context.get("read_count", 0)
        confidence = 0.7 if read_count < 2 else 0.6

        return [DetectedProblem(
            skill_id="multi_hop_reasoning_failure",
            problem_type="no_intermediate_verification",
            confidence=confidence,
            evidence="Multi-hop question answered without explicit intermediate reasoning steps",
            suggested_action="inject_text",
        )]

    def _detect_language_barrier(
        self, thought: str, action_type: str, action_arg: str, step_context: Dict[str, Any],
    ) -> List[DetectedProblem]:
        """Detect non-ASCII content in sources that may cause transliteration issues."""
        if action_type != "FINAL":
            return []

        all_read = step_context.get("all_read_contents", "")
        if not all_read:
            return []

        # Check for significant non-ASCII content in sources
        non_ascii_count = sum(1 for c in all_read if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / len(all_read) if all_read else 0

        if non_ascii_ratio < 0.05:
            return []

        return [DetectedProblem(
            skill_id="language_barrier",
            problem_type="non_english_content",
            confidence=0.6,
            evidence=f"Source documents contain {non_ascii_ratio:.0%} non-ASCII characters",
            suggested_action="inject_text",
        )]

    # _detect_source_authority — REMOVED (86% false positive rate)


def _extract_specific_claims(text: str) -> List[str]:
    """Extract years (4-digit numbers), proper nouns (capitalized multi-word),
    and specific numbers from text.

    Returns a list of string claims suitable for cross-referencing.
    """
    if not text:
        return []

    claims = []

    # Years: 4-digit numbers (1000-2099)
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    claims.extend(years)

    # Specific numbers (2+ digits, not years)
    numbers = re.findall(r'\b(\d{2,})\b', text)
    for n in numbers:
        if n not in claims:
            claims.append(n)

    # Proper nouns: capitalized multi-word phrases
    proper_nouns = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    claims.extend(proper_nouns)

    return claims


def _extract_entities_from_text(text: str) -> List[str]:
    """Extract capitalized multi-word phrases (proper nouns) from text.

    Based on pattern from conditions.py:_read_has_multiple_entities, but
    without the lookbehind so entities after sentence boundaries are captured.
    Strips leading sentence-starter words (The, This, That, etc.) to avoid
    false matches like "The John Smith" instead of "John Smith".
    """
    if not text:
        return []
    proper_nouns = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    # Strip common sentence-starter words from the beginning of matches
    _STARTERS = {"The", "This", "That", "These", "Those", "Some", "Each",
                 "Every", "Many", "Most", "Both", "All", "Any", "Our", "His",
                 "Her", "Its", "Their", "My", "Your"}
    cleaned = []
    for phrase in proper_nouns:
        words = phrase.split()
        while words and words[0] in _STARTERS:
            words = words[1:]
        if len(words) >= 2:
            cleaned.append(" ".join(words))
    return list(set(cleaned))
