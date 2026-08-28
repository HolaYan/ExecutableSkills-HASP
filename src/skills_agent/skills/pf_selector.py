"""
PF Selector — uses PF helper to select top-K program functions per question.

Instead of statically disabling PFs via config, the PF helper dynamically
selects the most relevant PFs for each question. This ensures each question
gets the right interventions without attention dilution from irrelevant PFs.
"""

import re
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Brief descriptions of each PF for the PF helper prompt
PF_CATALOG: Dict[str, str] = {
    "insufficient_exploration": "Block FINAL if agent hasn't searched or read any documents",
    "retrieval_failure": "Shorten overly long search queries (>12 words) to improve results",
    "hallucination": "Block FINAL if answering without reading any source documents",
    "adversarial_distraction": "Warn when search results contain conflicting/contradictory claims",
    "temporal_confusion": "Check if years in answer are supported by read documents",
    "numerical_reasoning_error": "Check if numbers in answer are supported by read documents",
    "negation_oversight": "Remind about negation words (not/never/except) in question during FINAL",
    "citation_mismatch": "Check if named entities in answer exist in read documents",
    "outdated_information": "Warn when sources are old but question asks about current info",
    "format_extraction_error": "Clean answer formatting (strip prefixes, trailing references)",
    "wrong_entity_confusion": "Warn when search results contain similarly-named entities",
    "reading_comprehension_error": "Remind to read carefully when document has many entities/numbers",
    "multi_hop_reasoning_failure": "Block FINAL if multi-hop question answered without reading",
    "answer_completeness": "Check if multi-part questions have all parts addressed",
    "reasoning_error": "Detect contradictory reasoning in agent's thought process",
    "language_barrier": "Note when document contains non-English text",
    "decompose_complex_question": "Decompose multi-hop questions into focused sub-queries",
    "evidence_synthesis": "Verify key question entities are found in read documents",
    "comparison_analyzer": "Ensure both entities are researched for comparison questions",
    "query_decomposition": "Break complex search queries into focused sub-queries",
    "iterative_refinement": "Detect repeated similar searches and suggest different approach",
    "search_depth_controller": "Enforce minimum search/read depth for complex questions",
    "claim_triangulation": "Suggest cross-verification when answer is from single source",
    "misinformation_detector": "Detect contradictions across multiple read documents",
    "constraint_search": "Decompose multi-constraint questions into constraint-based searches",
    "search_result_reranker": "Rerank search results by relevance to question keywords",
    "relevant_content_extractor": "Extract question-relevant paragraphs from read content",
    "answer_confidence_guard": "Track candidate answers across steps and warn against over-refinement regressions",
}

_SELECTOR_PROMPT = """Select the most useful program functions (PFs) for a web search agent answering this question. Select at most {top_k}.

Question: {question}

Available PFs:
{pf_list}

Rules:
- These 3 PFs are ALWAYS required for ALL question types: format_extraction_error, retrieval_failure, relevant_content_extractor
- For computation/math/logic questions (calculate, ISBN, tables): select ONLY the 3 mandatory PFs
- For simple factoid questions (1 hop, short): select format_extraction_error + retrieval_failure + search_result_reranker + relevant_content_extractor. Do NOT add adversarial_distraction, reasoning_error, or decomposition PFs
- For moderate questions (2 hops): add adversarial_distraction, reasoning_error, answer_confidence_guard
- For multi-hop reasoning (3+ hops): add decompose_complex_question, insufficient_exploration, hallucination, answer_completeness
- Do NOT over-select — fewer well-targeted PFs is better than many irrelevant ones
- PFs that block FINAL (insufficient_exploration, hallucination) should only be selected for complex 3+ hop questions
- adversarial_distraction and reasoning_error should NOT be selected for simple questions — they cause false positive interventions

Respond with ONLY the PF IDs, one per line. No explanations."""


class PFSelector:
    """Selects top-K program functions per question using PF helper."""

    def __init__(self, teacher_model=None, top_k: int = 10,
                 all_pf_ids: Optional[List[str]] = None):
        """
        Args:
            teacher_model: APIModelWrapper for LLM-based selection. If None, heuristic.
            top_k: Max number of PFs to select.
            all_pf_ids: List of candidate PF IDs. Defaults to all in PF_CATALOG.
        """
        self._teacher = teacher_model
        self._top_k = top_k
        self._all_pf_ids = all_pf_ids or list(PF_CATALOG.keys())

    def select(self, question: str) -> List[str]:
        """Select top-K PFs for the given question."""
        if self._teacher is not None:
            return self._select_llm(question)
        return self._select_heuristic(question)

    def _select_llm(self, question: str) -> List[str]:
        """Use PF helper to select PFs."""
        try:
            pf_list = "\n".join(
                f"- {pid}: {PF_CATALOG.get(pid, 'No description')}"
                for pid in self._all_pf_ids
            )
            prompt = _SELECTOR_PROMPT.format(
                question=question, pf_list=pf_list, top_k=self._top_k,
            )
            response = self._teacher.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            selected = []
            for line in (response or "").strip().split("\n"):
                line = line.strip().strip("- ").strip()
                if line in self._all_pf_ids:
                    selected.append(line)
            if len(selected) < 3:
                logger.warning(
                    f"[PFSelector] LLM returned {len(selected)} PFs, "
                    f"falling back to heuristic"
                )
                return self._select_heuristic(question)
            # Ensure mandatory PFs are always included for all questions
            for mandatory in ["relevant_content_extractor", "retrieval_failure", "format_extraction_error"]:
                if mandatory not in selected:
                    selected.insert(0, mandatory)
            logger.info(f"[PFSelector] LLM selected {len(selected)} PFs: {selected}")
            return selected[:self._top_k]
        except Exception as e:
            logger.warning(f"[PFSelector] LLM selection failed: {e}")
            return self._select_heuristic(question)

    # ---- Computation / logic detection ----
    # These questions don't benefit from search-oriented PFs. Using PFs on
    # them causes regressions (GAIA #1 compute, #32 ISBN, #44 table logic).
    _COMPUTE_KEYWORDS = [
        "compute", "calculate", "check digit", "isbn", "checksum",
        "given this table", "given the table", "define *",
        "solve", "evaluate", "simplify",
    ]
    _COMPUTE_PATTERNS = [
        r'\|.*\|.*\|',          # Markdown table rows
        r'[+\-*/^=]{2,}',      # Math operators
        r'p-value',             # Statistical computation
        r'%\s*\d|mod\s*\d',    # Modular arithmetic
    ]

    def _is_computation_question(self, question: str) -> bool:
        """Detect computation/logic/math questions that PFs tend to hurt."""
        q_lower = question.lower()
        if any(kw in q_lower for kw in self._COMPUTE_KEYWORDS):
            return True
        if any(re.search(p, question) for p in self._COMPUTE_PATTERNS):
            return True
        return False

    # ---- Hop count estimation ----
    _COMPARISON_HOP_WORDS = {"first", "before", "after", "earlier", "later",
                             "older", "younger", "previous", "next", "preceding"}

    def _estimate_hop_count(self, question: str) -> int:
        """Estimate the number of reasoning hops needed for a question.

        Counts structural signals: possessives, relative clauses, "of the"
        phrases, sentence count, comparison words, and named entity count.
        Returns an integer estimate (1 = simple, 2 = low-hop, 3+ = complex).
        """
        q_lower = question.lower()
        hops = 1  # Every question requires at least 1 hop (the lookup)

        # Possessive chains ("X's Y")
        possessives = question.count("'s")
        hops += possessives

        # Relative clauses (exclude sentence-initial question words like "Who ..." or "Which ...")
        for w in ["whose", "which", "that", "who"]:
            # Count occurrences that are NOT at the very start of the question
            count = f" {q_lower} ".count(f" {w} ")
            # If the question starts with this word, subtract one
            if q_lower.startswith(f"{w} "):
                count -= 1
            hops += max(count, 0)

        # "of the" phrases (nested references)
        of_the = q_lower.count(" of the ")
        hops += of_the

        # Multiple sentences often indicate multi-hop
        sentences = [s.strip() for s in re.split(r'[.!?]', question) if s.strip()]
        if len(sentences) >= 3:
            hops += len(sentences) - 2

        # Comparison words (require looking up 2+ entities)
        # Use word boundary matching to handle punctuation after the word
        if any(re.search(r'\b' + w + r'\b', q_lower) for w in self._COMPARISON_HOP_WORDS):
            hops += 1

        # Multiple named entities (2+ multi-word proper nouns → likely multi-hop)
        entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        unique_entities = set(e.lower() for e in entities)
        if len(unique_entities) >= 3:
            hops += 1
        elif len(unique_entities) >= 2:
            # Two entities may still be a comparison/relation question
            pass

        return hops

    def _is_low_hop(self, question: str) -> bool:
        """True if estimated hops <= 2. These need search but not complex decomposition."""
        return self._estimate_hop_count(question) <= 2

    # ---- Simple factoid detection ----
    def _is_simple_factoid(self, question: str) -> bool:
        """Detect simple lookup questions where minimal PFs suffice."""
        words = question.split()
        q_lower = question.lower()
        # Short question with no multi-hop signals
        possessives = question.count("'s")
        relatives = sum(1 for w in ["whose", "which", "that", "who"]
                        if f" {w} " in f" {q_lower} ")
        of_the = q_lower.count(" of the ")
        hop_signal = possessives + relatives + of_the
        return len(words) < 15 and hop_signal == 0

    def _select_heuristic(self, question: str) -> List[str]:
        """Heuristic PF selection: question-type-aware, regression-safe.

        Design principles (learned from GAIA/FRAMES regression analysis):
        1. Computation/logic questions → minimal PFs (only answer cleanup)
        2. Simple factoids → safe PFs only (no search intervention)
        3. Multi-hop questions → add decomposition + depth PFs
        4. Never add PFs that encourage over-searching on non-complex questions
        """
        q_lower = question.lower()

        # ── Computation / logic → absolute minimal PFs ──
        # Regressions: GAIA #1 (Nature p-value), #32 (ISBN check digit),
        #              #44 (table logic) — PFs caused wrong answers
        if self._is_computation_question(question):
            result = ["format_extraction_error", "retrieval_failure", "relevant_content_extractor"]
            logger.info(
                f"[PFSelector] Computation question detected, minimal PFs: {result}"
            )
            return result

        # ── Simple factoid → light PFs, avoid reasoning/distraction noise ──
        if self._is_simple_factoid(question):
            result = [
                "format_extraction_error",
                "retrieval_failure",
                "search_result_reranker",
                "relevant_content_extractor",
            ]
            logger.info(
                f"[PFSelector] Simple factoid detected, light PFs: {result}"
            )
            return result

        # ── Low-hop questions (1-2 hops) → safe PFs, minimal intervention ──
        if self._is_low_hop(question):
            result = [
                "format_extraction_error",
                "adversarial_distraction",
                "reasoning_error",
                "retrieval_failure",
                "search_result_reranker",
                "relevant_content_extractor",
                "answer_confidence_guard",
            ]
            logger.info(
                f"[PFSelector] Low-hop question detected (hops={self._estimate_hop_count(question)}), "
                f"safe PFs: {result}"
            )
            return result

        # ── Standard questions (3+ hops): safe base + question-type-specific ──
        selected = [
            "format_extraction_error",
            "adversarial_distraction",
            "reasoning_error",
            "retrieval_failure",
            "search_result_reranker",
            "relevant_content_extractor",
            "answer_confidence_guard",
        ]

        # Multi-hop detection (3+ hops reach here)
        hop_count = self._estimate_hop_count(question)

        if hop_count >= 4:
            # Strong multi-hop: add decomposition + guard PFs
            selected.extend([
                "decompose_complex_question",
                "insufficient_exploration",
                "hallucination",
            ])
        elif hop_count >= 3:
            # Moderate multi-hop: decomposition helper + exploration guard
            selected.extend([
                "decompose_complex_question",
                "insufficient_exploration",
            ])

        # Comparison detection
        comparison_words = {"first", "before", "after", "earlier", "later",
                            "more", "less", "bigger", "smaller", "older", "younger"}
        if any(f" {w} " in f" {q_lower} " for w in comparison_words):
            selected.append("comparison_analyzer")

        # Negation detection
        if any(f" {w} " in f" {q_lower} " for w in ["not", "never", "except", "without"]):
            selected.append("negation_oversight")

        # Temporal detection (only for questions with explicit year references)
        year_matches = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', question)
        if len(year_matches) >= 2:
            # Multiple years → temporal confusion risk
            selected.append("temporal_confusion")

        # Multi-constraint (BrowseComp-style)
        constraint_phrases = ["born", "between", "joined", "attended",
                              "located", "founded", "published"]
        constraint_count = sum(1 for p in constraint_phrases if p in q_lower)
        if constraint_count >= 2:
            selected.append("constraint_search")

        # Entity-rich questions (2+ multi-word proper nouns)
        entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', question)
        if len(set(e.lower() for e in entities)) >= 3:
            selected.append("wrong_entity_confusion")

        # Current/recent info
        if any(w in q_lower for w in ["current", "currently", "now", "latest", "recent"]):
            selected.append("outdated_information")

        # Deduplicate and limit
        seen = set()
        unique = []
        for pf_id in selected:
            if pf_id not in seen and pf_id in self._all_pf_ids:
                seen.add(pf_id)
                unique.append(pf_id)

        result = unique[:self._top_k]
        logger.info(f"[PFSelector] Heuristic selected {len(result)} PFs: {result}")
        return result
