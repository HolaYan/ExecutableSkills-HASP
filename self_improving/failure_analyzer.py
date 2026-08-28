"""
Failure analyzer — cluster recurring failures and identify patterns not covered
by existing skills.

Analyzes PF-aware trajectories to find:
  1. Recurring residual failures (seed skills didn't help)
  2. Missed intervention opportunities (PFs should have fired but didn't)
  3. Harmful interventions (PFs fired but made things worse)
  4. Common failure patterns that could be abstracted into new skills
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from training.signals.trajectory import EpisodeTrajectory, StepRecord

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """A recurring failure pattern extracted from trajectories."""
    pattern_id: str
    category: str  # e.g., "premature_final", "query_failure", "reasoning_error", ...
    description: str
    # Evidence: sample_ids exhibiting this pattern
    evidence_ids: List[str] = field(default_factory=list)
    # Representative trajectory excerpts
    representative_steps: List[Dict[str, Any]] = field(default_factory=list)
    # How many times this pattern occurred
    frequency: int = 0
    # Which existing PFs were active but didn't help
    ineffective_pf_ids: List[str] = field(default_factory=list)
    # Severity: how much EM/F1 was lost
    avg_f1_loss: float = 0.0
    # Whether existing skills cover this pattern
    covered_by_existing: bool = False
    # Novelty: is this genuinely new or a variant of an existing skill?
    novelty_score: float = 0.0
    # LLM-abstracted summary {abstraction, trigger, intervention_hint}
    # when the pattern came from the LLM failure summarizer.
    llm_summary: Optional[Dict[str, str]] = None


@dataclass
class FailureCluster:
    """A cluster of similar failures ready for skill proposal."""
    cluster_id: str
    patterns: List[FailurePattern] = field(default_factory=list)
    total_frequency: int = 0
    avg_severity: float = 0.0
    # Suggested skill category (existing or new)
    suggested_category: Optional[str] = None
    is_new_category: bool = False


class FailureAnalyzer:
    """Analyzes failed trajectories to discover skill-worthy failure patterns."""

    # Heuristic pattern detectors
    PATTERN_DETECTORS = {
        "premature_final": "_detect_premature_final",
        "repeated_search": "_detect_repeated_search",
        "no_read_before_final": "_detect_no_read",
        "query_too_broad": "_detect_broad_query",
        "query_too_narrow": "_detect_narrow_query",
        "wrong_entity_focus": "_detect_wrong_entity",
        "reasoning_hallucination": "_detect_hallucination",
        "format_mismatch": "_detect_format_mismatch",
        "partial_answer": "_detect_partial_answer",
        "contradictory_evidence_ignored": "_detect_contradiction_ignored",
        "excessive_steps_no_progress": "_detect_no_progress",
        "pf_override_harmful": "_detect_harmful_override",
    }

    # Existing skill categories for overlap checking
    EXISTING_CATEGORIES = {
        "exploration_control": [
            "insufficient_exploration", "retrieval_failure", "search_depth_controller",
        ],
        "reasoning_guard": [
            "hallucination", "reasoning_error", "multi_hop_reasoning_failure",
        ],
        "entity_verification": [
            "wrong_entity_confusion", "temporal_confusion", "numerical_reasoning_error",
        ],
        "information_synthesis": [
            "claim_triangulation", "evidence_synthesis", "misinformation_detector",
        ],
        "query_strategy": [
            "query_decomposition", "constraint_search", "iterative_refinement",
        ],
        "format_output": [
            "format_extraction_error", "answer_completeness", "citation_mismatch",
        ],
    }

    def __init__(
        self,
        existing_skill_ids: List[str],
        min_cluster_size: int = 3,
        output_dir: Optional[str] = None,
        mode: str = "heuristic",
        teacher_model: Optional[Any] = None,
        llm_concurrency: int = 8,
        llm_dedup_threshold: float = 0.5,
    ):
        self.existing_skill_ids = set(existing_skill_ids)
        self.min_cluster_size = min_cluster_size
        self.output_dir = Path(output_dir) if output_dir else None
        self._patterns: List[FailurePattern] = []
        self._clusters: List[FailureCluster] = []
        self.mode = mode  # "heuristic" | "llm" | "both"
        self.teacher_model = teacher_model
        self.llm_concurrency = llm_concurrency
        self.llm_dedup_threshold = llm_dedup_threshold

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(self, trajectories: List[EpisodeTrajectory]) -> List[FailureCluster]:
        """Analyze failed trajectories and return clusters for skill proposal."""
        failed = [t for t in trajectories if not t.exact_match]
        logger.info("Analyzing %d failed trajectories (out of %d total)",
                     len(failed), len(trajectories))

        if not failed:
            return []

        # Step 1: Detect patterns (heuristic / LLM / both)
        all_patterns: List[FailurePattern] = []
        if self.mode in ("heuristic", "both"):
            for traj in failed:
                all_patterns.extend(self._detect_patterns(traj))
        if self.mode in ("llm", "both") and self.teacher_model is not None:
            try:
                llm_patterns = self._llm_summarize_failures(failed)
                all_patterns.extend(llm_patterns)
                logger.info("LLM summarizer produced %d patterns", len(llm_patterns))
            except Exception as e:
                logger.warning("LLM failure summarizer failed: %s — keeping heuristic patterns only", e)

        # Step 2: Aggregate patterns by category
        category_patterns = defaultdict(list)
        for p in all_patterns:
            category_patterns[p.category].append(p)

        # Step 3: Merge patterns within each category
        merged_patterns = []
        for category, patterns in category_patterns.items():
            merged = self._merge_patterns(category, patterns)
            merged_patterns.extend(merged)

        # Step 4: Check novelty (overlap with existing skills)
        for p in merged_patterns:
            p.novelty_score = self._compute_novelty(p)
            p.covered_by_existing = p.novelty_score < 0.3

        self._patterns = merged_patterns

        # Step 5: Cluster patterns and filter by frequency
        self._clusters = self._build_clusters(merged_patterns)

        logger.info("Found %d patterns, %d clusters (min size %d)",
                     len(merged_patterns), len(self._clusters), self.min_cluster_size)

        if self.output_dir:
            self._save_analysis()

        return self._clusters

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def _detect_patterns(self, traj: EpisodeTrajectory) -> List[FailurePattern]:
        """Run all heuristic detectors on a single trajectory."""
        detected = []
        for pattern_name, method_name in self.PATTERN_DETECTORS.items():
            method = getattr(self, method_name)
            result = method(traj)
            if result:
                result.evidence_ids.append(traj.sample_id)
                detected.append(result)
        return detected

    def _detect_premature_final(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent issued FINAL with very few steps (< 3) or no search."""
        if len(traj.steps) < 3:
            return FailurePattern(
                pattern_id=f"premature_final_{traj.sample_id}",
                category="premature_final",
                description=f"Agent gave FINAL after only {len(traj.steps)} steps",
                frequency=1,
                representative_steps=[{
                    "total_steps": len(traj.steps),
                    "final_answer": traj.final_answer[:200],
                }],
            )
        return None

    def _detect_repeated_search(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent searched with same/similar query multiple times."""
        search_queries = []
        for step in traj.steps:
            if step.final_action_type == "SEARCH":
                search_queries.append(step.final_action_arg.lower().strip())

        if len(search_queries) < 2:
            return None

        # Check for near-duplicates
        duplicates = 0
        for i in range(len(search_queries)):
            for j in range(i + 1, len(search_queries)):
                if self._query_similarity(search_queries[i], search_queries[j]) > 0.8:
                    duplicates += 1

        if duplicates >= 2:
            return FailurePattern(
                pattern_id=f"repeated_search_{traj.sample_id}",
                category="repeated_search",
                description=f"Agent repeated similar searches {duplicates} times",
                frequency=1,
                representative_steps=[{"queries": search_queries[:5]}],
            )
        return None

    def _detect_no_read(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent never used READ action before giving FINAL."""
        has_read = any(s.final_action_type == "READ" for s in traj.steps)
        if not has_read and len(traj.steps) >= 2:
            return FailurePattern(
                pattern_id=f"no_read_{traj.sample_id}",
                category="no_read_before_final",
                description="Agent never read any document before answering",
                frequency=1,
            )
        return None

    def _detect_broad_query(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent used overly broad search queries."""
        broad_queries = []
        for step in traj.steps:
            if step.final_action_type == "SEARCH":
                q = step.final_action_arg
                # Very short queries (1-2 words) or just the raw question
                words = q.split()
                if len(words) <= 2 or len(q) > 150:
                    broad_queries.append(q)

        if broad_queries:
            return FailurePattern(
                pattern_id=f"broad_query_{traj.sample_id}",
                category="query_too_broad",
                description=f"Agent used {len(broad_queries)} overly broad/long queries",
                frequency=1,
                representative_steps=[{"queries": broad_queries[:3]}],
            )
        return None

    def _detect_narrow_query(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent used overly specific queries that returned no results."""
        narrow = []
        for step in traj.steps:
            if step.final_action_type == "SEARCH":
                ctx = step.step_context_snapshot
                if ctx.get("empty_results", False):
                    narrow.append(step.final_action_arg)

        if len(narrow) >= 2:
            return FailurePattern(
                pattern_id=f"narrow_query_{traj.sample_id}",
                category="query_too_narrow",
                description=f"Agent had {len(narrow)} searches with empty results",
                frequency=1,
                representative_steps=[{"failed_queries": narrow[:3]}],
            )
        return None

    def _detect_wrong_entity(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent's final answer mentions an entity not in the gold answers."""
        if not traj.gold_answers:
            return None
        gold_lower = " ".join(traj.gold_answers).lower()
        answer_lower = traj.final_answer.lower()
        # If answer has a proper noun not in gold, might be wrong entity
        if answer_lower and answer_lower not in gold_lower and len(answer_lower) > 3:
            # Simple heuristic: check if answer tokens overlap with gold
            answer_tokens = set(answer_lower.split())
            gold_tokens = set(gold_lower.split())
            overlap = answer_tokens & gold_tokens
            if len(overlap) < len(answer_tokens) * 0.3:
                return FailurePattern(
                    pattern_id=f"wrong_entity_{traj.sample_id}",
                    category="wrong_entity_focus",
                    description="Agent's answer has low overlap with gold answer tokens",
                    frequency=1,
                    representative_steps=[{
                        "answer": traj.final_answer[:100],
                        "gold": traj.gold_answers[:3],
                    }],
                )
        return None

    def _detect_hallucination(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Agent produced answer not grounded in any read content."""
        read_content = ""
        for step in traj.steps:
            if step.final_action_type == "READ":
                read_content += " " + step.observation_summary

        if not read_content.strip() and traj.final_answer:
            return FailurePattern(
                pattern_id=f"hallucination_{traj.sample_id}",
                category="reasoning_hallucination",
                description="Agent answered without reading any content",
                frequency=1,
            )

        if read_content and traj.final_answer:
            answer_lower = traj.final_answer.lower()
            content_lower = read_content.lower()
            # Check if key answer tokens appear in read content
            answer_tokens = set(answer_lower.split()) - {"the", "a", "an", "of", "in", "is", "was"}
            grounded = sum(1 for t in answer_tokens if t in content_lower)
            if answer_tokens and grounded / len(answer_tokens) < 0.3:
                return FailurePattern(
                    pattern_id=f"hallucination_{traj.sample_id}",
                    category="reasoning_hallucination",
                    description="Answer poorly grounded in read content",
                    frequency=1,
                    representative_steps=[{
                        "answer": traj.final_answer[:100],
                        "grounding_ratio": grounded / max(len(answer_tokens), 1),
                    }],
                )
        return None

    def _detect_format_mismatch(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Answer format doesn't match expected format."""
        if not traj.gold_answers or not traj.final_answer:
            return None
        gold = traj.gold_answers[0]
        answer = traj.final_answer

        # Check obvious format mismatches
        issues = []
        if gold.replace(",", "").isdigit() and not answer.replace(",", "").replace(".", "").isdigit():
            issues.append("expected_number_got_text")
        if len(answer) > 5 * len(gold) and len(gold) < 50:
            issues.append("answer_much_longer_than_expected")
        if "\n" in answer and "\n" not in gold:
            issues.append("unexpected_multiline")

        if issues:
            return FailurePattern(
                pattern_id=f"format_{traj.sample_id}",
                category="format_mismatch",
                description=f"Format issues: {', '.join(issues)}",
                frequency=1,
                representative_steps=[{
                    "answer": answer[:100],
                    "gold": gold[:100],
                    "issues": issues,
                }],
            )
        return None

    def _detect_partial_answer(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Answer is partially correct but incomplete."""
        if not traj.gold_answers:
            return None
        gold_lower = traj.gold_answers[0].lower()
        answer_lower = traj.final_answer.lower()

        # Partial overlap: answer contains part of gold or vice versa
        if (answer_lower in gold_lower and len(answer_lower) > 3) or \
           (gold_lower in answer_lower and len(gold_lower) > 3):
            if answer_lower != gold_lower:
                return FailurePattern(
                    pattern_id=f"partial_{traj.sample_id}",
                    category="partial_answer",
                    description="Answer is a substring of gold or vice versa",
                    frequency=1,
                    representative_steps=[{
                        "answer": traj.final_answer[:100],
                        "gold": traj.gold_answers[0][:100],
                    }],
                )
        return None

    def _detect_contradiction_ignored(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Contradictory sources were present but agent didn't verify."""
        for step in traj.steps:
            ctx = step.step_context_snapshot
            if ctx.get("contradictory_sources", False):
                # Check if agent did verification after
                subsequent_reads = sum(
                    1 for s in traj.steps[step.step_index:]
                    if s.final_action_type == "READ"
                )
                if subsequent_reads < 2:
                    return FailurePattern(
                        pattern_id=f"contradiction_{traj.sample_id}",
                        category="contradictory_evidence_ignored",
                        description="Contradictory sources found but insufficient verification",
                        frequency=1,
                    )
        return None

    def _detect_no_progress(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """Many steps taken but no meaningful progress toward answer."""
        if len(traj.steps) >= 15:
            unique_actions = set()
            for s in traj.steps:
                unique_actions.add((s.final_action_type, s.final_action_arg[:50]))
            if len(unique_actions) < len(traj.steps) * 0.5:
                return FailurePattern(
                    pattern_id=f"no_progress_{traj.sample_id}",
                    category="excessive_steps_no_progress",
                    description=f"{len(traj.steps)} steps but only {len(unique_actions)} unique actions",
                    frequency=1,
                )
        return None

    def _detect_harmful_override(self, traj: EpisodeTrajectory) -> Optional[FailurePattern]:
        """PF modified an action but the episode still failed — harmful intervention."""
        harmful = []
        for step in traj.steps:
            if step.was_modified:
                harmful.append({
                    "step": step.step_index,
                    "original": f"{step.proposed_action_type}({step.proposed_action_arg[:50]})",
                    "modified": f"{step.final_action_type}({step.final_action_arg[:50]})",
                    "pf_ids": [a.pf_id for a in step.pf_activations if a.activated],
                })

        if harmful:
            return FailurePattern(
                pattern_id=f"harmful_override_{traj.sample_id}",
                category="pf_override_harmful",
                description=f"PF modified {len(harmful)} actions but episode still failed",
                frequency=1,
                ineffective_pf_ids=[
                    pf_id for h in harmful for pf_id in h["pf_ids"]
                ],
                representative_steps=harmful[:3],
            )
        return None

    # ------------------------------------------------------------------
    # Merging and clustering
    # ------------------------------------------------------------------

    def _merge_patterns(self, category: str, patterns: List[FailurePattern]) -> List[FailurePattern]:
        """Merge multiple pattern instances of the same category into one."""
        if not patterns:
            return []

        merged = FailurePattern(
            pattern_id=f"merged_{category}",
            category=category,
            description=patterns[0].description,
            evidence_ids=[eid for p in patterns for eid in p.evidence_ids],
            frequency=len(patterns),
            representative_steps=patterns[0].representative_steps[:3],
            ineffective_pf_ids=list(set(
                pid for p in patterns for pid in p.ineffective_pf_ids
            )),
        )
        return [merged]

    def _build_clusters(self, patterns: List[FailurePattern]) -> List[FailureCluster]:
        """Group patterns into clusters and filter by minimum size."""
        # Map patterns to broader categories
        category_map = {
            "premature_final": "exploration_control",
            "no_read_before_final": "exploration_control",
            "repeated_search": "query_strategy",
            "query_too_broad": "query_strategy",
            "query_too_narrow": "query_strategy",
            "wrong_entity_focus": "entity_verification",
            "reasoning_hallucination": "reasoning_guard",
            "format_mismatch": "format_output",
            "partial_answer": "format_output",
            "contradictory_evidence_ignored": "information_synthesis",
            "excessive_steps_no_progress": "exploration_control",
            "pf_override_harmful": "meta_pf_quality",
        }

        cluster_groups = defaultdict(list)
        for p in patterns:
            # Heuristic categories get remapped to broad buckets; LLM-summarized
            # categories pass through unchanged so each distinct LLM category
            # forms its own cluster.
            broad_cat = category_map.get(p.category, p.category)
            cluster_groups[broad_cat].append(p)

        clusters = []
        for cat, cat_patterns in cluster_groups.items():
            total_freq = sum(p.frequency for p in cat_patterns)
            if total_freq < self.min_cluster_size:
                continue

            is_new = cat not in self.EXISTING_CATEGORIES
            cluster = FailureCluster(
                cluster_id=f"cluster_{cat}",
                patterns=cat_patterns,
                total_frequency=total_freq,
                avg_severity=1.0 - sum(
                    p.avg_f1_loss for p in cat_patterns
                ) / max(len(cat_patterns), 1),
                suggested_category=cat,
                is_new_category=is_new,
            )
            clusters.append(cluster)

        # Sort by frequency (most common failures first)
        clusters.sort(key=lambda c: c.total_frequency, reverse=True)
        return clusters

    def _compute_novelty(self, pattern: FailurePattern) -> float:
        """Compute novelty score: 1.0 = entirely new, 0.0 = fully covered."""
        # Check if pattern's category maps to an existing skill group
        for group_name, skill_ids in self.EXISTING_CATEGORIES.items():
            overlap = set(skill_ids) & self.existing_skill_ids
            if overlap:
                # Check if the pattern's ineffective PFs overlap with this group
                group_ineffective = set(pattern.ineffective_pf_ids) & overlap
                if group_ineffective:
                    # Skills exist for this but they didn't help — partial novelty
                    return 0.5
                # If no PFs from this group were even selected, might be a gap
                # But the category is covered conceptually
                category_map = {
                    "premature_final": "exploration_control",
                    "no_read_before_final": "exploration_control",
                    "repeated_search": "query_strategy",
                    "query_too_broad": "query_strategy",
                    "query_too_narrow": "query_strategy",
                    "wrong_entity_focus": "entity_verification",
                    "reasoning_hallucination": "reasoning_guard",
                    "format_mismatch": "format_output",
                    "partial_answer": "format_output",
                    "contradictory_evidence_ignored": "information_synthesis",
                    "excessive_steps_no_progress": "exploration_control",
                }
                if category_map.get(pattern.category) == group_name:
                    return 0.3  # Low novelty, existing coverage
        return 0.8  # High novelty — not covered

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _query_similarity(q1: str, q2: str) -> float:
        """Simple token-level Jaccard similarity between two queries."""
        t1 = set(q1.split())
        t2 = set(q2.split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    def _save_analysis(self) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save patterns
        patterns_path = self.output_dir / "failure_patterns.json"
        with open(patterns_path, "w") as f:
            json.dump([{
                "pattern_id": p.pattern_id,
                "category": p.category,
                "description": p.description,
                "frequency": p.frequency,
                "novelty_score": p.novelty_score,
                "covered_by_existing": p.covered_by_existing,
                "evidence_count": len(p.evidence_ids),
                "ineffective_pf_ids": p.ineffective_pf_ids,
            } for p in self._patterns], f, indent=2)

        # Save clusters
        clusters_path = self.output_dir / "failure_clusters.json"
        with open(clusters_path, "w") as f:
            json.dump([{
                "cluster_id": c.cluster_id,
                "suggested_category": c.suggested_category,
                "is_new_category": c.is_new_category,
                "total_frequency": c.total_frequency,
                "avg_severity": c.avg_severity,
                "pattern_categories": [p.category for p in c.patterns],
            } for c in self._clusters], f, indent=2)

    # ------------------------------------------------------------------
    # LLM failure summarizer
    # ------------------------------------------------------------------

    _LLM_SYSTEM = (
        "You are an expert failure analyst for a ReAct web-search agent. "
        "For each failed QA trajectory you must produce a concise, reusable "
        "abstraction of WHY the agent failed so that a new Program Function "
        "(PF) skill can be authored to fix this class of failure."
    )

    _LLM_USER_TMPL = (
        "Question:\n{question}\n\n"
        "Gold answer(s): {gold}\n\n"
        "Agent's final answer: {pred}\n\n"
        "Trajectory (truncated):\n{trace}\n\n"
        "Return STRICT JSON with keys:\n"
        "  category: short snake_case category id (4-30 chars, e.g. "
        "'constraint_leak', 'premature_summary', 'stale_evidence'). Reuse "
        "an existing id if the failure mode matches; otherwise coin a new one.\n"
        "  abstraction: 1-sentence description of the recurring failure mode.\n"
        "  trigger: when in a ReAct step this failure manifests (observable signal).\n"
        "  intervention_hint: what a PF should do to prevent this (action/context edit).\n"
        "Output ONLY the JSON object."
    )

    @staticmethod
    def _summarize_trace(traj: EpisodeTrajectory, max_chars: int = 1800) -> str:
        lines = []
        for step in getattr(traj, "steps", []) or []:
            thought = getattr(step, "proposed_reasoning", "") or ""
            act_t = getattr(step, "final_action_type", getattr(step, "proposed_action_type", ""))
            act_a = getattr(step, "final_action_arg", getattr(step, "proposed_action_arg", ""))
            lines.append(f"[{getattr(step, 'step_index', 0)}] {act_t}({str(act_a)[:120]}) :: {thought[:160]}")
        txt = "\n".join(lines)
        return txt[:max_chars] + ("…" if len(txt) > max_chars else "")

    def _llm_summarize_failures(
        self, failed: List[EpisodeTrajectory],
    ) -> List[FailurePattern]:
        """Ask the PF helper to abstract each failure; dedup by category +
        Jaccard similarity over abstraction text. Returns `FailurePattern`s."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json as _json

        def _one(traj: EpisodeTrajectory) -> Optional[Dict[str, Any]]:
            prompt = self._LLM_USER_TMPL.format(
                question=traj.question[:500],
                gold=", ".join(traj.gold_answers[:3]) or "[none]",
                pred=(traj.final_answer or "")[:200],
                trace=self._summarize_trace(traj),
            )
            try:
                raw = self.teacher_model.generate(
                    prompt=prompt, system=self._LLM_SYSTEM,
                    temperature=0.3, max_tokens=400,
                )
            except Exception as e:
                logger.debug("LLM summarize failed for %s: %s", getattr(traj, "sample_id", "?"), e)
                return None
            # Be lenient about stray prose around JSON
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if not m:
                return None
            try:
                obj = _json.loads(m.group(0))
            except Exception:
                return None
            if not isinstance(obj, dict) or "category" not in obj:
                return None
            obj["sample_id"] = getattr(traj, "sample_id", "")
            obj["question"] = traj.question
            return obj

        summaries: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.llm_concurrency) as pool:
            for res in pool.map(_one, failed):
                if res:
                    summaries.append(res)

        # Dedup: group by (normalized category) → patterns per group
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in summaries:
            cat = re.sub(r"[^a-z0-9_]+", "_", str(s.get("category", "")).lower()).strip("_")
            if not cat:
                continue
            # Also fold near-duplicate abstractions within one category
            placed = False
            for members in [groups[cat]]:
                for m in members:
                    if self._query_similarity(
                        m.get("abstraction", ""), s.get("abstraction", "")
                    ) >= self.llm_dedup_threshold:
                        m.setdefault("_evidence", []).append(s.get("sample_id", ""))
                        m["_freq"] = m.get("_freq", 1) + 1
                        placed = True
                        break
            if not placed:
                s["_evidence"] = [s.get("sample_id", "")]
                s["_freq"] = 1
                groups[cat].append(s)

        patterns: List[FailurePattern] = []
        idx = 0
        for cat, members in groups.items():
            for m in members:
                idx += 1
                patterns.append(FailurePattern(
                    pattern_id=f"llm_{cat}_{idx:03d}",
                    category=cat,
                    description=str(m.get("abstraction", ""))[:400],
                    evidence_ids=[e for e in m.get("_evidence", []) if e],
                    frequency=int(m.get("_freq", 1)),
                    avg_f1_loss=1.0,
                    covered_by_existing=False,
                    novelty_score=0.9 if cat not in self.existing_skill_ids else 0.5,
                    llm_summary={
                        "abstraction": str(m.get("abstraction", "")),
                        "trigger": str(m.get("trigger", "")),
                        "intervention_hint": str(m.get("intervention_hint", "")),
                    },
                ))
        return patterns

    @property
    def patterns(self) -> List[FailurePattern]:
        return self._patterns

    @property
    def clusters(self) -> List[FailureCluster]:
        return self._clusters
