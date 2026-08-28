"""
SkillSelector — selects the most relevant skills for a given context.

Two levels of selection:
1. Episode-level: Select top-K skills for system prompt injection (based on mode, question, priority)
2. Step-level: Select a single skill reminder for the current step (based on step context)
"""

from typing import Optional, List, Dict, Any, Tuple
import re
import logging

from .skill import Skill, SkillLibrary, PhaseInstruction
from .conditions import ConditionEvaluator

logger = logging.getLogger(__name__)


class SkillSelector:
    """Selects relevant skills based on multi-signal weighted scoring."""

    def __init__(
        self,
        library: SkillLibrary,
        mode_weight: float = 0.5,
        trigger_weight: float = 0.5,
    ):
        self.library = library
        self.mode_weight = mode_weight
        self.trigger_weight = trigger_weight

    def select(
        self,
        question: str,
        mode: str,
        max_skills: int = 3,
    ) -> List[Skill]:
        """
        Episode-level selection: pick top-K skills for system prompt injection.

        Args:
            question: The question being asked
            mode: Evaluation mode (clean, adv_conflict_l1, etc.)
            max_skills: Maximum number of skills to return

        Returns:
            List of selected skills, sorted by relevance score descending
        """
        scores: Dict[str, float] = {}

        for skill in self.library.get_all():
            score = (
                self.mode_weight * self._mode_match(mode, skill.applicable_modes)
                + self.trigger_weight
                * self._trigger_keyword_match(question, skill.detection_triggers)
            )
            scores[skill.skill_id] = score

        # Sort by score descending, take top K
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        selected = []
        for skill_id in sorted_ids[:max_skills]:
            skill = self.library.get(skill_id)
            if skill is not None:
                selected.append(skill)

        logger.debug(
            f"Selected {len(selected)} skills for mode={mode}: "
            f"{[s.skill_id for s in selected]}"
        )
        return selected

    def select_for_step(self, step_context: Dict[str, Any]) -> Optional[Skill]:
        """
        Step-level selection: pick a single skill reminder for the current step.

        Args:
            step_context: Dictionary with current step information:
                - contradictory_sources: bool — search results contain conflicting info
                - step_count: int — current step number
                - has_read: bool — whether READ has been used
                - empty_results: bool — last search returned empty
                - search_count: int — number of searches performed
                - action_type: str — last action type
                - observation_text: str — last observation text

        Returns:
            A single Skill to use as a reminder, or None
        """
        # Rule 1: Contradictory sources → adversarial distraction
        if step_context.get("contradictory_sources", False):
            skill = self.library.get("adversarial_distraction")
            if skill:
                return skill

        # Rule 2: Many steps without READ → insufficient exploration
        step_count = step_context.get("step_count", 0)
        has_read = step_context.get("has_read", False)
        if step_count >= 5 and not has_read:
            skill = self.library.get("insufficient_exploration")
            if skill:
                return skill

        # Rule 3: Empty search results → retrieval failure
        if step_context.get("empty_results", False):
            skill = self.library.get("retrieval_failure")
            if skill:
                return skill

        # Rule 4: About to answer (high step count) without much exploration
        search_count = step_context.get("search_count", 0)
        if step_count >= 3 and search_count <= 1 and not has_read:
            skill = self.library.get("insufficient_exploration")
            if skill:
                return skill

        # Rule 5: Multiple searches for similar entities → entity confusion risk
        if step_context.get("similar_entity_results", False):
            skill = self.library.get("wrong_entity_confusion")
            if skill:
                return skill

        return None

    def select_for_phase(
        self,
        phase: str,
        active_skills: List[Skill],
        context: Dict[str, Any],
        max_instructions: int = 1,
    ) -> List[Tuple[Skill, PhaseInstruction]]:
        """Phase-gated selection: pick instructions for a specific phase.

        For each active skill, checks if it has a phase_instruction for the
        given phase, evaluates conditions, and returns top matches ranked by
        priority + priority_boost.

        Args:
            phase: Phase name ("post_search", "post_read", "pre_final")
            active_skills: Skills active for this episode
            context: Observation context dict for condition evaluation
            max_instructions: Max instructions to return

        Returns:
            List of (Skill, PhaseInstruction) tuples, sorted by score descending.
        """
        candidates: List[Tuple[float, Skill, PhaseInstruction]] = []

        for skill in active_skills:
            pi = skill.phase_instructions.get(phase)
            if pi is None:
                continue

            # Evaluate conditions (AND logic, empty → always True)
            if not ConditionEvaluator.evaluate(pi.conditions, context):
                continue

            score = pi.priority_boost
            candidates.append((score, skill, pi))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        result = [(skill, pi) for _, skill, pi in candidates[:max_instructions]]

        if result:
            logger.debug(
                f"Phase '{phase}': selected {[s.skill_id for s, _ in result]}"
            )

        return result

    @staticmethod
    def _mode_match(mode: str, applicable_modes: List[str]) -> float:
        """Score how well the current mode matches the skill's applicable modes."""
        if "all" in applicable_modes:
            return 0.5  # Moderate match for universal skills

        if mode in applicable_modes:
            return 1.0

        # Partial match: same mode family (e.g., adv_conflict_l1 matches adv_*)
        mode_prefix = mode.split("_")[0] if "_" in mode else mode
        for am in applicable_modes:
            if am.startswith(mode_prefix):
                return 0.7

        return 0.0

    @staticmethod
    def _trigger_keyword_match(
        question: str, detection_triggers: List[str]
    ) -> float:
        """Score keyword overlap between question and detection triggers."""
        if not question or not detection_triggers:
            return 0.0

        question_lower = question.lower()
        question_words = set(re.findall(r"\w+", question_lower))

        total_score = 0.0
        for trigger in detection_triggers:
            trigger_words = set(re.findall(r"\w+", trigger.lower()))
            if not trigger_words:
                continue
            overlap = len(question_words & trigger_words)
            # Normalize by trigger length
            trigger_score = overlap / len(trigger_words)
            total_score = max(total_score, trigger_score)

        return min(total_score, 1.0)

