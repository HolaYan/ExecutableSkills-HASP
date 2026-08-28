"""
SkillEpisode — extends Episode with skill tracking information.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json

from .episode import Episode, Step, Action, Observation, Evidence, AttackMetadata


@dataclass
class SkillEpisode(Episode):
    """
    Episode with skill usage tracking.

    Extends Episode with:
    - active_skill_ids: Skills injected into the system prompt
    - per_step_skill_reminders: Step-level reminders that were added
    """

    active_skill_ids: List[str] = field(default_factory=list)
    per_step_skill_reminders: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["active_skill_ids"] = self.active_skill_ids
        result["per_step_skill_reminders"] = self.per_step_skill_reminders
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEpisode":
        # Build base Episode fields
        trace = [Step.from_dict(s) for s in data.get("trace", [])]
        evidence = [Evidence.from_dict(e) for e in data.get("evidence", [])]

        attack_metadata = None
        if data.get("attack_metadata"):
            attack_metadata = AttackMetadata.from_dict(data["attack_metadata"])

        return cls(
            question=data.get("question", ""),
            trace=trace,
            final=data.get("final"),
            evidence=evidence,
            sample_id=data.get("sample_id"),
            seed=data.get("seed"),
            mode=data.get("mode"),
            model=data.get("model"),
            attack_metadata=attack_metadata,
            gold_answers=data.get("gold_answers", []),
            active_skill_ids=data.get("active_skill_ids", []),
            per_step_skill_reminders=data.get("per_step_skill_reminders", []),
        )

    @classmethod
    def from_episode(
        cls,
        episode: Episode,
        active_skill_ids: Optional[List[str]] = None,
        per_step_skill_reminders: Optional[List[Optional[str]]] = None,
    ) -> "SkillEpisode":
        """Convert a regular Episode to a SkillEpisode."""
        return cls(
            question=episode.question,
            trace=episode.trace,
            final=episode.final,
            evidence=episode.evidence,
            sample_id=episode.sample_id,
            seed=episode.seed,
            mode=episode.mode,
            model=episode.model,
            attack_metadata=episode.attack_metadata,
            gold_answers=episode.gold_answers,
            active_skill_ids=active_skill_ids or [],
            per_step_skill_reminders=per_step_skill_reminders or [],
        )
