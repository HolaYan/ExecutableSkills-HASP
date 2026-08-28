"""
PF-aware trajectory logger.

Records step-level detail for every episode:
  - Student's proposed action (before PF intervention)
  - PF activations and their interventions
  - Action after PF modification
  - Local outcome (observation quality)
  - PF helper skill selection for this episode
  - Final episode outcome

This data feeds into failure analysis, pseudo-gradient computation,
and training data generation.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)





# The record types moved to training/signals/trajectory.py, next to the signals
# that read them, so neither package has to import the other for a type. They
# are re-exported here because this module's own API is built on them.
from training.signals.trajectory import (  # noqa: E402,F401
    EpisodeTrajectory, PFActivationRecord, StepRecord,
)


class TrajectoryLogger:
    """Collects and persists PF-aware trajectories across an epoch."""

    def __init__(self, output_dir: str, epoch: int = 0):
        self.output_dir = Path(output_dir) / f"epoch_{epoch}" / "trajectories"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epoch = epoch
        self._trajectories: List[EpisodeTrajectory] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def new_episode(
        self,
        sample_id: str,
        question: str,
        gold_answers: List[str],
        dataset_name: str = "",
        difficulty_score: int = 0,
        skills_enabled: bool = True,
        selected_pf_ids: Optional[List[str]] = None,
    ) -> EpisodeTrajectory:
        """Create a new trajectory for an episode."""
        traj = EpisodeTrajectory(
            sample_id=sample_id,
            question=question,
            gold_answers=gold_answers,
            dataset_name=dataset_name,
            difficulty_score=difficulty_score,
            skills_enabled=skills_enabled,
            selected_pf_ids=selected_pf_ids or [],
            epoch=self.epoch,
        )
        self._trajectories.append(traj)
        return traj

    @staticmethod
    def record_step(
        trajectory: EpisodeTrajectory,
        step_index: int,
        proposed_action_type: str,
        proposed_action_arg: str,
        proposed_reasoning: str,
        final_action_type: str,
        final_action_arg: str,
        pf_records: List[Dict[str, Any]],
        context_injections: List[str],
        observation_summary: str = "",
        step_context: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        """Record a single step in the trajectory."""
        # Build PF activation records
        pf_activations = []
        for rec in pf_records:
            pf_activations.append(PFActivationRecord(
                pf_id=rec.get("skill_id", ""),
                activated=rec.get("activated", False),
                intervention_type=rec.get("intervention_type", "noop"),
                reason=rec.get("reason", ""),
                original_action=proposed_action_type if rec.get("activated") else None,
                original_arg=proposed_action_arg if rec.get("activated") else None,
                modified_action=rec.get("new_action_type") if rec.get("intervention_type") == "modify_action" else None,
                modified_arg=rec.get("new_action_arg") if rec.get("intervention_type") == "modify_action" else None,
                injected_text=rec.get("context_text", "")[:200] if rec.get("intervention_type") == "inject_context" else None,
            ))

        was_modified = (proposed_action_type != final_action_type or proposed_action_arg != final_action_arg)

        # Snapshot selected step_context fields
        snapshot = {}
        if step_context:
            for key in ("step_count", "has_read", "search_count", "read_count",
                        "empty_results", "contradictory_sources", "max_steps"):
                if key in step_context:
                    snapshot[key] = step_context[key]

        step_rec = StepRecord(
            step_index=step_index,
            proposed_action_type=proposed_action_type,
            proposed_action_arg=proposed_action_arg,
            proposed_reasoning=proposed_reasoning[:500],
            final_action_type=final_action_type,
            final_action_arg=final_action_arg,
            was_modified=was_modified,
            pf_activations=pf_activations,
            context_injections=context_injections,
            observation_summary=observation_summary[:300],
            step_context_snapshot=snapshot,
        )
        trajectory.steps.append(step_rec)
        return step_rec

    @staticmethod
    def finalize_episode(
        trajectory: EpisodeTrajectory,
        final_answer: str,
        exact_match: bool,
        f1_score: float = 0.0,
        ablation: str = "",
    ) -> None:
        """Finalize an episode trajectory with outcome."""
        trajectory.final_answer = final_answer
        trajectory.exact_match = exact_match
        trajectory.f1_score = f1_score
        trajectory.ablation = ablation
        trajectory.compute_stats()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = "trajectories.jsonl") -> Path:
        """Save all trajectories to JSONL."""
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for traj in self._trajectories:
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d trajectories to %s", len(self._trajectories), path)
        return path

    def load(self, filename: str = "trajectories.jsonl") -> List[EpisodeTrajectory]:
        """Load trajectories from JSONL."""
        path = self.output_dir / filename
        if not path.exists():
            return []
        trajectories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(EpisodeTrajectory.from_dict(json.loads(line)))
        self._trajectories = trajectories
        return trajectories

    @property
    def trajectories(self) -> List[EpisodeTrajectory]:
        return self._trajectories

    def get_failed_trajectories(self) -> List[EpisodeTrajectory]:
        """Return trajectories where the agent got the wrong answer."""
        return [t for t in self._trajectories if not t.exact_match]

    def get_rescued_steps(self) -> List[Dict[str, Any]]:
        """Return steps where PF modified the action and the episode succeeded."""
        rescued = []
        for traj in self._trajectories:
            if not traj.exact_match:
                continue
            for step in traj.steps:
                if step.was_modified:
                    rescued.append({
                        "sample_id": traj.sample_id,
                        "question": traj.question,
                        "step_index": step.step_index,
                        "original_action": step.proposed_action_type,
                        "original_arg": step.proposed_action_arg,
                        "modified_action": step.final_action_type,
                        "modified_arg": step.final_action_arg,
                        "reasoning": step.proposed_reasoning,
                    })
        return rescued

    def get_missed_interventions(self) -> List[Dict[str, Any]]:
        """Return failed episodes where no PF ever activated (potential missed interventions)."""
        missed = []
        for traj in self._trajectories:
            if traj.exact_match:
                continue
            if traj.total_pf_activations == 0 and traj.skills_enabled:
                missed.append({
                    "sample_id": traj.sample_id,
                    "question": traj.question,
                    "dataset_name": traj.dataset_name,
                    "steps": len(traj.steps),
                    "final_answer": traj.final_answer,
                    "gold_answers": traj.gold_answers,
                    "selected_pf_ids": traj.selected_pf_ids,
                })
        return missed
