"""The trajectory schema the credit signals read.

A signal is a function of one step inside one rollout, and what it needs to see
is not the text but the *decision*: what the agent proposed, what it ended up
doing, and which skills fired in between. The gap between proposed and final is
what separates a rescue from a false positive — without it, a skill that
successfully rewrote an action looks exactly like one that fired for nothing.

These three dataclasses are that record. They live here, next to the signals
that consume them, because both `training/signals/` and `evolving/review.py`
need them and neither should have to import the other's package to get a type.

`self_improving/trajectory_logger.py` re-exports them and adds the logger that
writes and reloads whole runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PFActivationRecord:
    """Record of a single PF check on one step."""
    pf_id: str
    activated: bool
    intervention_type: str = "noop"  # noop / modify_action / inject_context
    reason: str = ""
    # Before/after if modify_action
    original_action: Optional[str] = None
    original_arg: Optional[str] = None
    modified_action: Optional[str] = None
    modified_arg: Optional[str] = None
    # Injected context if inject_context
    injected_text: Optional[str] = None


@dataclass
class StepRecord:
    """Full record of a single agent step."""
    step_index: int
    # Student's proposed action (before PF)
    proposed_action_type: str = ""
    proposed_action_arg: str = ""
    proposed_reasoning: str = ""
    # After PF intervention
    final_action_type: str = ""
    final_action_arg: str = ""
    # Was the action modified by a PF?
    was_modified: bool = False
    # PF activation records
    pf_activations: List[PFActivationRecord] = field(default_factory=list)
    # Context injected by PFs
    context_injections: List[str] = field(default_factory=list)
    # Observation summary (truncated)
    observation_summary: str = ""
    # Step context snapshot (selected fields)
    step_context_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    """Full PF-aware trajectory for one episode."""
    # Identity
    sample_id: str = ""
    question: str = ""
    dataset_name: str = ""
    gold_answers: List[str] = field(default_factory=list)

    # PF helper decisions
    difficulty_score: int = 0
    skills_enabled: bool = True
    selected_pf_ids: List[str] = field(default_factory=list)

    # Step-by-step records
    steps: List[StepRecord] = field(default_factory=list)

    # Final outcome
    final_answer: str = ""
    exact_match: bool = False
    f1_score: float = 0.0

    # Aggregated PF stats
    total_pf_activations: int = 0
    total_action_modifications: int = 0
    total_context_injections: int = 0

    # Epoch metadata
    epoch: int = 0
    ablation: str = ""

    def compute_stats(self) -> None:
        """Compute aggregated stats from step records."""
        self.total_pf_activations = sum(
            sum(1 for a in s.pf_activations if a.activated)
            for s in self.steps
        )
        self.total_action_modifications = sum(1 for s in self.steps if s.was_modified)
        self.total_context_injections = sum(len(s.context_injections) for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeTrajectory":
        steps = [
            StepRecord(
                **{k: v for k, v in s.items() if k != "pf_activations"},
                pf_activations=[PFActivationRecord(**a) for a in s.get("pf_activations", [])],
            )
            for s in d.pop("steps", [])
        ]
        return cls(**d, steps=steps)
