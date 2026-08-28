"""
Pseudo-gradient computation — PF-mediated credit signals for Student and PF helper.

PFs are not gradient-updated themselves; they serve as observable interfaces that produce
fine-grained credit signals for model-centric updates.

Signal types:
  Student:
    - g_student_action: PF rescue direction (corrected - proposed)
    - g_student_risk: safer action preference under similar context
    - g_student_skillgen: quality of proposed skills

  PF helper:
    - g_teacher_select: skill selection effectiveness
    - g_teacher_judge: candidate skill review calibration

Signal taxonomy:
  All per-step scoring is delegated to `training.signals` (S1-S4) so that
  self-improving's reward formulation stays aligned with post-training.
  The composite `advantage` is a direct S4 aggregate; the full S1-S4
  breakdown is attached to every emitted record for downstream filtering.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from .configs import PseudoGradientConfig
from training.signals.trajectory import EpisodeTrajectory, StepRecord

# S1-S4 signal registry (shared with post-training)
from training.signals import SignalAggregator
from training.signals.aggregator import AggregatorConfig
from training.signals.registry import SignalRegistry
from training.signals import (  # noqa: F401 — registers signals
    s1_activation_timing, s2_intervention_stage, s3_content_quality, s4_benefit,
)

logger = logging.getLogger(__name__)


# Sub-signals whose sign is "bigger = better" (used for the risk proxy)
_S1_RISK_POS = "s1.fn"      # risky step, no PF fired — higher = should-have-caught
_S4_HARMFUL = "s4.side_effect"

# Default set of sub-signals attached to every per-step record (breakdown).
_DEFAULT_BREAKDOWN_SIGNALS = [
    "s1.tp", "s1.fp", "s1.fn", "s1.phase",
    "s2.pre_action", "s2.post_obs",
    "s3.syntactic", "s3.semantic",
    "s4.local", "s4.downstream", "s4.cost", "s4.side_effect",
]


@dataclass
class StepAdvantage:
    """Per-step advantage computed from PF mediation.

    ``local_improvement``/``downstream_success``/``extra_cost``/``harmful_side_effect``
    are direct mirrors of S4.* sub-signals (kept for backward compat with
    existing training_data_builder and gradient json consumers).
    ``signals`` carries the full S1-S4 breakdown.
    """
    sample_id: str
    step_index: int
    # Components (mirror of S4.*)
    local_improvement: float = 0.0     # = s4.local
    downstream_success: float = 0.0    # = s4.downstream
    extra_cost: float = 0.0            # = s4.cost
    harmful_side_effect: float = 0.0   # = s4.side_effect
    # Composite (S4-weighted)
    advantage: float = 0.0
    # Full S1..S4 breakdown (sub_id → value)
    signals: Dict[str, float] = field(default_factory=dict)
    # Context
    proposed_action: str = ""
    proposed_arg: str = ""
    corrected_action: str = ""
    corrected_arg: str = ""
    pf_ids: List[str] = field(default_factory=list)


@dataclass
class StudentGradient:
    """Pseudo-gradient signal for the student model."""
    # Action correction signals (from PF rescues)
    action_corrections: List[Dict[str, Any]] = field(default_factory=list)
    # Risk signals (from missed interventions)
    risk_signals: List[Dict[str, Any]] = field(default_factory=list)
    # Skill generation signals
    skillgen_signals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_corrections": self.action_corrections,
            "risk_signals": self.risk_signals,
            "skillgen_signals": self.skillgen_signals,
        }


@dataclass
class TeacherGradient:
    """Pseudo-gradient signal for the PF helper."""
    # Selection effectiveness signals
    selection_signals: List[Dict[str, Any]] = field(default_factory=list)
    # Judging calibration signals
    judging_signals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_signals": self.selection_signals,
            "judging_signals": self.judging_signals,
        }


class PseudoGradientComputer:
    """Computes PF-mediated pseudo-gradients for Student and PF helper.

    All per-step scoring is delegated to the shared S1-S4 signal registry
    (``training.signals``). The composite ``advantage`` is an S4-weighted
    aggregate mirroring PseudoGradientConfig (alpha/beta/gamma/delta),
    while ``signals`` carries the full S1-S4 breakdown for downstream
    filtering.
    """

    def __init__(self, config: PseudoGradientConfig, output_dir: Optional[str] = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        # S4-only aggregator with the pseudo_gradient.yaml weights — this
        # preserves the composite `advantage` semantics while reusing
        # the shared signal implementations.
        self._s4_agg = SignalAggregator(AggregatorConfig(
            enabled=["s4.local", "s4.downstream", "s4.cost", "s4.side_effect"],
            weights={
                "s4.local": float(getattr(config, "alpha_local", 0.5)),
                "s4.downstream": float(getattr(config, "beta_downstream", 0.3)),
                "s4.cost": -float(getattr(config, "gamma_cost", 0.1)),
                "s4.side_effect": -float(getattr(config, "delta_side_effect", 0.1)),
            },
            normalize=False,
        ))

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def compute_student_gradient(
        self,
        trajectories: List[EpisodeTrajectory],
        skill_quality_scores: Optional[Dict[str, float]] = None,
    ) -> StudentGradient:
        """Compute pseudo-gradient for the student model.

        Args:
            trajectories: PF-aware episode trajectories
            skill_quality_scores: Optional mapping of proposed skill_id → Q_skill
        """
        grad = StudentGradient()

        for traj in trajectories:
            # A. Action correction signals (from PF rescues)
            corrections = self._compute_action_corrections(traj)
            grad.action_corrections.extend(corrections)

            # B. Risk signals (from failures where PFs didn't help enough)
            risks = self._compute_risk_signals(traj)
            grad.risk_signals.extend(risks)

        # C. Skill generation signals
        if skill_quality_scores:
            for skill_id, q_score in skill_quality_scores.items():
                grad.skillgen_signals.append({
                    "skill_id": skill_id,
                    "quality_score": q_score,
                    "signal": q_score - 0.5,  # Center around 0.5 baseline
                })

        if self.output_dir:
            self._save_gradient("student_gradient.json", grad.to_dict())

        return grad

    def compute_teacher_gradient(
        self,
        trajectories: List[EpisodeTrajectory],
        judging_outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> TeacherGradient:
        """Compute pseudo-gradient for the PF helper.

        Args:
            trajectories: PF-aware episode trajectories
            judging_outcomes: Optional list of {skill_id, teacher_predicted_quality, actual_quality}
        """
        grad = TeacherGradient()

        for traj in trajectories:
            signals = self._compute_selection_signals(traj)
            grad.selection_signals.extend(signals)

        # Judging calibration
        if judging_outcomes:
            for outcome in judging_outcomes:
                predicted = outcome.get("teacher_predicted_quality", 0.5)
                actual = outcome.get("actual_quality", 0.5)
                grad.judging_signals.append({
                    "skill_id": outcome.get("skill_id", ""),
                    "predicted_quality": predicted,
                    "actual_quality": actual,
                    "calibration_error": abs(predicted - actual),
                    "signal": -(abs(predicted - actual)),
                })

        if self.output_dir:
            self._save_gradient("teacher_gradient.json", grad.to_dict())

        return grad

    # ------------------------------------------------------------------
    # Student: action correction (Situation A — PF rescue)
    # ------------------------------------------------------------------

    def _compute_action_corrections(self, traj: EpisodeTrajectory) -> List[Dict[str, Any]]:
        """Extract action correction signals from PF-modified steps."""
        corrections = []

        for step in traj.steps:
            if not step.was_modified:
                continue

            # Compute per-step advantage (S4 composite) + full breakdown
            adv = self._compute_step_advantage(traj, step)

            corrections.append({
                "sample_id": traj.sample_id,
                "question": traj.question[:200],
                "step_index": step.step_index,
                # Student proposed
                "proposed_action": step.proposed_action_type,
                "proposed_arg": step.proposed_action_arg[:200],
                "proposed_reasoning": step.proposed_reasoning[:300],
                # PF corrected
                "corrected_action": step.final_action_type,
                "corrected_arg": step.final_action_arg[:200],
                # PF info
                "pf_ids": [a.pf_id for a in step.pf_activations if a.activated],
                # Advantage (S4 composite) + back-compat fields
                "advantage": adv.advantage,
                "local_improvement": adv.local_improvement,
                "downstream_success": adv.downstream_success,
                # Full S1..S4 breakdown (aligned with post-training signals)
                "signals": adv.signals,
                # Context for training
                "step_context": step.step_context_snapshot,
                "episode_success": traj.exact_match,
            })

        return corrections

    # ------------------------------------------------------------------
    # Student: risk signals (Situation B — PF miss)
    # ------------------------------------------------------------------

    def _compute_risk_signals(self, traj: EpisodeTrajectory) -> List[Dict[str, Any]]:
        """Identify risky steps in failed episodes that should have been caught.

        Risk is derived from S1 (activation timing): a step is risky when
        the S1 oracle flags it, and especially when S1.fn fires (risky +
        no PF activated). S4.side_effect adds additional risk weight when
        a PF fired but the episode still failed.
        """
        if traj.exact_match:
            return []

        signals = []
        for step in traj.steps:
            bd = self._compute_breakdown(traj, step)
            risk = self._risk_from_breakdown(bd, step)
            if risk > 0.3:
                signals.append({
                    "sample_id": traj.sample_id,
                    "question": traj.question[:200],
                    "step_index": step.step_index,
                    "action": step.final_action_type,
                    "arg": step.final_action_arg[:200],
                    "reasoning": step.proposed_reasoning[:300],
                    "risk_score": risk,
                    "was_modified": step.was_modified,
                    "any_pf_activated": any(a.activated for a in step.pf_activations),
                    "step_context": step.step_context_snapshot,
                    # Full S1..S4 breakdown (aligned with post-training signals)
                    "signals": bd,
                    # Signal: student should prefer safer action
                    "safer_action_hint": self._suggest_safer_action(step),
                })

        return signals

    # ------------------------------------------------------------------
    # PF helper: selection effectiveness
    # ------------------------------------------------------------------

    def _compute_selection_signals(self, traj: EpisodeTrajectory) -> List[Dict[str, Any]]:
        """Compute credit signals for PF helper's skill selection.

        Usefulness = PF actually fired on at least one step (S2 > 0).
        We do NOT require the PF to have been in the pre-selected top-k
        set — that would just reinforce the selector's current choices
        and zero-out training data whenever selector/runtime disagree.
        Wasted = selected-but-never-fired (still a selector signal).
        """
        signals = []
        selected = set(traj.selected_pf_ids)

        useful_pfs = set()
        activated_pfs = set()

        for step in traj.steps:
            # S2: did ANY intervention fire on this step?
            bd = self._compute_breakdown(traj, step)
            any_intervention = (bd.get("s2.pre_action", 0.0) > 0
                                or bd.get("s2.post_obs", 0.0) > 0)
            if not any_intervention:
                continue
            for act in step.pf_activations:
                if act.activated:
                    activated_pfs.add(act.pf_id)
                    useful_pfs.add(act.pf_id)
        wasted_pfs = selected - activated_pfs

        # Credit scaled by S4.downstream (episode success = 1, failure = 0.3)
        downstream = 1.0 if traj.exact_match else 0.3
        rescue_credit = len(useful_pfs) * downstream
        waste_penalty = len(wasted_pfs) * 0.1

        signals.append({
            "sample_id": traj.sample_id,
            "question": traj.question[:200],
            "selected_pf_ids": sorted(selected),
            "useful_pf_ids": sorted(useful_pfs),
            "wasted_pf_ids": sorted(wasted_pfs),
            "episode_success": traj.exact_match,
            "rescue_credit": rescue_credit,
            "waste_penalty": waste_penalty,
            "net_signal": rescue_credit - waste_penalty,
        })

        return signals

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def _compute_breakdown(
        self, traj: EpisodeTrajectory, step: StepRecord
    ) -> Dict[str, float]:
        """Compute the full S1..S4 sub-signal breakdown for a step.

        Returns a flat {sub_id: raw_value} dict (weights NOT applied — keeps
        downstream filtering free to choose its own aggregation).
        """
        out = {}
        for sub_id in _DEFAULT_BREAKDOWN_SIGNALS:
            try:
                fn = SignalRegistry.get(sub_id)
            except KeyError:
                continue
            so = fn(traj, step, {})
            if so is not None:
                out[sub_id] = float(so.value)
        return out

    def _compute_step_advantage(self, traj: EpisodeTrajectory, step: StepRecord) -> StepAdvantage:
        """Compute advantage for a PF-modified step.

        Delegates per-signal scoring to the S1-S4 registry; the composite
        ``advantage`` is the S4 weighted aggregate using alpha/beta/gamma/delta
        from PseudoGradientConfig (identical algebra to the previous
        hand-coded formula, now sourced from the shared signal functions).
        """
        adv = StepAdvantage(
            sample_id=traj.sample_id,
            step_index=step.step_index,
            proposed_action=step.proposed_action_type,
            proposed_arg=step.proposed_action_arg,
            corrected_action=step.final_action_type,
            corrected_arg=step.final_action_arg,
            pf_ids=[a.pf_id for a in step.pf_activations if a.activated],
        )

        bd = self._compute_breakdown(traj, step)
        adv.signals = bd

        # Back-compat mirrored fields (S4.*)
        adv.local_improvement = bd.get("s4.local", 0.0)
        adv.downstream_success = bd.get("s4.downstream", 0.0)
        adv.extra_cost = bd.get("s4.cost", 0.0)
        adv.harmful_side_effect = bd.get("s4.side_effect", 0.0)

        # Composite advantage = alpha·S4.local + beta·S4.downstream
        #                       - gamma·S4.cost - delta·S4.side_effect
        adv.advantage = self._s4_agg.score_step(traj, step)

        return adv

    def _risk_from_breakdown(self, bd: Dict[str, float], step: StepRecord) -> float:
        """S1/S4-derived risk proxy for _compute_risk_signals.

        Combines S1 indicators (the oracle fires on final-without-read,
        premature FINAL, empty-result searches) and S4.side_effect
        (PF fired but episode failed).
        """
        risky = bd.get("s1.tp", 0.0) + bd.get("s1.fn", 0.0) > 0
        if not risky:
            return 0.0
        # Base risk: 0.8 when the oracle flags it
        risk = 0.8
        # +0.2 if PF missed it entirely (S1.fn)
        if bd.get(_S1_RISK_POS, 0.0) > 0:
            risk = min(risk + 0.2, 1.0)
        # +0.2 if PF fired but still harmful (S4.side_effect)
        if bd.get(_S4_HARMFUL, 0.0) > 0:
            risk = min(risk + 0.2, 1.0)
        return risk

    def _suggest_safer_action(self, step: StepRecord) -> str:
        """Suggest what action would have been safer."""
        if step.final_action_type == "FINAL":
            ctx = step.step_context_snapshot
            if not ctx.get("has_read", False):
                return "READ (should read a document before answering)"
            if ctx.get("step_count", 0) < 3:
                return "SEARCH (should gather more evidence)"
        if step.final_action_type == "SEARCH" and step.step_context_snapshot.get("empty_results"):
            return "SEARCH with simplified query"
        return ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_gradient(self, filename: str, data: Dict[str, Any]) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
