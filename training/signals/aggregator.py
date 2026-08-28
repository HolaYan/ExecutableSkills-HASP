"""Signal aggregator — combines per-signal outputs into scalar rewards.

Supports:
  * Single-signal ablation (enabled=[s4.downstream]) — S1..S4 signal ablation
  * Full-weighted mix (enabled=[s1.tp, s1.phase, s2.pre_action, ...])
  * Skill-level aggregation (Q_skill + ΔEM) for Objective B
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .registry import SignalRegistry, SignalOutput
from . import s1_activation_timing, s2_intervention_stage, s3_content_quality, s4_benefit  # noqa: F401 — register signals


# S-family parent → coarse 4-category name.
# timing  ← S1 (when to fire)
# modality← S2 (pre_action vs post_obs)
# correctness ← S3 (well-formed output + semantic quality)
# outcome ← S4 (actually helps / doesn't hurt)
FAMILY_MAP: Dict[str, str] = {
    "S1": "timing",
    "S2": "modality",
    "S3": "correctness",
    "S4": "outcome",
}

# Default weights for combining the 4 family scalars into one scalar.
# Outcome carries the most weight (end-to-end EM); correctness second;
# timing/modality smaller since they're easier to game.
DEFAULT_FAMILY_WEIGHTS: Dict[str, float] = {
    "correctness": 0.25,
    "timing": 0.15,
    "modality": 0.10,
    "outcome": 0.50,
}


@dataclass
class AggregatorConfig:
    enabled: List[str]              # e.g. ["s1.tp", "s4.downstream"]
    weights: Optional[Dict[str, float]] = None  # sub_id → weight; falls back to default_weight
    normalize: bool = True           # divide by sum of |weights| of enabled signals
    mode: str = "fine"               # "fine" (15-dim) | "coarse" (4-scalar)
    # Family weights used only when mode == "coarse".
    family_weights: Optional[Dict[str, float]] = None

    def weight_for(self, sub_id: str) -> float:
        if self.weights and sub_id in self.weights:
            return self.weights[sub_id]
        return SignalRegistry.get_spec(sub_id).default_weight

    def family_weight_for(self, family: str) -> float:
        if self.family_weights and family in self.family_weights:
            return self.family_weights[family]
        return DEFAULT_FAMILY_WEIGHTS.get(family, 0.0)


class SignalAggregator:
    """Aggregates a configured subset of signals into (step|episode|skill) rewards."""

    def __init__(self, config: AggregatorConfig):
        self.config = config
        # Validate that all enabled signals are registered
        for sub in config.enabled:
            SignalRegistry.get(sub)  # raises if missing

    # ------------------------------------------------------------------
    # Step-level
    # ------------------------------------------------------------------

    def score_step(self, traj, step, context: Optional[Dict[str, Any]] = None) -> float:
        if self.config.mode == "coarse":
            coarse = self.breakdown_coarse(traj, step, context)
            return sum(
                self.config.family_weight_for(fam) * val
                for fam, val in coarse.items()
            )
        # fine mode (legacy 15-dim weighted sum)
        context = context or {}
        total = 0.0
        wsum = 0.0
        for sub in self.config.enabled:
            fn = SignalRegistry.get(sub)
            out = fn(traj, step, context)
            if out is None:
                continue
            w = self.config.weight_for(sub)
            total += w * out.value
            wsum += abs(w)
        if self.config.normalize and wsum > 0:
            total /= wsum
        return total

    def score_episode(self, traj, context: Optional[Dict[str, Any]] = None) -> float:
        """Episode-level reward = mean over steps + outcome bonus."""
        step_scores = [self.score_step(traj, s, context) for s in traj.steps]
        step_mean = sum(step_scores) / len(step_scores) if step_scores else 0.0
        outcome = 1.0 if traj.exact_match else 0.0
        return 0.5 * step_mean + 0.5 * outcome

    # ------------------------------------------------------------------
    # Skill-level (Objective B)
    # ------------------------------------------------------------------

    def score_skill(
        self,
        q_skill: float,
        validation_delta_em: float = 0.0,
        lam: float = 0.5,
    ) -> float:
        """r_skill = Q_skill + λ · ΔEM(validation)."""
        return float(q_skill) + lam * float(validation_delta_em)

    # ------------------------------------------------------------------
    # Debug / breakdown
    # ------------------------------------------------------------------

    def breakdown(self, traj, step, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Raw 15-dim per-sub-signal weighted contributions (for audit / re-score)."""
        context = context or {}
        out = {}
        for sub in self.config.enabled:
            fn = SignalRegistry.get(sub)
            so = fn(traj, step, context)
            if so is not None:
                out[sub] = self.config.weight_for(sub) * so.value
        return out

    def breakdown_coarse(
        self, traj, step, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """4-scalar view {correctness, timing, modality, outcome} — family-wise
        normalized weighted sum. Keeps sign: outcome/timing can be negative
        when penalty sub-signals (s1.fp/fn, s4.cost/side_effect) dominate."""
        context = context or {}
        by_parent: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0.0, "wsum": 0.0})
        for sub in self.config.enabled:
            fn = SignalRegistry.get(sub)
            so = fn(traj, step, context)
            if so is None:
                continue
            parent = SignalRegistry.get_spec(sub).parent
            w = self.config.weight_for(sub)
            by_parent[parent]["total"] += w * so.value
            by_parent[parent]["wsum"] += abs(w)
        result: Dict[str, float] = {}
        for parent, family in FAMILY_MAP.items():
            d = by_parent.get(parent, {"total": 0.0, "wsum": 0.0})
            result[family] = (d["total"] / d["wsum"]) if d["wsum"] > 0 else 0.0
        return result

    def coarse_from_breakdown(
        self, breakdown_raw: Dict[str, float],
    ) -> Dict[str, float]:
        """Offline re-score path: reconstruct 4-scalar view from a pre-stored
        `signal_breakdown` dict (already weighted per-sub-signal). Used by
        `training/signals/rescore.py` — no trajectory needed."""
        by_parent: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0.0, "wsum": 0.0})
        for sub, weighted_val in breakdown_raw.items():
            try:
                parent = SignalRegistry.get_spec(sub).parent
                w = SignalRegistry.get_spec(sub).default_weight
            except Exception:
                continue
            by_parent[parent]["total"] += float(weighted_val)
            by_parent[parent]["wsum"] += abs(w)
        result: Dict[str, float] = {}
        for parent, family in FAMILY_MAP.items():
            d = by_parent.get(parent, {"total": 0.0, "wsum": 0.0})
            result[family] = (d["total"] / d["wsum"]) if d["wsum"] > 0 else 0.0
        return result

    def scalar_from_coarse(self, coarse: Dict[str, float]) -> float:
        """Collapse 4 family scalars into a single sample_weight."""
        return sum(self.config.family_weight_for(f) * v for f, v in coarse.items())
