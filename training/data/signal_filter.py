"""Filter step-level samples by a (subset of) signal(s).

Used by Strategy A — Signal Ablation — to create per-signal training
datasets for off-policy SFT / RS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..signals import SignalAggregator
from ..signals.registry import SignalRegistry


@dataclass
class FilterConfig:
    enabled_signals: List[str]     # e.g. ["s4.downstream"] for single-signal ablation
    threshold: float = 0.5         # keep samples with aggregate reward ≥ threshold
    top_k_per_episode: Optional[int] = None  # keep only top-k steps per episode (None = all)
    include_weights: bool = True   # attach sample weight = aggregate reward


class SignalFilter:
    def __init__(self, aggregator: SignalAggregator, config: FilterConfig):
        self.aggregator = aggregator
        self.config = config

    def filter_steps(
        self,
        trajectories: Iterable,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return step-level samples that pass the filter.

        Each sample includes: sample_id, step_index, step (StepRecord),
        trajectory (for episode context), aggregate_reward, breakdown.
        """
        kept: List[Dict[str, Any]] = []
        for traj in trajectories:
            step_scores = []
            for step in traj.steps:
                r = self.aggregator.score_step(traj, step, context)
                step_scores.append((step, r))

            # Optionally keep only top-k steps per episode
            if self.config.top_k_per_episode is not None:
                step_scores.sort(key=lambda x: x[1], reverse=True)
                step_scores = step_scores[: self.config.top_k_per_episode]

            for step, r in step_scores:
                if r < self.config.threshold:
                    continue
                kept.append({
                    "sample_id": traj.sample_id,
                    "step_index": step.step_index,
                    "step": step,
                    "trajectory": traj,
                    "aggregate_reward": r,
                    "breakdown": self.aggregator.breakdown(traj, step, context),
                    # 4-scalar coarse view — persisted alongside 15-dim raw.
                    "breakdown_coarse": self.aggregator.breakdown_coarse(traj, step, context),
                })
        return kept

    def filter_skill_candidates(
        self,
        candidates: List,
        reviews_by_id: Dict[str, Any],
        val_delta_by_id: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """For Objective B: keep candidates whose r_skill ≥ threshold."""
        val_delta_by_id = val_delta_by_id or {}
        kept = []
        for cand in candidates:
            review = reviews_by_id.get(cand.skill_id)
            if review is None:
                continue
            r = self.aggregator.score_skill(
                q_skill=getattr(review, "q_skill", 0.0),
                validation_delta_em=val_delta_by_id.get(cand.skill_id, 0.0),
            )
            if r < threshold:
                continue
            kept.append({
                "skill_id": cand.skill_id,
                "candidate": cand,
                "review": review,
                "reward": r,
            })
        return kept


_PARENTS = {"S1", "S2", "S3", "S4"}


def resolve_enabled_signals(spec: str) -> List[str]:
    """Expand short names like 'S1', 'S4', 'all', or 'all,-S1' to explicit sub_ids.

    Accepted forms:
      * "all"              → every registered sub_id
      * "S<i>"             → all sub_ids whose parent == S<i>
      * "S1,S2"            → comma list of parents (each expanded)
      * "all,-S1"          → all minus S1's sub_ids (signal-ablation form)
      * "s1.tp,s4.downstream" → explicit sub_ids
      * mixes are allowed: "S2,S3,S4" or "all,-S2,-S3"
    """
    if spec == "all":
        return SignalRegistry.list_signals()
    if spec in _PARENTS:
        return SignalRegistry.list_signals(parent=spec)
    if "," not in spec:
        return [spec]

    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    include: List[str] = []
    exclude: set = set()
    started_with_all = False
    for tok in tokens:
        if tok == "all":
            include.extend(SignalRegistry.list_signals())
            started_with_all = True
        elif tok.startswith("-"):
            ex = tok[1:]
            if ex in _PARENTS:
                exclude.update(SignalRegistry.list_signals(parent=ex))
            else:
                exclude.add(ex)
        elif tok in _PARENTS:
            include.extend(SignalRegistry.list_signals(parent=tok))
        else:
            include.append(tok)

    # If only excludes given (e.g. "-S1"), start from all.
    if not include and exclude:
        include = SignalRegistry.list_signals()
        started_with_all = True

    seen: set = set()
    out: List[str] = []
    for s in include:
        if s in exclude or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
