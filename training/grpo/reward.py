"""PF-based reward function for GRPO.

TRL `GRPOTrainer` invokes reward funcs as:

    reward_fn(completions, **kwargs) -> list[float]

where every *dataset column* is passed as a parallel list in ``kwargs``
(i.e. ``kwargs["sample_id"]`` is a list aligned with ``completions``).
We therefore expect the prompts jsonl to carry all fields the signal
aggregator needs (``sample_id``, ``step_index``, ``step_context``,
``question``).

Two scoring modes:

  * ``"action"`` — generation is a ReAct action line; we build a stub
    single-step trajectory around the generated action and let the
    ``SignalAggregator`` score it. **Note (V1 limitation):** because we
    do not actually execute the action, ``s4.downstream`` is left at 0
    — only S1/S2/S3 + ``s4.local`` / ``s4.cost`` / ``s4.side_effect``
    carry meaningful gradient. Full-episode GRPO is deferred to V2.

  * ``"skill"`` — generation is a SKILL.md + PF code; we rely on a
    `TeacherVerifier` (GPT-4o) as the sole reward signal.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..signals import SignalAggregator
from ..signals.aggregator import AggregatorConfig


logger = logging.getLogger(__name__)


_ACTION_PAT = re.compile(r"Action:\s*(SEARCH|READ|FINAL)\s*\((.*?)\)\s*$", re.IGNORECASE | re.DOTALL)


def _coerce(value):
    """Turn TRL's per-row value into a plain Python dict/scalar."""
    if isinstance(value, dict):
        return value
    return value


def _stub_trajectory(meta: Dict[str, Any], generation: str):
    """One-step stub trajectory for single-step action reward."""
    from training.signals.trajectory import (
        EpisodeTrajectory, StepRecord, PFActivationRecord,
    )

    m = _ACTION_PAT.search(generation or "")
    if m:
        final_type = m.group(1).upper()
        final_arg = m.group(2).strip()
    else:
        final_type, final_arg = "INVALID", ""

    step_context = _coerce(meta.get("step_context", {}) or {})
    step = StepRecord(
        step_index=int(meta.get("step_index", 0) or 0),
        proposed_action_type=final_type,
        proposed_action_arg=final_arg,
        proposed_reasoning="",
        final_action_type=final_type,
        final_action_arg=final_arg,
        was_modified=False,
        pf_activations=[PFActivationRecord(pf_id="generated", activated=False)],
        step_context_snapshot=step_context,
    )
    traj = EpisodeTrajectory(
        sample_id=str(meta.get("sample_id", "") or ""),
        question=str(meta.get("question", "") or ""),
        exact_match=False,
        steps=[step],
    )
    return traj, step


def _completion_text(comp) -> str:
    """Accept either str or chat-format list[dict] completions from TRL."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[-1], dict):
        return comp[-1].get("content", "")
    return str(comp)


def build_reward_fn(
    enabled_signals: List[str],
    weights: Optional[Dict[str, float]] = None,
    mode: str = "action",
    teacher_verifier: Optional[Callable] = None,
) -> Callable:
    """Factory: returns a reward fn with TRL-compatible signature.

    `reward_fn(completions, **kwargs) -> list[float]`

    ``kwargs`` must contain at least:
      - sample_id          (list[str])
      - step_index         (list[int])
      - step_context       (list[dict])
      - question           (list[str])
    when `mode == "action"`.

    For `mode == "skill"` the per-row ``target_failure_pattern`` is used.
    """
    agg = SignalAggregator(AggregatorConfig(enabled=enabled_signals, weights=weights or {}))

    def reward_fn(completions, **kwargs):
        n = len(completions)
        # Resolve per-column parallel lists (TRL provides them as lists)
        def _col(name, default=None):
            v = kwargs.get(name, None)
            if v is None:
                return [default] * n
            # TRL may pass a single value if the col is missing; normalise
            if isinstance(v, list):
                return v
            return [v] * n

        if mode == "action":
            sample_ids = _col("sample_id", "")
            step_indices = _col("step_index", 0)
            step_contexts = _col("step_context", {})
            questions = _col("question", "")

            rewards = []
            for i, comp in enumerate(completions):
                gen = _completion_text(comp)
                meta = {
                    "sample_id": sample_ids[i],
                    "step_index": step_indices[i],
                    "step_context": step_contexts[i] or {},
                    "question": questions[i] or "",
                }
                traj, step = _stub_trajectory(meta, gen)
                try:
                    r = agg.score_step(traj, step, context={})
                except Exception as e:
                    logger.warning("reward error (action): %s", e)
                    r = 0.0
                rewards.append(float(r))
            return rewards

        elif mode == "skill":
            if teacher_verifier is None:
                return [0.0] * n
            patterns = _col("target_failure_pattern", "")
            rewards = []
            for i, comp in enumerate(completions):
                gen = _completion_text(comp)
                row = {"target_failure_pattern": patterns[i] or ""}
                try:
                    vr = teacher_verifier.score(row, gen, prompt_index=i, group_index=0)
                    rewards.append(float(vr.score))
                except Exception as e:
                    logger.warning("reward error (skill): %s", e)
                    rewards.append(0.0)
            return rewards

        raise ValueError(f"Unknown reward mode: {mode}")

    reward_fn.__name__ = f"pf_signal_reward_{mode}"
    return reward_fn
