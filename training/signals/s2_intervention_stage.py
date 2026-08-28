"""S2 — Intervention stage signal.

Tracks where in the ReAct cycle the PF intervenes:
  - pre_action   (MODIFY_ACTION before the action runs)
  - post_obs     (INJECT_CONTEXT after observation)
  - pre_reasoning (reserved future)
  - post_action   (reserved future)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .registry import SignalRegistry, SignalSpec, SignalOutput


def _s2_pre_action(traj, step, context) -> Optional[SignalOutput]:
    v = 1.0 if any(
        a.activated and a.intervention_type == "modify_action"
        for a in step.pf_activations
    ) else 0.0
    return SignalOutput("s2.pre_action", v, traj.sample_id, step.step_index)


def _s2_post_obs(traj, step, context) -> Optional[SignalOutput]:
    v = 1.0 if any(
        a.activated and a.intervention_type == "inject_context"
        for a in step.pf_activations
    ) else 0.0
    return SignalOutput("s2.post_obs", v, traj.sample_id, step.step_index)


def _s2_pre_reasoning(traj, step, context) -> Optional[SignalOutput]:
    # Reserved — returns 0 until a new intervention type is added
    return SignalOutput("s2.pre_reasoning", 0.0, traj.sample_id, step.step_index)


def _s2_post_action(traj, step, context) -> Optional[SignalOutput]:
    return SignalOutput("s2.post_action", 0.0, traj.sample_id, step.step_index)


def compute_s2(traj, step, context: Optional[Dict[str, Any]] = None) -> Dict[str, SignalOutput]:
    return {
        "s2.pre_action": _s2_pre_action(traj, step, context),
        "s2.post_obs": _s2_post_obs(traj, step, context),
        "s2.pre_reasoning": _s2_pre_reasoning(traj, step, context),
        "s2.post_action": _s2_post_action(traj, step, context),
    }


SignalRegistry.register("s2.pre_action", _s2_pre_action, SignalSpec("s2.pre_action", "S2", "Pre-action MODIFY_ACTION", 0.35))
SignalRegistry.register("s2.post_obs", _s2_post_obs, SignalSpec("s2.post_obs", "S2", "Post-observation INJECT_CONTEXT", 0.35))
SignalRegistry.register("s2.pre_reasoning", _s2_pre_reasoning, SignalSpec("s2.pre_reasoning", "S2", "(reserved) pre-reasoning hint", 0.15))
SignalRegistry.register("s2.post_action", _s2_post_action, SignalSpec("s2.post_action", "S2", "(reserved) post-action reflection", 0.15))
