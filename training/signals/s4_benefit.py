"""S4 — Intervention benefit (StepAdvantage components)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .registry import SignalRegistry, SignalSpec, SignalOutput


def _s4_local(traj, step, context) -> Optional[SignalOutput]:
    v = 0.0
    if step.was_modified:
        if step.proposed_action_type == "FINAL" and step.final_action_type == "READ":
            v = 0.8
        elif step.proposed_action_type == "FINAL" and step.final_action_type == "SEARCH":
            v = 0.7
        elif step.proposed_action_type == "SEARCH" and step.final_action_type == "SEARCH":
            v = 0.5
        else:
            v = 0.3
    return SignalOutput("s4.local", v, traj.sample_id, step.step_index)


def _s4_downstream(traj, step, context) -> Optional[SignalOutput]:
    v = 1.0 if traj.exact_match else 0.0
    return SignalOutput("s4.downstream", v, traj.sample_id, step.step_index)


def _s4_cost(traj, step, context) -> Optional[SignalOutput]:
    total = len(traj.steps)
    v = min(max((total - 15) / 10.0, 0.0), 1.0) if total > 15 else 0.0
    return SignalOutput("s4.cost", v, traj.sample_id, step.step_index)


def _s4_side_effect(traj, step, context) -> Optional[SignalOutput]:
    v = 1.0 if (step.was_modified and not traj.exact_match) else 0.0
    return SignalOutput("s4.side_effect", v, traj.sample_id, step.step_index)


def compute_s4(traj, step, context: Optional[Dict[str, Any]] = None) -> Dict[str, SignalOutput]:
    return {
        "s4.local": _s4_local(traj, step, context),
        "s4.downstream": _s4_downstream(traj, step, context),
        "s4.cost": _s4_cost(traj, step, context),
        "s4.side_effect": _s4_side_effect(traj, step, context),
    }


SignalRegistry.register("s4.local", _s4_local, SignalSpec("s4.local", "S4", "Local step improvement", 0.40))
SignalRegistry.register("s4.downstream", _s4_downstream, SignalSpec("s4.downstream", "S4", "Episode success", 0.40))
SignalRegistry.register("s4.cost", _s4_cost, SignalSpec("s4.cost", "S4", "Extra-step cost (subtracted)", -0.10))
SignalRegistry.register("s4.side_effect", _s4_side_effect, SignalSpec("s4.side_effect", "S4", "Harmful modification (subtracted)", -0.10))
