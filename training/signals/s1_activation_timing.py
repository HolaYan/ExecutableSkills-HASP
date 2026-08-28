"""S1 — Activation timing signal.

Measures whether PFs fire at the right steps (TP / FP / FN) and
how early in the episode they activate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .registry import SignalRegistry, SignalSpec, SignalOutput


def _oracle_risky(step, traj) -> bool:
    """Heuristic oracle: was this step actually risky?

    Evaluates the *proposed* action (what the agent would have done
    without PF help), not the final action. Otherwise a successful
    MODIFY_ACTION rescue (e.g. FINAL→READ) hides the risk signal and
    every rescue counts as a false-positive.

    A step is risky if:
      - proposed FINAL without has_read
      - proposed FINAL at step_count < 3
      - proposed SEARCH after empty_results
    """
    ctx = step.step_context_snapshot or {}
    proposed = getattr(step, "proposed_action_type", None) or step.final_action_type
    if proposed == "FINAL" and not ctx.get("has_read", False):
        return True
    if proposed == "FINAL" and ctx.get("step_count", 99) < 3:
        return True
    if proposed == "SEARCH" and ctx.get("empty_results", False):
        return True
    return False


def _s1_tp(traj, step, context) -> Optional[SignalOutput]:
    risky = _oracle_risky(step, traj)
    any_activated = any(a.activated for a in step.pf_activations)
    value = 1.0 if (risky and any_activated) else 0.0
    return SignalOutput(
        sub_id="s1.tp", value=value,
        sample_id=traj.sample_id, step_index=step.step_index,
        extra={"risky": risky, "activated": any_activated},
    )


def _s1_fp(traj, step, context) -> Optional[SignalOutput]:
    risky = _oracle_risky(step, traj)
    any_activated = any(a.activated for a in step.pf_activations)
    value = 1.0 if (not risky and any_activated) else 0.0
    return SignalOutput(
        sub_id="s1.fp", value=value,
        sample_id=traj.sample_id, step_index=step.step_index,
    )


def _s1_fn(traj, step, context) -> Optional[SignalOutput]:
    risky = _oracle_risky(step, traj)
    any_activated = any(a.activated for a in step.pf_activations)
    value = 1.0 if (risky and not any_activated) else 0.0
    return SignalOutput(
        sub_id="s1.fn", value=value,
        sample_id=traj.sample_id, step_index=step.step_index,
    )


def _s1_phase(traj, step, context) -> Optional[SignalOutput]:
    total = max(len(traj.steps), 1)
    value = step.step_index / total
    return SignalOutput(
        sub_id="s1.phase", value=value,
        sample_id=traj.sample_id, step_index=step.step_index,
    )


def compute_s1(traj, step, context: Optional[Dict[str, Any]] = None) -> Dict[str, SignalOutput]:
    """Compute all S1 sub-signals for a single step."""
    out = {}
    for fn, sub in [(_s1_tp, "s1.tp"), (_s1_fp, "s1.fp"), (_s1_fn, "s1.fn"), (_s1_phase, "s1.phase")]:
        res = fn(traj, step, context)
        if res is not None:
            out[sub] = res
    return out


# Auto-register sub-signals
SignalRegistry.register("s1.tp", _s1_tp, SignalSpec("s1.tp", "S1", "PF activated on risky step", 0.25))
SignalRegistry.register("s1.fp", _s1_fp, SignalSpec("s1.fp", "S1", "PF activated on safe step (penalty)", -0.10))
SignalRegistry.register("s1.fn", _s1_fn, SignalSpec("s1.fn", "S1", "Risky step with no PF activation (penalty)", -0.10))
SignalRegistry.register("s1.phase", _s1_phase, SignalSpec("s1.phase", "S1", "Normalized step position of activation", 0.05))
