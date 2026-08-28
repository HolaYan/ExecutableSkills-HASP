"""S3 — Intervention content quality.

Three sub-signals:
  - s3.syntactic: action args are syntactically valid
  - s3.semantic:  PF helper-judged appropriateness (loaded from Helper review)
  - s3.domain:    domain-specific content quality (plugin)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .registry import SignalRegistry, SignalSpec, SignalOutput

VALID_ACTIONS = {"SEARCH", "READ", "FINAL"}


def _s3_syntactic(traj, step, context) -> Optional[SignalOutput]:
    """Check the *final* action is a valid ReAct action."""
    act_type = step.final_action_type
    arg = step.final_action_arg or ""
    ok = act_type in VALID_ACTIONS and len(arg.strip()) > 0
    return SignalOutput("s3.syntactic", 1.0 if ok else 0.0, traj.sample_id, step.step_index)


def _s3_semantic(traj, step, context) -> Optional[SignalOutput]:
    """PF helper-judged semantic quality.

    Expected in `context['teacher_step_scores']` — a dict
    {(sample_id, step_index): float in [0,1]}. Falls back to
    `local_improvement` heuristic if not provided.
    """
    if context:
        scores = context.get("teacher_step_scores") or {}
        key = (traj.sample_id, step.step_index)
        if key in scores:
            return SignalOutput("s3.semantic", float(scores[key]), traj.sample_id, step.step_index)

    # Fallback heuristic
    if step.proposed_action_type == "FINAL" and step.final_action_type in {"SEARCH", "READ"}:
        return SignalOutput("s3.semantic", 0.7, traj.sample_id, step.step_index)
    if step.was_modified:
        return SignalOutput("s3.semantic", 0.5, traj.sample_id, step.step_index)
    return SignalOutput("s3.semantic", 0.3, traj.sample_id, step.step_index)


def _s3_domain(traj, step, context) -> Optional[SignalOutput]:
    """Domain-specific quality. By default this is a no-op (0.0).
    Domain plugins (math, coding) override by registering
    `s3.domain.<name>` and this function aggregates them.
    """
    domain_scores = []
    if context:
        for key, fn in context.get("_domain_signal_fns", {}).items():
            res = fn(traj, step, context)
            if res is not None:
                domain_scores.append(res.value)
    value = sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
    return SignalOutput("s3.domain", value, traj.sample_id, step.step_index)


def compute_s3(traj, step, context: Optional[Dict[str, Any]] = None) -> Dict[str, SignalOutput]:
    return {
        "s3.syntactic": _s3_syntactic(traj, step, context),
        "s3.semantic": _s3_semantic(traj, step, context),
        "s3.domain": _s3_domain(traj, step, context),
    }


SignalRegistry.register("s3.syntactic", _s3_syntactic, SignalSpec("s3.syntactic", "S3", "Syntactic validity of final action", 0.20))
SignalRegistry.register("s3.semantic", _s3_semantic, SignalSpec("s3.semantic", "S3", "Teacher-judged semantic quality", 0.50))
SignalRegistry.register("s3.domain", _s3_domain, SignalSpec("s3.domain", "S3", "Domain-specific content quality (plugin)", 0.30))
