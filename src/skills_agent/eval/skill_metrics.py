"""
Skill-specific metrics for evaluating skill effectiveness.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .metrics import EpisodeMetrics, AggregatedMetrics, compute_metrics
from .episode import Episode


@dataclass
class SkillEpisodeMetrics(EpisodeMetrics):
    """Metrics for a single episode with skill tracking."""
    active_skill_ids: List[str] = field(default_factory=list)
    num_step_reminders: int = 0
    # Program function metrics
    pf_activations: int = 0
    pf_modify_actions: int = 0
    pf_inject_contexts: int = 0
    # Deliberation metrics (Plan 3)
    deliberation_count: int = 0
    deliberation_splits: int = 0  # Cases where PF helpers disagreed


@dataclass
class SkillAggregatedMetrics(AggregatedMetrics):
    """Aggregated metrics including skill-specific stats."""
    # Overall skill metrics
    avg_skills_per_episode: float = 0.0
    avg_reminders_per_episode: float = 0.0
    # Program function metrics
    avg_pf_activations: float = 0.0
    avg_pf_modify_actions: float = 0.0
    avg_pf_inject_contexts: float = 0.0
    # Deliberation metrics
    avg_deliberation_count: float = 0.0
    avg_deliberation_splits: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "avg_skills_per_episode": self.avg_skills_per_episode,
            "avg_reminders_per_episode": self.avg_reminders_per_episode,
            "avg_pf_activations": self.avg_pf_activations,
            "avg_pf_modify_actions": self.avg_pf_modify_actions,
            "avg_pf_inject_contexts": self.avg_pf_inject_contexts,
            "avg_deliberation_count": self.avg_deliberation_count,
            "avg_deliberation_splits": self.avg_deliberation_splits,
        })
        return result


def compute_skill_metrics(
    episode: Episode,
    gold_answers: Optional[List[str]] = None,
    active_skill_ids: Optional[List[str]] = None,
    num_step_reminders: int = 0,
) -> SkillEpisodeMetrics:
    """
    Compute metrics for a single episode with skill information.
    """
    base = compute_metrics(episode, gold_answers)

    # Count PF activations from episode records
    pf_records = getattr(episode, "pf_records", []) or []
    pf_activations = sum(1 for r in pf_records if r.get("activated", False))
    pf_modify_actions = sum(
        1 for r in pf_records
        if r.get("activated") and r.get("intervention_type") == "modify_action"
    )
    pf_inject_contexts = sum(
        1 for r in pf_records
        if r.get("activated") and r.get("intervention_type") == "inject_context"
    )

    # Count deliberation records
    delib_records = getattr(episode, "deliberation_records", []) or []
    deliberation_count = len(delib_records)
    deliberation_splits = sum(1 for r in delib_records if r.get("consensus") == "split")

    return SkillEpisodeMetrics(
        exact_match=base.exact_match,
        f1_score=base.f1_score,
        cover_exact_match=base.cover_exact_match,
        has_read=base.has_read,
        step_count=base.step_count,
        search_count=base.search_count,
        read_count=base.read_count,
        valid_structure=base.valid_structure,
        active_skill_ids=active_skill_ids or [],
        num_step_reminders=num_step_reminders,
        pf_activations=pf_activations,
        pf_modify_actions=pf_modify_actions,
        pf_inject_contexts=pf_inject_contexts,
        deliberation_count=deliberation_count,
        deliberation_splits=deliberation_splits,
    )


def aggregate_skill_metrics(
    metrics_list: List[SkillEpisodeMetrics],
) -> SkillAggregatedMetrics:
    """
    Aggregate skill-enhanced metrics across multiple episodes.
    """
    if not metrics_list:
        return SkillAggregatedMetrics()

    n = len(metrics_list)

    # Standard aggregation
    base_agg = AggregatedMetrics(
        answer_em=sum(m.exact_match for m in metrics_list) / n,
        answer_f1=sum(m.f1_score for m in metrics_list) / n,
        answer_cem=sum(m.cover_exact_match for m in metrics_list) / n,
        has_read_rate=sum(m.has_read for m in metrics_list) / n,
        avg_steps=sum(m.step_count for m in metrics_list) / n,
        avg_search_calls=sum(m.search_count for m in metrics_list) / n,
        avg_read_calls=sum(m.read_count for m in metrics_list) / n,
        valid_structure_rate=sum(m.valid_structure for m in metrics_list) / n,
        num_samples=n,
    )

    # Skill-specific aggregations
    total_skills = sum(len(m.active_skill_ids) for m in metrics_list)
    total_reminders = sum(m.num_step_reminders for m in metrics_list)
    total_pf_acts = sum(m.pf_activations for m in metrics_list)
    total_pf_modify = sum(m.pf_modify_actions for m in metrics_list)
    total_pf_inject = sum(m.pf_inject_contexts for m in metrics_list)
    total_delib = sum(m.deliberation_count for m in metrics_list)
    total_delib_splits = sum(m.deliberation_splits for m in metrics_list)

    return SkillAggregatedMetrics(
        answer_em=base_agg.answer_em,
        answer_f1=base_agg.answer_f1,
        answer_cem=base_agg.answer_cem,
        has_read_rate=base_agg.has_read_rate,
        avg_steps=base_agg.avg_steps,
        avg_search_calls=base_agg.avg_search_calls,
        avg_read_calls=base_agg.avg_read_calls,
        valid_structure_rate=base_agg.valid_structure_rate,
        num_samples=n,
        avg_skills_per_episode=total_skills / n,
        avg_reminders_per_episode=total_reminders / n,
        avg_pf_activations=total_pf_acts / n,
        avg_pf_modify_actions=total_pf_modify / n,
        avg_pf_inject_contexts=total_pf_inject / n,
        avg_deliberation_count=total_delib / n,
        avg_deliberation_splits=total_delib_splits / n,
    )


def compute_skill_effectiveness_report(
    metrics_list: List[SkillEpisodeMetrics],
) -> Dict[str, Any]:
    """
    Generate a detailed skill effectiveness report.

    Returns:
        Dictionary with per-skill breakdown and overall statistics
    """
    if not metrics_list:
        return {}

    # Separate episodes by whether skills were active
    with_skills = [m for m in metrics_list if m.active_skill_ids]
    without_skills = [m for m in metrics_list if not m.active_skill_ids]

    report = {
        "total_episodes": len(metrics_list),
        "episodes_with_skills": len(with_skills),
        "episodes_without_skills": len(without_skills),
    }

    if with_skills:
        report["with_skills_em"] = sum(m.exact_match for m in with_skills) / len(with_skills)
        report["with_skills_f1"] = sum(m.f1_score for m in with_skills) / len(with_skills)

    if without_skills:
        report["without_skills_em"] = sum(m.exact_match for m in without_skills) / len(without_skills)
        report["without_skills_f1"] = sum(m.f1_score for m in without_skills) / len(without_skills)

    # Per-skill breakdown
    skill_breakdown = {}
    for m in metrics_list:
        for skill_id in m.active_skill_ids:
            if skill_id not in skill_breakdown:
                skill_breakdown[skill_id] = {"total": 0, "success": 0, "f1_sum": 0.0}
            skill_breakdown[skill_id]["total"] += 1
            if m.exact_match:
                skill_breakdown[skill_id]["success"] += 1
            skill_breakdown[skill_id]["f1_sum"] += m.f1_score

    for skill_id, stats in skill_breakdown.items():
        stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_f1"] = stats["f1_sum"] / stats["total"] if stats["total"] > 0 else 0.0
        del stats["f1_sum"]

    report["per_skill_breakdown"] = skill_breakdown

    return report
