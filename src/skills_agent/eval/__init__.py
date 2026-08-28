"""
Evaluation module — copied from SFT_CISPO with skill extensions.
"""

from .evaluator import Evaluator, EvalConfig
from .data_loader import load_test_data, load_local_data, load_hf_dataset, create_sample_data
from .model_loader import (
    load_model_for_eval,
    load_base_model,
    load_sft_model,
    load_rl_model,
    ModelType,
)
from .agent_runner import AgentRunner, RunnerConfig
from .tools import ToolEnvironment, SerpAPISearch, WebReader, GPTSummarizer, SearchResult
from .metrics import compute_metrics, aggregate_metrics, aggregate_pass_at_k, EpisodeMetrics, AggregatedMetrics
from .episode import Episode, Step, Action, Observation, Evidence, AttackMetadata

# Skill extensions (lazy import to avoid circular dependency with agent module)
from .skill_episode import SkillEpisode
from .skill_metrics import (
    SkillEpisodeMetrics,
    SkillAggregatedMetrics,
    compute_skill_metrics,
    aggregate_skill_metrics,
    compute_skill_effectiveness_report,
)

__all__ = [
    # Original modules
    "Evaluator",
    "EvalConfig",
    "load_test_data",
    "load_local_data",
    "load_hf_dataset",
    "create_sample_data",
    "load_model_for_eval",
    "load_base_model",
    "load_sft_model",
    "load_rl_model",
    "ModelType",
    "AgentRunner",
    "RunnerConfig",
    "ToolEnvironment",
    "SerpAPISearch",
    "WebReader",
    "GPTSummarizer",
    "SearchResult",
    "compute_metrics",
    "aggregate_metrics",
    "aggregate_pass_at_k",
    "EpisodeMetrics",
    "AggregatedMetrics",
    "Episode",
    "Step",
    "Action",
    "Observation",
    "Evidence",
    "AttackMetadata",
    # Skill extensions
    "SkillEpisode",
    "SkillEpisodeMetrics",
    "SkillAggregatedMetrics",
    "compute_skill_metrics",
    "aggregate_skill_metrics",
    "compute_skill_effectiveness_report",
]
