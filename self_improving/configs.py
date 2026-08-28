"""
Configuration classes for the self-improving pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ValidationSetConfig:
    """Config for building the validation set.

    Seed and validation data come from the SAME datasets as the test set,
    but use only the TAIL portion (samples after the test set boundary).
    Datasets fully consumed by the test set (e.g. Bamboogle, GAIA) are
    automatically excluded.
    """
    # Path to the full source data (same JSONL files used by inference)
    # Resolved from the environment, like every other out-of-repo path in this
    # project — see hasp_paths.py. Empty means "not configured yet", which the
    # validation stage reports rather than failing on a stranger's filesystem.
    source_dir: str = ""
    # Path to inference results (used to detect exact test set boundaries)
    inference_results_dir: str = "outputs/skill_eval_adaptive_qwem3_30b/skill_eval_best/three_forced_skills"
    # Fallback: if inference results not found, assume this many test samples
    default_test_samples: int = 200
    # Per-dataset overrides for the test-set size, takes precedence over
    # both the detected count and default_test_samples. Use this to
    # shrink the test boundary for a dataset so its remaining samples
    # become available for seed/validation (e.g. Bamboogle has only 125
    # total; shrinking test to 60 frees 65 for seed/val).
    test_samples_overrides: Dict[str, int] = field(default_factory=dict)
    # The datasets to use (same 7 datasets as the test set)
    datasets: List[str] = field(default_factory=lambda: [
        "2WikiMultihopQA_rand1000",   # total=1000, test=200, avail=800
        "HotpotQA_rand1000",          # total=1000, test=200, avail=800
        "Bamboogle",                  # total=125,  test=125, avail=0 (auto-skipped)
        "BrowseComp",                 # total=295,  test=200, avail=95
        "DeepResearcher_rand1000",    # total=1000, test=200, avail=800
        "GAIA",                       # total=103,  test=103, avail=0 (auto-skipped)
        "frames",                     # total=824,  test=200, avail=624
    ])
    # Number of samples per dataset for seed (training signal for self-improving)
    seed_samples_per_dataset: int = 50
    # Number of samples per dataset for validation (measuring generalization)
    val_samples_per_dataset: int = 50
    # Relative output dir (under self_improving/) for persisting seed/validation
    # splits. Defaults to "data" (web_search). MATH domain uses "data_math".
    data_subdir: str = "data"


@dataclass
class SkillProposalConfig:
    """Config for student-based skill proposal."""
    # Max candidate skills per epoch
    max_candidates_per_epoch: int = 5
    # Min failure cluster size to trigger proposal
    min_cluster_size: int = 3
    # Student model for proposal generation (key in api_models)
    student_model: str = "gpt"
    # Temperature for proposal generation
    temperature: float = 0.7
    # Failure-analysis strategy:
    #   "heuristic" — regex/rule-based pattern detectors (legacy)
    #   "llm"       — PF helper abstracts each failed trajectory into a
    #                 (failure_abstraction, trigger, intervention) triple,
    #                 then dedup by Jaccard similarity. Targets E4-E6 where
    #                 the seed library has no direct coverage for residuals.
    #   "both"      — run heuristic first, then LLM to fill gaps.
    analyzer_mode: str = "both"
    # Max concurrent PF helper calls when analyzer_mode is llm/both
    llm_analyzer_concurrency: int = 8
    # Jaccard similarity threshold for merging LLM-summarized patterns
    llm_dedup_threshold: float = 0.5


@dataclass
class SkillReviewConfig:
    """Config for PF helper-based skill quality review."""
    # PF helper for review (key in api_models)
    teacher_model: str = "gpt"
    # Acceptance threshold (weighted quality score)
    acceptance_threshold: float = 0.6
    # Dimension weights for Q_skill = sum(w_i * Q_i)
    weight_concept: float = 0.25
    weight_trigger: float = 0.20
    weight_intervene: float = 0.20
    weight_exec: float = 0.20
    weight_validation: float = 0.15
    # Temperature for review generation
    temperature: float = 0.3


@dataclass
class PseudoGradientConfig:
    """Config for PF-mediated pseudo-gradient computation."""
    # Weights for advantage computation
    alpha_local: float = 0.4       # Local improvement weight
    beta_downstream: float = 0.4   # Downstream success weight
    gamma_cost: float = 0.1        # Extra cost penalty
    delta_side_effect: float = 0.1  # Harmful side-effect penalty
    # Student skill-generation reward scaling
    lambda_validation_gain: float = 0.5


@dataclass
class LibraryConfig:
    """Config for skill library management."""
    # Max total skills in library (prevents uncontrolled growth)
    max_library_size: int = 50
    # New-group creation threshold (higher than same-group refinement)
    new_group_threshold: float = 0.75
    # Same-group refinement threshold
    same_group_threshold: float = 0.60
    # Whether to allow entirely new skill categories
    allow_new_categories: bool = True


@dataclass
class SelfImprovingConfig:
    """Top-level configuration for the self-improving pipeline."""

    # Experiment metadata
    experiment_name: str = "self_improving_v1"
    output_dir: str = "./outputs/self_improving/"

    # Number of self-improving epochs
    num_epochs: int = 3

    # Skill directories under self_improving/skills/
    # seed: initial skills copied from configs/skills/ (read-only reference)
    # generated: new skills produced during self-improving epochs
    # The "active library" at any epoch = seed + generated
    seed_skill_dir: str = "self_improving/skills/seed/"
    generated_skill_dir: str = "self_improving/skills/generated/"
    skill_snapshots_dir: str = "self_improving/skills/snapshots/"

    # Sub-configs
    validation: ValidationSetConfig = field(default_factory=ValidationSetConfig)
    proposal: SkillProposalConfig = field(default_factory=SkillProposalConfig)
    review: SkillReviewConfig = field(default_factory=SkillReviewConfig)
    pseudo_gradient: PseudoGradientConfig = field(default_factory=PseudoGradientConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)

    # Budget constraints (inherited from existing system)
    max_steps: int = 25
    max_search_calls: int = 15
    max_read_calls: int = 15
    timeout_seconds: int = 600

    # Phase A two-stage prefilter:
    # When enabled, Phase A first runs a cheap raw rollout on the seed pool
    # (no PFs, no skill handlers) to tag exact_match per sample. Only the
    # failing samples (up to `prefilter_cap_k` randomly) are fed into the
    # expensive PF-aware Phase A main pass. This focuses compute on the
    # questions the current student actually can't solve.
    prefilter_baseline_failures: bool = False
    prefilter_cap_k: int = 0

    # Model config (key → api_models entry)
    student_model: str = "gpt"
    teacher_model: str = "gpt"

    # vLLM settings for local models (student + base)
    vllm_enabled: bool = True
    vllm_gpu_memory_utilization: float = 0.95
    vllm_max_model_len: int = 32768
    vllm_parallel_episodes: int = 16
    vllm_quantization: Optional[str] = "fp8"
    vllm_max_num_seqs: int = 512

    # API model definitions
    api_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "gpt": {
            "provider": "openai",
            "model_name": "",
            "max_concurrent": 16,
        },
        "gpt4o_mini": {
            "provider": "openai",
            "model_name": "",
            "max_concurrent": 16,
        },
        "claude": {
            "provider": "anthropic",
            "model_name": "claude-sonnet-4-20250514",
            "max_concurrent": 16,
        },
    })

    # API keys (fallback to env vars)
    api_keys: Dict[str, str] = field(default_factory=dict)

    # Base model for ReAct execution (local vLLM or API)
    base_model_path: str = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
    base_model_backend: Optional[str] = None  # None = local vLLM

    # PF selection config (reused from existing system)
    enable_pf_selection: bool = True
    pf_selection_model: str = "gpt"
    pf_top_k: int = 10
    enable_difficulty_gating: bool = True
    difficulty_threshold: int = 3

    # Training data output
    save_training_data: bool = True
    training_data_formats: List[str] = field(default_factory=lambda: ["sft", "dpo"])

    # Domain dispatch — controls system prompt + scoring evaluator.
    #   "web_search" → ReAct + AnswerEvaluator (string EM)
    #   "math"       → math system prompt + MathAnswerEvaluator (numeric/LaTeX)
    #   "code"       → code system prompt + CodeAnswerEvaluator (sandbox pass@1)
    domain: str = "web_search"


def _expand_env(node):
    """Expand `${VAR}` in every string of a loaded config.

    Paths outside the repository are named by environment variable rather than
    hard-coded, so a config can be committed without carrying one machine's
    filesystem in it. An unset variable is left as-is, so the error names the
    variable instead of pointing at a path that never existed.
    """
    import os
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str) and "${" in node:
        return os.path.expandvars(node)
    return node


def load_config_from_yaml(yaml_path: str) -> SelfImprovingConfig:
    """Load SelfImprovingConfig from a YAML file."""
    import yaml
    from pathlib import Path

    with open(yaml_path, "r") as f:
        cfg = _expand_env(yaml.safe_load(f))

    # Build sub-configs
    val_cfg = ValidationSetConfig(**cfg.get("validation", {}))
    proposal_cfg = SkillProposalConfig(**cfg.get("proposal", {}))
    review_cfg = SkillReviewConfig(**cfg.get("review", {}))
    pg_cfg = PseudoGradientConfig(**cfg.get("pseudo_gradient", {}))
    lib_cfg = LibraryConfig(**cfg.get("library", {}))

    # Build top-level
    top = cfg.get("experiment", {})
    models = cfg.get("models", {})
    budget = cfg.get("budget", {})
    vllm = cfg.get("vllm", {})
    skills = cfg.get("skills", {})
    pf_sel = skills.get("pf_selection", {})
    diff_gate = skills.get("difficulty_gating", {})

    return SelfImprovingConfig(
        experiment_name=top.get("name", "self_improving_v1"),
        output_dir=top.get("output_dir", "./outputs/self_improving/"),
        num_epochs=top.get("num_epochs", 3),
        seed_skill_dir=skills.get("seed_skill_dir", "self_improving/skills/seed/"),
        generated_skill_dir=skills.get("generated_skill_dir", "self_improving/skills/generated/"),
        skill_snapshots_dir=skills.get("skill_snapshots_dir", "self_improving/skills/snapshots/"),
        validation=val_cfg,
        proposal=proposal_cfg,
        review=review_cfg,
        pseudo_gradient=pg_cfg,
        library=lib_cfg,
        max_steps=budget.get("max_steps", 25),
        max_search_calls=budget.get("max_search_calls", 15),
        max_read_calls=budget.get("max_read_calls", 15),
        timeout_seconds=budget.get("timeout_seconds", 600),
        student_model=cfg.get("roles", {}).get("student", "gpt"),
        teacher_model=cfg.get("roles", {}).get("teacher", "gpt"),
        vllm_enabled=vllm.get("enabled", True),
        vllm_gpu_memory_utilization=vllm.get("gpu_memory_utilization", 0.95),
        vllm_max_model_len=vllm.get("max_model_len", 32768),
        vllm_parallel_episodes=vllm.get("parallel_episodes", 16),
        vllm_quantization=vllm.get("quantization"),
        vllm_max_num_seqs=vllm.get("max_num_seqs", 512),
        api_models=cfg.get("api_models", {}),
        api_keys=cfg.get("api_keys", {}),
        base_model_path=models.get("base", {}).get("path", "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"),
        base_model_backend=models.get("base", {}).get("backend"),
        enable_pf_selection=pf_sel.get("enabled", True),
        pf_selection_model=pf_sel.get("model", "gpt"),
        pf_top_k=pf_sel.get("top_k", 10),
        enable_difficulty_gating=diff_gate.get("enabled", True),
        difficulty_threshold=diff_gate.get("threshold", 3),
        save_training_data=cfg.get("training", {}).get("save_data", True),
        training_data_formats=cfg.get("training", {}).get("formats", ["sft", "dpo"]),
        domain=cfg.get("domain", "web_search"),
    )
