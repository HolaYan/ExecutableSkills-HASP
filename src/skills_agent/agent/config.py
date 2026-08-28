"""
SkillAgentConfig — configuration for the skill-enhanced agent.
"""

from typing import Optional, List
from dataclasses import dataclass, field

_VALID_SKILL_SOURCE_FORMATS = ("auto", "json", "markdown")


@dataclass
class SkillAgentConfig:
    """Configuration for the skill-enhanced agent."""

    # Skill library path (file or directory)
    skill_library_path: str = "configs/skills/default_skills.json"

    # Skill source format: "auto" (detect from path), "json", "markdown"
    skill_source_format: str = "auto"

    # Skill selection
    max_skills_in_prompt: int = 3  # Max skills injected into system prompt
    enable_step_reminders: bool = True  # Enable per-step skill reminders

    # Skill selector weights
    mode_weight: float = 0.5
    trigger_weight: float = 0.5

    # Phase-gated injection
    enable_phase_injection: bool = True  # Master switch for phase-gated injection
    max_phase_instructions: int = 1  # Max phase instructions per observation
    pre_final_step_threshold: int = 3  # Inject pre-final check after N steps

    # Program functions (direct intervention on every step)
    enable_program_functions: bool = True  # Enable program function checks on every step
    disabled_program_functions: List[str] = field(default_factory=list)  # PF skill_ids to skip

    # Code-level skill handlers
    enable_skill_handlers: bool = True  # Enable code-level handlers (pure-code + LLM-assisted)
    teacher_api_provider: Optional[str] = None  # "anthropic" | "openai" | "google"
    teacher_api_model: Optional[str] = None  # e.g. "claude-sonnet-4-20250514", "gpt-5"
    teacher_api_key: Optional[str] = None  # API key for PF helper
    handler_vote_threshold: int = 3  # Min handlers that must trigger before intervention

    # Multi-PF helper deliberation (Plan 3)
    enable_multi_teacher: bool = False  # Enable multi-model deliberation
    teacher_models: List[dict] = field(default_factory=list)  # [{provider, model_name, api_key}]
    deliberation_strategy: str = "majority"  # "majority" | "unanimous" | "any"

    # Ablation controls
    skills_enabled: bool = True  # Master switch for skills
    compact_format: bool = False  # Use compact skill format (fewer tokens)
    pf_only_mode: bool = False  # When True: skills ONLY through PF/OT (no prompt injection)
    enable_prompt_only_skills: bool = False  # Inject non-PF skills via prompt even in pf_only_mode

    # Adaptive skill activation (difficulty gating)
    enable_difficulty_gating: bool = False  # Enable dynamic difficulty-based skill activation
    difficulty_model: Optional[str] = None  # API model key for difficulty estimation (e.g., "gpt4o_mini")
    difficulty_threshold: int = 3  # Difficulty score >= threshold → enable skills (1-5 scale)

    # PF selection (PF helper selects top-K PFs per question)
    enable_pf_selection: bool = False  # Enable dynamic PF selection
    pf_selection_model: Optional[str] = None  # API model key (e.g., "gpt4o_mini"); None = heuristic
    pf_selection_provider: Optional[str] = None  # Resolved provider for PF selection model
    pf_selection_model_name: Optional[str] = None  # Resolved model name
    pf_selection_api_key: Optional[str] = None  # Resolved API key
    pf_top_k: int = 10  # Max PFs to select per question

    # Self-judge mode: route both PFSelector and PF dispatch through the BASE
    # inference model (the one being evaluated), not an external API PF helper.
    # When True, skill_agent_runner builds a tokenizer-aware shim around the
    # vLLM model and passes it as `teacher_model`. PFs that have
    # `needs_helper=True` will then receive the base model in `intervene()`.
    pf_self_judge: bool = False

    # Multi-round PF helper retry: PF helper judges answer, retries if insufficient
    enable_teacher_retry: bool = False  # Enable PF helper-based answer validation and retry
    max_retry_rounds: int = 5  # Max outer retry rounds (total rounds including first attempt)

    # PF helper-based answer format postprocessing (runs on ALL questions, independent of skills)
    enable_teacher_format_postprocess: bool = False  # Use PF helper to normalize final answer format
    format_postprocess_val_dir: Optional[str] = None  # Path to validation data dir for few-shot examples
    format_postprocess_test_dir: Optional[str] = None  # Fallback: test data dir
    format_postprocess_dataset_name: Optional[str] = None  # Current dataset name (set per-dataset)

    # Output
    save_skill_snapshots: bool = True  # Save skill library snapshots with results
    save_trajectories: bool = True  # Save full message trajectories per episode

    def __post_init__(self):
        if self.skill_source_format not in _VALID_SKILL_SOURCE_FORMATS:
            raise ValueError(
                f"skill_source_format must be one of {_VALID_SKILL_SOURCE_FORMATS}, "
                f"got {self.skill_source_format!r}"
            )
