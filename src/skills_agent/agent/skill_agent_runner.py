"""
SkillAgentRunner — extends AgentRunner with phase-gated skill injection.

Key changes from base AgentRunner:
1. _build_system_prompt() — compact skill awareness priming in system prompt
2. _format_observation() — phase-gated skill instructions after observations (sync path)
3. _post_tool_observation() — phase-gated skill instructions (async path)
4. _pre_dispatch_intervention() — reactive override with code-level handlers (Layer 0)
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import logging
import re

from ..eval.agent_runner import AgentRunner, RunnerConfig
from ..eval.episode import Episode, Action, Observation, Step, Evidence, AttackMetadata
from ..eval.tools import ToolEnvironment
from ..eval.metrics import compute_metrics, EpisodeMetrics
from ..eval.model_loader import VLLMModelWrapper, APIModelWrapper, load_model_api

from ..skills.skill import Skill, SkillLibrary
from ..skills.selector import SkillSelector
from ..skills.output_detector import OutputProblemDetector
from ..skills.skill_handlers import (
    execute_handler, execute_handler_multi_teacher,
    HandlerRecord, DeliberationRecord,
)
from ..skills.program_functions import (
    execute_program_functions, PFRecord, get_all_program_functions,
    execute_observation_transformers,
)
from ..skills.pf_selector import PFSelector
from ..skills.prompts import (
    format_skills_section,
    format_skills_compact,
    format_skills_awareness,
    format_phase_instruction,
    format_step_reminder,
)
from .config import SkillAgentConfig
from ..skills.difficulty_gate import DifficultyGate

logger = logging.getLogger(__name__)

# Maps each skill_id to its handler_id (or a legacy action string).
# Legacy actions are prefixed with "_legacy:" and handled separately.
_SKILL_HANDLER_MAP: Dict[str, str] = {
    # Legacy handlers (3) — handled inline
    "insufficient_exploration": "_legacy:force_read_or_search",
    "retrieval_failure": "_legacy:reformulate_search",
    "format_extraction_error": "_legacy:postprocess_answer",
    # Pure-code handlers (5)
    "temporal_confusion": "verify_temporal_claims",
    "numerical_reasoning_error": "verify_numerical_claims",
    "negation_oversight": "check_negation",
    "outdated_information": "check_source_freshness",
    "citation_mismatch": "verify_citations",
    # LLM-assisted handlers (7)
    "multi_hop_reasoning_failure": "verify_reasoning_chain",
    "answer_completeness": "verify_answer_relevance",
    "adversarial_distraction": "verify_adversarial_distraction",
    "hallucination": "verify_hallucination_grounding",
    "reading_comprehension_error": "verify_reading_comprehension",
    "reasoning_error": "verify_reasoning_steps",
    "wrong_entity_confusion": "verify_entity_disambiguation",
    "language_barrier": "check_language_handling",
    # New skills (Plan 2) — no separate handlers, PFs handle intervention
    # Reasoning / Distillation
    "decompose_complex_question": None,
    "evidence_synthesis": None,
    "comparison_analyzer": None,
    # Search Optimization
    "query_decomposition": None,
    "iterative_refinement": None,
    "search_depth_controller": None,
    # Adversarial Defense
    "claim_triangulation": None,
    "misinformation_detector": None,
}


class _SelfTeacherShim:
    """Adapter that exposes a PF helper-like ``generate(messages=...)`` API but
    routes the call through the BASE inference model (vLLM) instead of an
    external API provider.

    Used by `pf_self_judge` mode so that PFSelector and PF dispatch use the
    student's own model in place of a GPT-4o PF helper — keeps the whole
    ablation arm self-contained (no OpenAI calls).
    """

    def __init__(self, vllm_model, tokenizer):
        self._vllm = vllm_model
        self._tok = tokenizer

    def _run(self, messages, max_tokens: int, temperature: float, top_p: float):
        if not messages:
            return ""
        try:
            from vllm import SamplingParams
        except Exception:
            return ""
        try:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Best-effort: concatenate plain text if template fails.
            prompt = "\n".join(m.get("content", "") for m in messages)
        sp = SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        try:
            llm = getattr(self._vllm, "llm", None) or self._vllm
            outs = llm.generate(
                [prompt], sp,
                lora_request=getattr(self._vllm, "lora_request", None),
            )
            if not outs or not outs[0].outputs:
                return ""
            return outs[0].outputs[0].text or ""
        except Exception:
            return ""

    def generate(self, messages=None, prompt: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.0,
                 top_p: float = 1.0, **_):
        # Two callers in-tree pass `prompt=` instead of `messages=`. Wrap.
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        return self._run(messages, max_tokens, temperature, top_p)

    def generate_from_messages(self, messages, max_tokens: int = 512,
                               temperature: float = 0.0, top_p: float = 1.0, **_):
        return self._run(messages, max_tokens, temperature, top_p)


class SkillAgentRunner(AgentRunner):
    """
    Skill-enhanced agent runner with phase-gated injection.

    Extends AgentRunner with:
    - Compact skill awareness priming in system prompt
    - Phase-gated skill instructions injected after observations
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[RunnerConfig] = None,
        env: Optional[ToolEnvironment] = None,
        skill_library: Optional[SkillLibrary] = None,
        skill_selector: Optional[SkillSelector] = None,
        skill_config: Optional[SkillAgentConfig] = None,
    ):
        super().__init__(model, tokenizer, config, env)

        self.skill_config = skill_config or SkillAgentConfig()

        # Initialize skill components
        if skill_library is None:
            skill_library = self._load_skill_library()
        self.skill_library = skill_library

        # Auto-load `dynamic_program_functions.py` (if any) sitting next to the
        # skill library so its @register_pf side-effects populate the global
        # PF registry. This makes eval-time runs (run_skill_eval.py) see the
        # same code-domain PFs as training-time runs do via the explicit
        # _load_dynamic_pfs in skill_rollout.py — no more `avg_pf_activations: 0`.
        try:
            import importlib.util
            from pathlib import Path as _P
            _lib = _P(str(self.skill_config.skill_library_path))
            for _candidate in (_lib / "dynamic_program_functions.py",
                                _lib.parent / "dynamic_program_functions.py"):
                if _candidate.is_file():
                    _mod_name = f"_dyn_pfs_{abs(hash(str(_candidate))) & 0xFFFFFF:06x}"
                    _spec = importlib.util.spec_from_file_location(_mod_name, str(_candidate))
                    if _spec and _spec.loader:
                        _m = importlib.util.module_from_spec(_spec)
                        _spec.loader.exec_module(_m)
                        logger.info("[SkillAgentRunner] Loaded dynamic PFs from %s", _candidate)
                        break
        except Exception as _e:
            logger.warning("[SkillAgentRunner] dynamic PF auto-load skipped: %s", _e)

        if skill_selector is None:
            skill_selector = SkillSelector(
                library=self.skill_library,
                mode_weight=self.skill_config.mode_weight,
                trigger_weight=self.skill_config.trigger_weight,
            )
        self.skill_selector = skill_selector

        # Output problem detector (Layer 0)
        self._output_detector = OutputProblemDetector()

        # PF helper for LLM-assisted handlers and helper-backed PFs
        self._teacher_model = None
        self._teacher_models = []  # Multi-PF helper list (Plan 3)
        # Init PF helper if needed by handlers OR by PF-only mode (helper-backed PFs)
        _needs_teacher = (
            (self.skill_config.enable_skill_handlers or self.skill_config.pf_only_mode)
            and self.skill_config.teacher_api_provider
        )
        if _needs_teacher:
            self._init_teacher_model()
        if self.skill_config.enable_multi_teacher and self.skill_config.teacher_models:
            self._init_multi_teachers()

        # Difficulty gate for adaptive skill activation
        self._difficulty_gate = None
        if self.skill_config.enable_difficulty_gating:
            self._init_difficulty_gate()

        # Self-judge shim: when pf_self_judge is on, the BASE inference model
        # plays the role of PF helper in PFSelector + PF dispatch.
        self._self_teacher = None
        if self.skill_config.pf_self_judge:
            self._self_teacher = _SelfTeacherShim(self.model, self.tokenizer)
            logger.info(
                "[SelfJudge] Enabled — base model will be used as PF helper in "
                "PFSelector and PF dispatch"
            )

        # PF selector for dynamic PF selection per question
        self._pf_selector = None
        if self.skill_config.enable_pf_selection:
            self._init_pf_selector()

        # Track current episode skill state
        self._current_active_skills: List[Skill] = []
        self._current_step_reminders: List[Optional[str]] = []
        # Handler activity records (per-episode, reset in run_episode)
        self._handler_records: List[HandlerRecord] = []
        # Deliberation records (per-episode, Plan 3)
        self._deliberation_records: List[DeliberationRecord] = []
        # Program function records (per-episode)
        self._pf_records: List[PFRecord] = []
        self._pf_context_injections: List[str] = []  # pending injections for next obs
        # Active PF IDs for current episode (set by PF selector, None = use disabled list)
        self._active_pf_ids: Optional[List[str]] = None

    # ========================================================================
    # Skill library loading (auto-detect format)
    # ========================================================================

    def _load_skill_library(self) -> SkillLibrary:
        """Load skill library based on ``skill_source_format`` config."""
        path = self.skill_config.skill_library_path
        fmt = self.skill_config.skill_source_format

        p = Path(path)
        if fmt == "auto":
            fmt = "markdown" if p.is_dir() else "json"

        if fmt == "markdown":
            return SkillLibrary.load_from_directory(path)
        else:
            return SkillLibrary(path)

    # ========================================================================
    # Override 1: System prompt with compact skill awareness
    # ========================================================================

    def _build_system_prompt(
        self,
        with_examples: bool = True,
        question: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """Build system prompt with skill injection.

        If difficulty gating is enabled, skills are only injected when the
        question is estimated to be hard enough to benefit from guidance.
        """
        base = super()._build_system_prompt(with_examples)

        if not self.skill_config.skills_enabled:
            return base

        # Difficulty gating: skip skills for easy questions
        if self._difficulty_gate and question:
            if not self._difficulty_gate.should_enable_skills(question):
                self._current_active_skills = []
                return base

        if question and mode:
            selected = self.skill_selector.select(
                question=question,
                mode=mode,
                max_skills=self.skill_config.max_skills_in_prompt,
            )
        else:
            all_skills = self.skill_library.get_all()
            all_skills.sort(key=lambda s: s.skill_id)
            selected = all_skills[: self.skill_config.max_skills_in_prompt]

        self._current_active_skills = selected

        # PF-only mode: skills work entirely through PF/OT, no prompt injection
        # BUT: if enable_prompt_only_skills is set, inject skills that have no PF handler
        if self.skill_config.pf_only_mode:
            if self.skill_config.enable_prompt_only_skills:
                prompt_only = [s for s in selected if self._is_prompt_only_skill(s)]
                if prompt_only:
                    skills_text = format_skills_awareness(prompt_only)
                    base += "\n\n" + skills_text
            return base

        if selected:
            if self.skill_config.enable_phase_injection:
                skills_text = format_skills_awareness(selected)
            elif self.skill_config.compact_format:
                skills_text = format_skills_compact(selected)
            else:
                skills_text = format_skills_section(selected)
            base += "\n\n" + skills_text

        return base

    @staticmethod
    def _is_prompt_only_skill(skill) -> bool:
        """Check if a skill has no PF handler (prompt-only)."""
        handler = _SKILL_HANDLER_MAP.get(skill.skill_id)
        return handler is None

    # ========================================================================
    # Override 2: _format_observation (sync path, phase-gated injection)
    # ========================================================================

    def _format_observation(
        self, obs_text: str, action_type: str, step_context: Dict[str, Any]
    ) -> str:
        """Inject phase-gated skill instructions after observation (sync path)."""
        if action_type in ("READ", "SUMMARY"):
            step_context["all_read_contents"] = step_context.get("all_read_contents", "") + "\n" + obs_text

        if not self.skill_config.skills_enabled:
            return obs_text

        # Observation transformers (programmatic, runs before text injection)
        if self.skill_config.enable_program_functions:
            active_ids = self._get_active_pf_ids()
            _teacher = self._teacher_model if self._teacher_model else (
                self._teacher_models[0] if self._teacher_models else None
            )
            obs_text = execute_observation_transformers(
                active_ids, obs_text, action_type, step_context,
                teacher_model=_teacher,
            )

        # Flush pending PF context injections
        if self._pf_context_injections:
            obs_text += "\n".join(self._pf_context_injections)
            self._pf_context_injections.clear()

        # PF-only mode: skip prompt-based injection for PF skills
        # BUT: still inject phase instructions for prompt-only skills if enabled
        if self.skill_config.pf_only_mode:
            if self.skill_config.enable_prompt_only_skills:
                return self._inject_phase_instructions(
                    obs_text, action_type, step_context, prompt_only=True,
                )
            return obs_text

        if self.skill_config.enable_phase_injection:
            return self._inject_phase_instructions(obs_text, action_type, step_context)

        return self._maybe_add_reminder(obs_text, step_context)

    def _inject_phase_instructions(
        self, obs_text: str, action_type: str, step_context: Dict[str, Any],
        prompt_only: bool = False,
    ) -> str:
        """Core phase-gated injection logic (shared by sync and async paths).

        Args:
            prompt_only: If True, only inject for skills without PF handlers.
        """
        phase = {
            "SEARCH": "post_search",
            "READ": "post_read",
            "SUMMARY": "post_read",
        }.get(action_type)

        if not phase:
            return obs_text

        injections = []
        for skill in self._current_active_skills:
            # Filter: if prompt_only, skip skills that have PF handlers
            if prompt_only and not self._is_prompt_only_skill(skill):
                continue
            if not skill.phase_instructions:
                continue
            pi = skill.phase_instructions.get(phase)
            if pi is None:
                continue
            # Check conditions
            from ..skills.conditions import ConditionEvaluator
            if not ConditionEvaluator.evaluate(pi.conditions, step_context):
                continue
            formatted = format_phase_instruction(skill.name, pi.instruction)
            if formatted:
                injections.append(formatted)

        if not injections:
            return obs_text

        max_inj = self.skill_config.max_phase_instructions
        if max_inj and len(injections) > max_inj:
            injections = injections[:max_inj]

        return obs_text + "\n\n" + "\n".join(injections)

    # ========================================================================
    # Override 3: run_episode with skill tracking (sync path)
    # ========================================================================

    def run_episode(
        self,
        question: str,
        gold_answers: List[str] = None,
        sample_id: str = None,
        mode: str = "clean",
        model_name: str = "model",
        seed: int = None,
        attack_metadata: AttackMetadata = None,
    ) -> Episode:
        """Run a single episode with skill tracking."""
        self._current_active_skills = []
        self._current_step_reminders = []
        self._handler_records = []
        self._deliberation_records = []
        self._pf_records = []
        self._pf_context_injections = []

        # Select PFs for this episode
        if self._pf_selector:
            self._active_pf_ids = self._pf_selector.select(question)
        else:
            self._active_pf_ids = None

        episode = super().run_episode(
            question=question,
            gold_answers=gold_answers,
            sample_id=sample_id,
            mode=mode,
            model_name=model_name,
            seed=seed,
            attack_metadata=attack_metadata,
        )

        # Attach handler, deliberation, and PF records to episode
        episode.handler_records = [r.to_dict() for r in self._handler_records] if self._handler_records else []
        episode.deliberation_records = [r.to_dict() for r in self._deliberation_records] if self._deliberation_records else []
        episode.pf_records = [r.to_dict() for r in self._pf_records] if self._pf_records else []

        return episode

    # ========================================================================
    # PF helper initialization
    # ========================================================================

    def _init_teacher_model(self):
        """Lazy-init the PF helper for LLM-assisted handlers."""
        cfg = self.skill_config
        try:
            self._teacher_model, _ = load_model_api(
                provider=cfg.teacher_api_provider,
                model_name=cfg.teacher_api_model,
                api_key=cfg.teacher_api_key,
                max_tokens=200,
                temperature=0.0,
            )
            logger.info(
                f"PF helper initialized: {cfg.teacher_api_provider}/{cfg.teacher_api_model}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize the PF helper: {e}")
            self._teacher_model = None

    def _init_multi_teachers(self):
        """Initialize multiple PF helper models for multi-perspective deliberation."""
        for teacher_cfg in self.skill_config.teacher_models:
            try:
                model, _ = load_model_api(
                    provider=teacher_cfg["provider"],
                    model_name=teacher_cfg["model_name"],
                    api_key=teacher_cfg.get("api_key"),
                    max_tokens=200,
                    temperature=0.0,
                )
                self._teacher_models.append(model)
                logger.info(
                    f"Multi-PF helper model initialized: "
                    f"{teacher_cfg['provider']}/{teacher_cfg['model_name']}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to init multi-teacher {teacher_cfg.get('model_name')}: {e}"
                )

    def _init_difficulty_gate(self):
        """Initialize difficulty gate for adaptive skill activation."""
        teacher_model = None
        cfg = self.skill_config

        # Try to use the specified difficulty model
        if cfg.difficulty_model:
            # Reuse an existing PF helper if available
            if self._teacher_models:
                teacher_model = self._teacher_models[0]
                logger.info("[DifficultyGate] Reusing first PF helper for difficulty estimation")
            elif self._teacher_model:
                teacher_model = self._teacher_model
                logger.info("[DifficultyGate] Reusing PF helper for difficulty estimation")

        self._difficulty_gate = DifficultyGate(
            teacher_model=teacher_model,
            threshold=cfg.difficulty_threshold,
        )
        mode = "LLM" if teacher_model else "heuristic"
        logger.info(
            f"[DifficultyGate] Initialized ({mode} mode, threshold={cfg.difficulty_threshold})"
        )

    def _init_pf_selector(self):
        """Initialize PF selector for dynamic per-question PF selection."""
        teacher_model = None
        cfg = self.skill_config

        # Self-judge mode short-circuits external PF helper resolution: the base
        # model itself picks the top-K PFs.
        if cfg.pf_self_judge and self._self_teacher is not None:
            self._pf_selector = PFSelector(
                teacher_model=self._self_teacher,
                top_k=cfg.pf_top_k,
            )
            logger.info(
                f"[PFSelector] Initialized (self-judge mode, top_k={cfg.pf_top_k})"
            )
            return

        if cfg.pf_selection_model:
            # Check if the PF selection model is the SAME as the PF helper
            # If different, create a dedicated model; if same, reuse PF helper
            teacher_is_same = (
                cfg.pf_selection_provider == cfg.teacher_api_provider
                and cfg.pf_selection_model_name == cfg.teacher_api_model
            )
            if teacher_is_same and self._teacher_model:
                teacher_model = self._teacher_model
                logger.info("[PFSelector] Reusing PF helper (same provider/model)")
            elif teacher_is_same and self._teacher_models:
                teacher_model = self._teacher_models[0]
                logger.info("[PFSelector] Reusing first PF helper (same provider/model)")
            else:
                # Create a dedicated model for PF selection
                try:
                    teacher_model, _ = load_model_api(
                        provider=cfg.pf_selection_provider,
                        model_name=cfg.pf_selection_model_name,
                        api_key=cfg.pf_selection_api_key,
                        max_tokens=200,
                        temperature=0.0,
                    )
                    logger.info(
                        f"[PFSelector] Created dedicated model: "
                        f"{cfg.pf_selection_provider}/{cfg.pf_selection_model_name}"
                    )
                except Exception as e:
                    logger.warning(f"[PFSelector] Failed to create model: {e}")

        self._pf_selector = PFSelector(
            teacher_model=teacher_model,
            top_k=cfg.pf_top_k,
        )
        mode = "LLM" if teacher_model else "heuristic"
        logger.info(f"[PFSelector] Initialized ({mode} mode, top_k={cfg.pf_top_k})")

    def _get_active_pf_ids(self, ep_data: Optional[Dict] = None) -> List[str]:
        """Get active PF IDs for the current episode.

        Priority: ep_data["active_pf_ids"] > self._active_pf_ids > skill-based IDs.

        When PFSelector is disabled and the SkillLibrary contains many
        prompt-only skills with non-PF names (e.g. code domain: 3 registered
        PFs + 57 evolved generic skills), the skill-based fallback returns
        skill_ids that aren't in `_PF_REGISTRY`, so dispatch silently no-ops.
        Final fallback returns ALL registered PFs and lets each PF's own
        `should_activate(step_context, ...)` gate by domain.
        """
        # Async path: check ep_data first
        if ep_data is not None:
            pf_ids = ep_data.get("active_pf_ids")
            if pf_ids is not None:
                return pf_ids

        # Sync path: use PF selector results
        if self._active_pf_ids is not None:
            return self._active_pf_ids

        # Fallback: use skill-based IDs (legacy 1-to-1 skill_id==pf_id path)
        if self._current_active_skills:
            return [s.skill_id for s in self._current_active_skills]

        # Check ep_data active_skills (async fallback)
        if ep_data is not None:
            active_skills = ep_data.get("active_skills", [])
            if active_skills:
                return [s.skill_id for s in active_skills]

        # Last resort: every domain-relevant registered PF is a candidate.
        return self._domain_pf_ids()

    def _domain_pf_ids(self) -> List[str]:
        """All registered PFs that match the runner's domain.

        Convention:
          • code domain   → only `code_*` PFs
          • math domain   → only the math PF set in math_program_functions.py
          • web_search    → everything else (the original PF_REGISTRY)
        We use a name prefix for code (`code_*`) and a hardcoded list for
        math because math PFs predate the prefix convention. This stops
        cross-domain leak (e.g. `final_format_error` rewriting code FINALs).
        """
        try:
            from ..skills.program_functions import get_all_program_functions
        except Exception:
            return []
        all_ids = list(get_all_program_functions().keys())
        domain = getattr(self.config, "domain", "web_search")
        _MATH_IDS = {
            "arithmetic_slip", "algebraic_sign_error", "case_incompleteness",
            "boundary_violation", "substitution_invalid",
            "simplification_incomplete", "overgeneralization",
            "units_dimension_mismatch", "final_format_error",
            "proof_step_gap", "verification_missing",
        }
        if domain == "code":
            return [k for k in all_ids if k.startswith("code_")]
        if domain == "math":
            return [k for k in all_ids if k in _MATH_IDS]
        # web_search: everything except math + code
        return [k for k in all_ids
                if not k.startswith("code_") and k not in _MATH_IDS]

    # ========================================================================
    # Override 4: _pre_dispatch_intervention (Layer 0 reactive override)
    # ========================================================================

    def _pre_dispatch_intervention(
        self, ep_data, action_type, arg, reasoning
    ) -> Tuple[str, str]:
        """Override: execute handlers before tool dispatch.

        On FINAL actions: force-run ALL handlers (pure-code + LLM-assisted)
        unconditionally, regardless of detector output.
        On non-FINAL actions: use OutputProblemDetector to gate legacy handlers.
        """
        if not self.skill_config.skills_enabled:
            return action_type, arg

        # ── Program Functions (run every step) ──
        if self.skill_config.enable_program_functions:
            step_context_pf = ep_data.get("step_context")
            if step_context_pf is None:
                step_context_pf = {}
            # Use PF selector results if available; otherwise fall back to skill-based IDs
            active_ids = self._get_active_pf_ids(ep_data)
            if not active_ids:
                return action_type, arg  # No PFs to run
            _disabled_pfs = None if self._active_pf_ids is not None else (
                set(self.skill_config.disabled_program_functions)
                if self.skill_config.disabled_program_functions else None
            )
            # Pass PF helper to PFs that need it (e.g., answer extraction, reasoning check)
            _teacher = self._teacher_model if self._teacher_model else (
                self._teacher_models[0] if self._teacher_models else None
            )
            # Self-judge mode: when no external PF helper is configured, fall
            # back to the base-model shim so `needs_helper=True` PFs still
            # get a callable rather than NOOP-ing.
            if _teacher is None and self._self_teacher is not None:
                _teacher = self._self_teacher
            pf_action, pf_arg, pf_records, pf_injections = execute_program_functions(
                active_skill_ids=active_ids,
                step_context=step_context_pf,
                action_type=action_type,
                arg=arg,
                reasoning=reasoning or "",
                disabled_pfs=_disabled_pfs,
                teacher_model=_teacher,
            )
            # Collect records
            records_dest = ep_data.get("pf_records", self._pf_records)
            records_dest.extend(pf_records)
            # Store context injections for next _format_observation call
            inj_dest = ep_data.get("pf_context_injections", self._pf_context_injections)
            inj_dest.extend(pf_injections)

            if pf_action != action_type or pf_arg != arg:
                action_type, arg = pf_action, pf_arg
            elif (
                action_type == "FINAL"
                and pf_injections
                and getattr(self.skill_config, "final_revision_on_inject", True)
                and not ep_data.get("_final_revision_done")
            ):
                # HASP FINAL-revision: Case-C evidence injected at FINAL would
                # otherwise be dropped (no next observation). Convert to the
                # base runner's RETRY path, which feeds the injections back as
                # an Observation and lets the model produce ONE revised FINAL
                # (matching pf_select eval's Turn-2). Once per episode; the
                # original answer stands if the model re-commits it.
                ep_data["_final_revision_done"] = True
                # RETRY branch consumes self._pf_context_injections; move the
                # just-added injections there so they become the feedback text.
                for x in pf_injections:
                    if inj_dest is not self._pf_context_injections:
                        try:
                            inj_dest.remove(x)
                        except ValueError:
                            pass
                        self._pf_context_injections.append(x)
                action_type = "RETRY"

        if not self.skill_config.enable_skill_handlers:
            return action_type, arg

        # Build step context
        step_context = ep_data.get("step_context")
        if step_context is None:
            step_context = {
                "question": ep_data.get("episode", ep_data).question
                    if hasattr(ep_data.get("episode", ep_data), "question")
                    else ep_data.get("question", ""),
                "has_read": ep_data.get("has_read", False),
                "read_count": ep_data.get("read_count", 0),
                "search_count": ep_data.get("search_count", 0),
                "step_count": ep_data.get("step_count", 0),
                "all_read_contents": ep_data.get("all_read_contents", ""),
                "last_search_results_text": ep_data.get("last_search_results_text", ""),
                "action_history": ep_data.get("action_history", []),
                "max_steps": self.config.max_steps,
            }

        # Build shared handler context
        records_list = (
            ep_data.get("handler_records", self._handler_records)
            if ep_data else self._handler_records
        )
        handler_context = {
            "question": step_context.get("question", ""),
            "thought": reasoning or "",
            "action_arg": arg,
            "all_read_contents": step_context.get("all_read_contents", ""),
            "last_search_results_text": step_context.get("last_search_results_text", ""),
            "read_count": step_context.get("read_count", 0),
            "search_count": step_context.get("search_count", 0),
            "step_count": step_context.get("step_count", 0),
            "max_steps": step_context.get("max_steps", self.config.max_steps),
            "has_read": step_context.get("has_read", False),
            "action_history": step_context.get("action_history", []),
            "_handler_records": records_list,
            "step_context": step_context,  # Mutable ref for cross-step state
        }

        if action_type == "FINAL":
            # Force-run ALL handlers unconditionally on FINAL
            return self._execute_all_handlers(
                action_type, arg, reasoning, handler_context, records_list,
                ep_data=ep_data,
            )

        # Non-FINAL: use detector to gate legacy handlers only
        problems = self._output_detector.detect(
            thought=reasoning or "",
            action_type=action_type,
            action_arg=arg,
            step_context=step_context,
        )
        if not problems:
            return action_type, arg

        return self._execute_reactive_override(
            problems, action_type, arg, handler_context, records_list,
        )

    def _execute_all_handlers(
        self,
        action_type: str,
        arg: str,
        reasoning: str,
        handler_context: Dict[str, Any],
        records_list: list,
        ep_data=None,
    ) -> Tuple[str, str]:
        """Force-run ALL registered handlers on FINAL with vote-based intervention.

        Every pure-code and LLM-assisted handler is executed. If the number of
        triggered handlers meets or exceeds ``handler_vote_threshold``, the FINAL
        action is overridden to SEARCH (forcing re-exploration). Otherwise, the
        action proceeds as-is (audit-only).

        High-precision pure-code handlers (verify_temporal_claims,
        verify_numerical_claims, verify_citations) always trigger intervention
        individually, regardless of the vote count.

        When multi-PF helper is enabled, LLM-assisted handlers use multi-model
        deliberation with consensus strategy (majority/unanimous/any).
        """
        # Handlers that are precise enough to intervene on their own
        # (verify_citations removed — heuristic entity matching has too many false positives)
        _SOLO_OVERRIDE_HANDLERS = {
            "verify_temporal_claims",
            "verify_numerical_claims",
        }

        use_multi = (
            self.skill_config.enable_multi_teacher
            and len(self._teacher_models) >= 2
        )
        strategy = self.skill_config.deliberation_strategy

        triggered_count = 0
        triggered_ids = []
        solo_override = None  # First solo-override intervention text

        # De-duplicate: same handler_id mapped from multiple skill_ids
        seen_handlers = set()

        from ..skills.skill_handlers import get_handler

        for skill_id, handler_id in _SKILL_HANDLER_MAP.items():
            if handler_id is None:
                continue
            if handler_id.startswith("_legacy:"):
                continue
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)

            # Check if this is an LLM-assisted handler
            entry = get_handler(handler_id)
            is_llm_assisted = entry is not None and entry[1]  # requires_api

            if use_multi and is_llm_assisted:
                # Multi-PF helper deliberation
                result, delib_record = execute_handler_multi_teacher(
                    handler_id, handler_context, self._teacher_models,
                    strategy=strategy, skill_id=skill_id,
                )
                if delib_record is not None:
                    delib_dest = (
                        ep_data.get("deliberation_records", self._deliberation_records)
                        if ep_data else self._deliberation_records
                    )
                    delib_dest.append(delib_record)
            else:
                # Single-PF helper or pure-code handler
                result = execute_handler(
                    handler_id, handler_context, self._teacher_model,
                    skill_id=skill_id,
                )

            if result is not None:
                triggered_count += 1
                triggered_ids.append(handler_id)
                logger.info(f"[vote] {skill_id}: handler '{handler_id}' fired ({triggered_count} total)")

                # Solo-override for high-precision handlers
                if handler_id in _SOLO_OVERRIDE_HANDLERS and solo_override is None:
                    solo_override = result

        threshold = self.skill_config.handler_vote_threshold

        # Guard: if the agent already attempted FINAL before in this episode,
        # do NOT override again (prevents the FINAL→SEARCH→FINAL→SEARCH loop).
        step_ctx = handler_context.get("step_context", {})
        prior_finals = step_ctx.get("_prior_final_overrides", 0)
        _MAX_FINAL_OVERRIDES = 1  # Allow at most 1 handler-driven FINAL→SEARCH

        # Solo override: high-precision pure-code handler fired
        if solo_override is not None and prior_finals < _MAX_FINAL_OVERRIDES:
            step_ctx["_prior_final_overrides"] = prior_finals + 1
            logger.info(
                f"[intervention] Solo override by high-precision handler. "
                f"Overriding FINAL → SEARCH."
            )
            return "SEARCH", handler_context.get("question", arg)

        # Vote-based override: enough handlers agree there's a problem
        if triggered_count >= threshold and prior_finals < _MAX_FINAL_OVERRIDES:
            step_ctx["_prior_final_overrides"] = prior_finals + 1
            logger.info(
                f"[intervention] Vote threshold met: {triggered_count}/{threshold} "
                f"handlers triggered ({triggered_ids}). Overriding FINAL → SEARCH."
            )
            return "SEARCH", handler_context.get("question", arg)

        if prior_finals >= _MAX_FINAL_OVERRIDES and (solo_override is not None or triggered_count >= threshold):
            logger.info(
                f"[audit] Handler override suppressed (already overridden {prior_finals} time(s)). "
                f"Proceeding with FINAL."
            )

        if triggered_count > 0:
            logger.info(
                f"[audit] {triggered_count}/{threshold} handlers triggered "
                f"(below threshold) — proceeding with FINAL."
            )

        return action_type, arg

    def _execute_reactive_override(
        self,
        problems,
        action_type: str,
        arg: str,
        handler_context: Dict[str, Any],
        records_list: list,
    ) -> Tuple[str, str]:
        """Apply legacy handlers for detector-identified problems (non-FINAL)."""
        for problem in problems:
            handler_id = _SKILL_HANDLER_MAP.get(problem.skill_id)
            if handler_id is None:
                continue

            if handler_id == "_legacy:force_read_or_search":
                if not handler_context.get("has_read"):
                    logger.info(f"[reactive] {problem.skill_id}: overriding → SEARCH")
                    records_list.append(HandlerRecord(
                        handler_id=handler_id, skill_id=problem.skill_id,
                        handler_type="legacy", result="triggered",
                        intervention_text="Override → SEARCH",
                    ))
                    return "SEARCH", handler_context.get("question", arg)

            elif handler_id == "_legacy:reformulate_search":
                if action_type == "SEARCH":
                    words = arg.split()
                    if len(words) > 10:
                        shortened = " ".join(words[:8])
                        logger.info(f"[reactive] retrieval_failure: shortening query")
                        records_list.append(HandlerRecord(
                            handler_id=handler_id, skill_id=problem.skill_id,
                            handler_type="legacy", result="triggered",
                            intervention_text=f"Shortened query to: {shortened}",
                        ))
                        return action_type, shortened

        return action_type, arg

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _maybe_add_reminder(self, obs_text, step_context):
        """Legacy: add a step-level skill reminder to the observation."""
        if not self.skill_config.enable_step_reminders:
            return obs_text

        skill = self.skill_selector.select_for_step(step_context)
        if skill is not None:
            reminder = format_step_reminder(skill, step_context)
            self._current_step_reminders.append(reminder)
            return obs_text + reminder

        self._current_step_reminders.append(None)
        return obs_text

    @staticmethod
    def _detect_contradictions(results) -> bool:
        """Simple heuristic to detect contradictory information in search results."""
        if len(results) < 2:
            return False
        snippets = []
        for r in results:
            snippet = getattr(r, "snippet", "") or ""
            if isinstance(r, dict):
                snippet = r.get("snippet", "")
            snippets.append(snippet.lower())
        contradiction_indicators = [
            "however", "contrary", "incorrect", "not true",
            "actually", "in fact", "disputed", "false",
        ]
        for snippet in snippets:
            for indicator in contradiction_indicators:
                if indicator in snippet:
                    return True
        return False

    def get_active_skill_ids(self) -> List[str]:
        """Get IDs of skills active in the current/last episode."""
        return [s.skill_id for s in self._current_active_skills]

    def get_step_reminders(self) -> List[Optional[str]]:
        """Get step reminders from the current/last episode."""
        return list(self._current_step_reminders)

    # ========================================================================
    # Override: _postprocess_answer — strip meta-commentary and artifacts
    # ========================================================================

    # Patterns to strip from the beginning of the answer
    _ANSWER_PREFIX_PATTERNS = [
        r'^The answer is:?\s*',
        r'^Based on my research,?\s*',
        r'^After reviewing[\w\s]*,\s*',
        r'^From the[\w\s]+,\s*',
        r'^According to[\w\s]+,\s*',
        r'^In summary,?\s*',
    ]
    # Patterns to strip from the end of the answer
    _ANSWER_SUFFIX_PATTERNS = [
        r'\s*\(note:.*?\)\s*$',
        r'\s*\[note:.*?\]\s*$',
        r'\s*\(Source:.*?\)\s*$',
        r'\s*\[\d+\]\s*$',
    ]

    def _postprocess_answer(self, answer, question="", step_context=None):
        """Post-process final answer: strip meta-commentary, formatting artifacts."""
        if not answer:
            return answer

        cleaned = answer.strip()

        # Handle TRUNCATED marker: extract content before it
        if "TRUNCATED" in cleaned:
            before_trunc = cleaned.split("TRUNCATED")[0].strip()
            if before_trunc and len(before_trunc) > 3:
                cleaned = before_trunc.rstrip(".[… ")

        # Strip prefix meta-commentary
        for pattern in self._ANSWER_PREFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, count=1, flags=re.IGNORECASE).strip()

        # Strip suffix meta-commentary
        for pattern in self._ANSWER_SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Strip markdown formatting artifacts
        # Bold: **text** → text
        cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
        # Headers: ## text → text
        cleaned = re.sub(r'^#+\s*', '', cleaned)

        # Strip meta-text leakage from prompt template (e.g., "answer): Provide your final answer...")
        cleaned = re.sub(r'\banswer\)\s*:?\s*(Provide|should|Your|When)[\s\S]*$', '', cleaned, flags=re.IGNORECASE).strip()

        # Strip "FINAL answer should be..." type leakage
        cleaned = re.sub(r'\bFINAL\s+answer\s+should\b.*$', '', cleaned, flags=re.IGNORECASE).strip()

        # Strip reasoning that leaked into answer (e.g. '"Bong Joon-ho"), so the answer...')
        reasoning_match = re.match(r'^"?(.+?)"?\s*\)\s*[,;]?\s*(?:so\s+the|but\s+the|the\s+system)', cleaned, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            candidate = reasoning_match.group(1).strip().strip('"').strip("'").strip()
            if candidate and len(candidate) >= 2:
                cleaned = candidate

        # If answer starts with a quote and rest is meta-text, keep only quoted part
        quote_match = re.match(r'^["\'](.+?)["\'](.*)$', cleaned, re.DOTALL)
        if quote_match:
            quoted_part = quote_match.group(1).strip()
            rest = quote_match.group(2).strip()
            # If the rest looks like meta-commentary
            if rest and re.match(r'^[\s,.]*(which|note|based|according|source|however|so\s+the|but\s+the)', rest, re.IGNORECASE):
                cleaned = quoted_part

        # PF helper-based format postprocessing (runs on ALL questions)
        cleaned = self._teacher_format_postprocess(cleaned, question)

        return cleaned if cleaned else answer

    _FORMAT_PROMPT_HEADER = (
        "You are a strict answer-format normalizer. Given a question and a raw answer from an agent, "
        "output ONLY the core answer string, in the EXACT format used by the gold examples below.\n"
        "Rules (follow ALL):\n"
        "- Match the style, granularity, capitalization, and length of the gold answers exactly.\n"
        "- Strip ALL reasoning, explanation, meta-text, prefixes like 'The answer is', 'Based on', 'According to'.\n"
        "- For yes/no questions, output exactly 'yes' or 'no' (lowercase).\n"
        "- For dates, preserve the gold-answer date format (e.g. '1984', 'March 5, 1984', '1984-03-05').\n"
        "- For entity/name answers, output just the entity — no articles, no commentary.\n"
        "- For numerical answers, output just the number — include units only if the gold answers do.\n"
        "- If the raw answer already matches the gold format, return it as-is with no change.\n"
        "- NEVER invent new information; only reformat what's in the raw answer.\n"
        "- If the raw answer has no valid content, return the raw answer unchanged.\n"
        "- Output ONLY the formatted answer. NO quotes, NO labels, NO explanation.\n\n"
    )

    # Cache: dataset_name -> list of (question, gold_answer) few-shot pairs
    _format_fewshot_cache: Dict[str, List[Tuple[str, str]]] = {}

    def _load_format_fewshot(self) -> List[Tuple[str, str]]:
        """Load dataset-specific few-shot Q/A pairs from validation or test data."""
        import json as _json

        ds_name = self.skill_config.format_postprocess_dataset_name
        if not ds_name:
            return []

        if ds_name in self._format_fewshot_cache:
            return self._format_fewshot_cache[ds_name]

        # Try validation dir first, then test dir
        candidates = []
        for dir_path in [
            self.skill_config.format_postprocess_val_dir,
            self.skill_config.format_postprocess_test_dir,
        ]:
            if dir_path:
                p = Path(dir_path) / f"{ds_name}.jsonl"
                if p.exists():
                    candidates.append(p)
                    break

        pairs = []
        if candidates:
            try:
                with open(candidates[0], "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = _json.loads(line)
                        q = item.get("question", "")
                        a = item.get("answer", "")
                        if q and a:
                            pairs.append((q, a))
                        if len(pairs) >= 5:  # 5 few-shot examples max
                            break
            except Exception as e:
                logger.warning(f"[format_postprocess] Failed to load few-shot data: {e}")

        self._format_fewshot_cache[ds_name] = pairs
        if pairs:
            logger.info(f"[format_postprocess] Loaded {len(pairs)} few-shot examples for {ds_name}")
        return pairs

    def _build_format_prompt(self, question: str, answer: str) -> str:
        """Build format postprocess prompt with dataset-specific few-shot examples."""
        parts = [self._FORMAT_PROMPT_HEADER]

        fewshot = self._load_format_fewshot()
        if fewshot:
            parts.append("Gold answer examples from this dataset:\n")
            for q, a in fewshot:
                parts.append(f"Q: {q}\nGold: {a}\n\n")
        parts.append(f"Now format this answer to match the style above:\n")
        parts.append(f"Q: {question}\nRaw: {answer}\nA:")
        return "".join(parts)

    @staticmethod
    def _needs_format_postprocess(answer):
        """Quick heuristic: does this answer look like it needs PF helper reformatting?"""
        if not answer or len(answer) <= 3:
            return False
        # Already concise (≤5 words, no sentence markers) → skip
        words = answer.split()
        if len(words) <= 5 and not answer.endswith('.') and 'answer' not in answer.lower():
            return False
        # Triggers: verbose, has sentence structure, meta-text, reasoning leakage
        if len(words) > 8:
            return True
        if answer.endswith('.') and len(words) > 3:
            return True
        if any(kw in answer.lower() for kw in ('the answer', 'based on', 'according', 'answer)', 'final answer', 'so the')):
            return True
        if answer[0] == '"' and ')' in answer:
            return True
        return False

    def _teacher_format_postprocess(self, answer, question=""):
        """
        Unified format normalization via PF helper.
        Called on ALL final answers before returning (not gated by heuristic).
        Uses dataset-specific few-shot examples to align answer format with gold.
        """
        if not self.skill_config.enable_teacher_format_postprocess:
            return answer
        if not answer or len(answer.strip()) == 0:
            return answer

        teacher = self._teacher_model or (
            self._teacher_models[0] if self._teacher_models else None
        )
        if teacher is None:
            return answer

        try:
            result = teacher.generate(
                messages=[{"role": "user", "content": self._build_format_prompt(
                    question=question, answer=answer,
                )}],
                max_tokens=100,
                temperature=0.0,
            )
            if result and result.strip():
                formatted = result.strip().strip('"').strip("'").strip()
                # Strip trailing "A:" artifacts from few-shot leakage
                formatted = re.sub(r'\s*A:\s*$', '', formatted).strip()
                # Sanity: result should be non-empty and bounded length (prevent runaway generation)
                if formatted and len(formatted) <= 200:
                    if formatted != answer:
                        logger.debug(
                            f"[format_postprocess] '{answer}' → '{formatted}'"
                        )
                    return formatted
        except Exception as e:
            from ..skills.quota import note_api_error
            if note_api_error(e): raise
            logger.warning(f"[format_postprocess] PF helper call failed: {e}")

        return answer

    def get_handler_records(self) -> List[Dict[str, Any]]:
        """Get handler activity records from the current/last episode.

        Returns a list of dicts, each representing one handler invocation with:
        - handler_id, skill_id, handler_type, result
        - teacher_prompt, teacher_response, teacher_model_name (for LLM-assisted)
        - latency_ms, error_message (if applicable)
        """
        return [r.to_dict() for r in self._handler_records]

    # ========================================================================
    # Async batch hooks (override base AgentRunner hooks)
    # ========================================================================

    def _build_async_messages(self, episode, mode, use_examples):
        """Override: pass question/mode to skill selector for system prompt injection."""
        system_prompt = self._build_system_prompt(
            with_examples=use_examples,
            question=episode.question,
            mode=mode,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {episode.question}"},
        ]

    def _init_async_ep_data(self, ep_data, sample, mode):
        """Override: store per-episode skill state from the initialization loop."""
        ep_data["active_skills"] = list(self._current_active_skills)
        ep_data["step_reminders"] = []
        ep_data["handler_records"] = []  # Collect handler records per async episode
        ep_data["deliberation_records"] = []  # Collect deliberation records per async episode
        ep_data["pf_records"] = []       # Collect PF records per async episode
        ep_data["pf_context_injections"] = []  # Pending PF injections
        # PF selection for async: pick the candidates the dispatch loop
        # will run `should_activate` on for THIS episode.
        #
        # When PFSelector is disabled we used to fall back to
        # `_current_active_skills`, but on a mixed library where prompt-only
        # skills outnumber registered PFs (code domain: 3 PFs + 57 evolved
        # prompt-only skills), the SkillSelector picks 10 skills by
        # question-text relevance. On prose-heavy questions (MBPP+ / BCB)
        # zero of the 3 registered code PFs make the cut, so dispatch
        # silently skipped them. Fix: when PFSelector is off, pass the full
        # set of registered PFs filtered to the current domain. Each PF's
        # own `should_activate(step_context, ...)` does the rest.
        if self._pf_selector:
            ep_data["active_pf_ids"] = self._pf_selector.select(ep_data["episode"].question)
        else:
            ep_data["active_pf_ids"] = self._domain_pf_ids()
        ep_data["step_context"] = {
            "step_count": 0,
            "has_read": False,
            "search_count": 0,
            "read_count": 0,
            "empty_results": False,
            "contradictory_sources": False,
            "similar_entity_results": False,
            "max_steps": self.config.max_steps,
            "action_history": [],
            "question": ep_data["episode"].question,
            "last_search_results_text": "",
            "all_read_contents": "",
            # PFs gate on this (e.g. code_* PFs check `domain == "code"`).
            # Without this key the gating heuristic falls back to keyword-
            # matching the question, which works most of the time but is
            # not reliable. Setting domain explicitly is the right fix.
            "domain": getattr(self.config, "domain", "web_search"),
            # Code-domain test data (passed through normalize_samples). These
            # let the sandbox quick-check PF run the model's draft FINAL.
            #
            # Two shapes coexist:
            #   • LCB-style: per-test list under public_tests/private_tests.
            #   • EvalPlus + BigCodeBench: a single combined-driver script in
            #     `eval_test_code`, plus `entry_point`. Sandbox PFs prefer the
            #     driver path when present (`evaluate_with_test_script`).
            "public_tests": sample.get("public_tests", []),
            "private_tests": sample.get("private_tests", []),
            "starter_code": sample.get("starter_code", ""),
            "func_name": (sample.get("metadata") or {}).get("func_name", "")
                or sample.get("entry_point", ""),
            "platform": sample.get("platform", ""),
            "eval_test_code": sample.get("eval_test_code", ""),
            # Doctest-derived public examples (visible to model in the
            # docstring) — used by code_sandbox_quick_check as a clean
            # correctness signal that's not data leakage.
            "public_test_code": sample.get("public_test_code", ""),
            "entry_point": sample.get("entry_point", ""),
            "variant": sample.get("variant", ""),
        }

    def _post_tool_observation(self, ep_data, action_type):
        """Override: inject phase-gated instructions or legacy reminders after tool observation."""
        step_context = ep_data.get("step_context")
        if step_context is None:
            return

        last_msg = ep_data["messages"][-1]["content"] if ep_data["messages"] else ""
        if action_type == "SEARCH":
            step_context["search_count"] = ep_data["search_count"]
            step_context["empty_results"] = "No results found" in last_msg
            step_context["last_search_results_text"] = last_msg if "No results found" not in last_msg else ""
        elif action_type == "READ":
            step_context["has_read"] = True
            step_context["read_count"] = ep_data.get("read_count", 0)
            step_context["all_read_contents"] = step_context.get("all_read_contents", "") + "\n" + last_msg
        step_context["step_count"] = ep_data["step_count"]

        action_history = step_context.get("action_history")
        if action_history is not None:
            trace = ep_data["episode"].trace
            arg = trace[-1].action.doc_id if trace else ""
            action_history.append({
                "action_type": action_type,
                "arg": arg,
                "step": ep_data["step_count"],
            })

        # Observation transformers (async path, programmatic)
        if self.skill_config.enable_program_functions and self.skill_config.skills_enabled:
            active_ids = self._get_active_pf_ids(ep_data)
            if active_ids:
                _teacher = self._teacher_model if self._teacher_model else (
                    self._teacher_models[0] if self._teacher_models else None
                )
                old_content = ep_data["messages"][-1]["content"]
                new_content = execute_observation_transformers(
                    active_ids, old_content, action_type, step_context,
                    teacher_model=_teacher,
                )
                if new_content != old_content:
                    ep_data["messages"][-1]["content"] = new_content

        # Flush pending PF context injections (async path)
        pf_injections = ep_data.get("pf_context_injections", [])
        if pf_injections:
            ep_data["messages"][-1]["content"] += "\n".join(pf_injections)
            pf_injections.clear()

        # PF-only mode: skip prompt-based injection for PF skills (async path)
        # BUT: still inject for prompt-only skills if enabled
        if self.skill_config.pf_only_mode:
            if not self.skill_config.enable_prompt_only_skills:
                return
            # Fall through to inject phase instructions for prompt-only skills only

        if self.skill_config.enable_phase_injection and self.skill_config.skills_enabled:
            active_skills = ep_data.get("active_skills", [])
            saved = self._current_active_skills
            self._current_active_skills = active_skills
            prompt_only_flag = self.skill_config.pf_only_mode  # filter to prompt-only if in pf_only_mode
            try:
                old_text = ep_data["messages"][-1]["content"]
                new_text = self._inject_phase_instructions(
                    old_text, action_type, step_context, prompt_only=prompt_only_flag,
                )
                if new_text != old_text:
                    ep_data["messages"][-1]["content"] = new_text
                    ep_data["step_reminders"].append(new_text[len(old_text):])
                else:
                    ep_data["step_reminders"].append(None)
            finally:
                self._current_active_skills = saved
        elif self.skill_config.enable_step_reminders:
            skill = self.skill_selector.select_for_step(step_context)
            if skill is not None:
                reminder = format_step_reminder(skill, step_context)
                ep_data["step_reminders"].append(reminder)
                ep_data["messages"][-1]["content"] += reminder
            else:
                ep_data["step_reminders"].append(None)

    def _on_async_episode_done(self, ep_data):
        """Override: attach handler and PF records to episode."""
        records = ep_data.get("handler_records", [])
        delib_records = ep_data.get("deliberation_records", [])
        pf_records = ep_data.get("pf_records", [])
        ep = ep_data.get("episode")
        if ep is not None:
            ep.handler_records = [r.to_dict() for r in records] if records else []
            ep.deliberation_records = [r.to_dict() for r in delib_records] if delib_records else []
            ep.pf_records = [r.to_dict() for r in pf_records] if pf_records else []
