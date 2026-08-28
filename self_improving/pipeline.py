"""
SelfImprovingPipeline — main orchestrator for the self-improving loop.

Runs K epochs of:
  Phase A: Seed execution (run ReAct with current skills, log PF-aware trajectories)
  Phase B: Failure analysis (cluster recurring residual failures)
  Phase C: Skill proposal (student generates candidate MD + PF)
  Phase D: Skill validation (automated executable checks)
  Phase E: Skill review (PF helper 5-dim quality scoring)
  Phase F: Library update (accept / revise / reject → library evolution)
  Phase G: Pseudo-gradient computation (PF-mediated credit signals)
  Phase H: Training data generation (SFT / DPO samples for post-training)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .configs import SelfImprovingConfig
from .validation_set import ValidationSetManager
from .trajectory_logger import TrajectoryLogger, EpisodeTrajectory
from .failure_analyzer import FailureAnalyzer
from .skill_proposer import SkillProposer, CandidateSkill
from .skill_reviewer import SkillReviewer
from .skill_validator import SkillValidator
from .library_manager import LibraryManager
from .pseudo_gradient import PseudoGradientComputer
from .training_data_builder import TrainingDataBuilder

logger = logging.getLogger(__name__)


class APIModelWrapper:
    """Thin wrapper around API model calls (OpenAI / Anthropic / Google).

    Matches the interface expected by SkillProposer, SkillReviewer, etc.
    """

    def __init__(self, provider: str, model_name: str, api_key: str, max_concurrent: int = 16):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text from the API model."""
        from src.skills_agent.skills.quota import guard, note_api_error
        guard()
        self._ensure_client()

        try:
            if self.provider == "openai":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.content[0].text

            elif self.provider == "google":
                resp = self._client.generate_content(
                    prompt if not system else f"{system}\n\n{prompt}",
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return resp.text

            raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            # Trip the circuit breaker on fatal quota/auth errors; other
            # exceptions propagate normally without tripping.
            note_api_error(e)
            raise


class VLLMStudentWrapper:
    """Wrapper around a vLLM model providing the same ``generate()`` interface.

    Loads the model once via vLLM and reuses the engine for both ReAct
    execution (Phase A) and skill proposal generation (Phase C).
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 32768,
        quantization: str = None,
        max_num_seqs: int = 512,
    ):
        self.model_path = model_path
        self.provider = "vllm"
        self.model_name = model_path
        self._llm = None
        self._tokenizer = None
        # vLLM loading args (deferred until first use)
        self._load_kwargs = dict(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            max_num_seqs=max_num_seqs,
        )

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        logger.info("Loading vLLM student model: %s", self.model_path)
        from src.skills_agent.eval.model_loader import load_model_vllm
        model_wrapper, tok_wrapper = load_model_vllm(
            self.model_path,
            gpu_memory_utilization=self._load_kwargs["gpu_memory_utilization"],
            max_model_len=self._load_kwargs["max_model_len"],
            quantization=self._load_kwargs.get("quantization"),
            max_num_seqs=self._load_kwargs["max_num_seqs"],
        )
        self._llm = model_wrapper.llm          # raw vLLM LLM engine
        self._model_wrapper = model_wrapper     # VLLMModelWrapper (for Phase A)
        self._tokenizer = tok_wrapper
        logger.info("vLLM student loaded: %s", self.model_path)

    # -- Text-in / text-out interface (used by SkillProposer) --

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 3000,
    ) -> str:
        """Generate text using vLLM (chat-template applied via tokenizer)."""
        self._ensure_loaded()
        from vllm import SamplingParams

        # Build chat messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template via the tokenizer
        try:
            templated = self._tokenizer.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: concat system + prompt
            templated = f"{system}\n\n{prompt}" if system else prompt

        sampling = SamplingParams(
            temperature=max(temperature, 0.01),  # vLLM needs temp > 0 for sampling
            max_tokens=max_tokens,
            top_p=0.95,
        )
        outputs = self._llm.generate([templated], sampling_params=sampling)
        return outputs[0].outputs[0].text

    # -- Accessors for Phase A (ReAct execution) --

    @property
    def vllm_model(self):
        """Return the VLLMModelWrapper for use with SkillAgentRunner."""
        self._ensure_loaded()
        return self._model_wrapper

    @property
    def tokenizer(self):
        """Return the VLLMTokenizerWrapper."""
        self._ensure_loaded()
        return self._tokenizer


class SelfImprovingPipeline:
    """Orchestrates the full self-improving loop across K epochs."""

    def __init__(self, config: SelfImprovingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Models (lazy init)
        self._student_model: Optional[APIModelWrapper] = None
        self._teacher_model: Optional[APIModelWrapper] = None

        # Components (initialized in setup())
        self.val_manager: Optional[ValidationSetManager] = None
        self.library_manager: Optional[LibraryManager] = None
        self.training_builder: Optional[TrainingDataBuilder] = None

        # wandb run (lazy init)
        self._wandb_run = None

        # Epoch tracking
        self._epoch_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize all components. Call before run()."""
        logger.info("Setting up self-improving pipeline: %s", self.config.experiment_name)

        # 1. Initialize API models
        self._init_models()

        # 2. Initialize validation set (stored under self_improving/data/)
        self.val_manager = ValidationSetManager(
            config=self.config.validation,
        )
        # Force re-split when sentinel "-1" (use-all-pool) is set — cached
        # split from a prior small-sample run would otherwise stay in use.
        force_rebuild = self.config.validation.seed_samples_per_dataset < 0
        if force_rebuild or not self.val_manager.load_from_saved():
            self.val_manager.load()

        # 3. Initialize library manager (seed + generated under self_improving/skills/)
        self.library_manager = LibraryManager(
            config=self.config.library,
            seed_skill_dir=self.config.seed_skill_dir,
            generated_skill_dir=self.config.generated_skill_dir,
            snapshots_dir=self.config.skill_snapshots_dir,
        )
        self.library_manager.initialize()

        # 4. Initialize training data builder
        self.training_builder = TrainingDataBuilder(
            output_dir=str(self.output_dir),
            formats=self.config.training_data_formats,
        )

        # 5. Initialize wandb
        self._init_wandb()

        # Save config
        self._save_config()
        logger.info("Setup complete. Seed: %d samples, Val: %d samples, Library: %d skills",
                     len(self.val_manager.get_seed_flat()),
                     len(self.val_manager.get_validation_flat()),
                     len(self.library_manager.skill_ids))

    def _init_models(self) -> None:
        """Initialize student and PF helper models.

        Student: either a local vLLM model (if student_model looks like a
        model path, e.g. "Qwen/Qwen3-30B-A3B") or an API model (if it
        matches a key in api_models, e.g. "gpt").

        PF helper: always an API model.
        """
        # --- Student ---
        student_key = self.config.student_model
        if student_key in self.config.api_models:
            # API-based student
            student_cfg = self.config.api_models[student_key]
            student_api_key = self._resolve_api_key(student_cfg.get("provider", "openai"))
            self._student_model = APIModelWrapper(
                provider=student_cfg.get("provider", "openai"),
                model_name=student_cfg.get("model_name", ""),
                api_key=student_api_key,
            )
        else:
            # Local vLLM student (student_key is a HuggingFace model path)
            self._student_model = VLLMStudentWrapper(
                model_path=student_key,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                quantization=self.config.vllm_quantization,
                max_num_seqs=self.config.vllm_max_num_seqs,
            )

        # --- PF helper (always API) ---
        teacher_key = self.config.teacher_model
        teacher_cfg = self.config.api_models.get(teacher_key, {})
        teacher_api_key = self._resolve_api_key(teacher_cfg.get("provider", "openai"))
        self._teacher_model = APIModelWrapper(
            provider=teacher_cfg.get("provider", "openai"),
            model_name=teacher_cfg.get("model_name", ""),
            api_key=teacher_api_key,
        )

        logger.info("Student: %s/%s, PF helper: %s/%s",
                     self._student_model.provider, self._student_model.model_name,
                     self._teacher_model.provider, self._teacher_model.model_name)

    def _resolve_api_key(self, provider: str) -> str:
        """Resolve API key from config or environment."""
        key_map = {
            "openai": ("openai_key", "OPENAI_API_KEY"),
            "anthropic": ("anthropic_key", "ANTHROPIC_API_KEY"),
            "google": ("google_key", "GOOGLE_API_KEY"),
        }
        config_key, env_key = key_map.get(provider, ("", ""))
        return (
            self.config.api_keys.get(config_key)
            or os.environ.get(env_key, "")
        )

    def _init_wandb(self) -> None:
        """Initialize wandb for logging pseudo-gradients, scores, and metrics."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project="self_improving_skills",
                name=self.config.experiment_name,
                config={
                    "num_epochs": self.config.num_epochs,
                    "student_model": self.config.student_model,
                    "teacher_model": self.config.teacher_model,
                    "base_model": self.config.base_model_path,
                    "seed_samples": self.config.validation.seed_samples_per_dataset,
                    "val_samples": self.config.validation.val_samples_per_dataset,
                    "pf_top_k": self.config.pf_top_k,
                    "acceptance_threshold": self.config.review.acceptance_threshold,
                    "max_library_size": self.config.library.max_library_size,
                },
                reinit=True,
            )
            logger.info("wandb initialized: %s", self._wandb_run.url)
        except ImportError:
            logger.info("wandb not installed, skipping logging (pip install wandb)")
            self._wandb_run = None
        except Exception as e:
            logger.warning("wandb init failed: %s", e)
            self._wandb_run = None

    def _log_epoch_wandb(self, epoch: int, result: Dict[str, Any]) -> None:
        """Log epoch metrics to wandb."""
        if not self._wandb_run:
            return
        try:
            import wandb
            metrics = {"epoch": epoch}

            # Phase A: Seed execution
            seed = result.get("phases", {}).get("seed_execution", {})
            metrics["seed/success_rate"] = seed.get("success_rate", 0)
            metrics["seed/total_episodes"] = seed.get("total_episodes", 0)
            metrics["seed/total_pf_activations"] = seed.get("total_pf_activations", 0)

            # Phase B: Failure analysis
            fa = result.get("phases", {}).get("failure_analysis", {})
            metrics["failure/patterns_found"] = fa.get("patterns_found", 0)
            metrics["failure/clusters_found"] = fa.get("clusters_found", 0)

            # Phase C: Skill proposal
            sp = result.get("phases", {}).get("skill_proposal", {})
            metrics["proposal/candidates"] = sp.get("candidates", 0)

            # Phase D: Validation
            sv = result.get("phases", {}).get("skill_validation", {})
            metrics["validation/passed"] = sv.get("passed", 0)
            metrics["validation/failed"] = sv.get("failed", 0)

            # Phase E: Review scores
            sr = result.get("phases", {}).get("skill_review", {})
            scores = sr.get("scores", {})
            if scores:
                metrics["review/mean_q_skill"] = sum(scores.values()) / len(scores)
                metrics["review/max_q_skill"] = max(scores.values())
                metrics["review/min_q_skill"] = min(scores.values())
            decisions = sr.get("decisions", {})
            metrics["review/accepted"] = sum(1 for d in decisions.values() if d == "accept")
            metrics["review/revised"] = sum(1 for d in decisions.values() if d == "revise")
            metrics["review/rejected"] = sum(1 for d in decisions.values() if d == "reject")

            # Phase F: Library
            lib = result.get("phases", {}).get("library_update", {})
            metrics["library/total_size"] = lib.get("library_size", 0)
            metrics["library/accepted_this_epoch"] = lib.get("accepted", 0)

            # Phase G: Pseudo-gradients
            pg = result.get("phases", {}).get("pseudo_gradient", {})
            metrics["gradient/student_action_corrections"] = pg.get("student_action_corrections", 0)
            metrics["gradient/student_risk_signals"] = pg.get("student_risk_signals", 0)
            metrics["gradient/teacher_selection_signals"] = pg.get("teacher_selection_signals", 0)

            metrics["elapsed_seconds"] = result.get("elapsed_seconds", 0)

            wandb.log(metrics, step=epoch)

            # Log per-skill review scores as a table
            if scores:
                table = wandb.Table(columns=["skill_id", "q_skill", "decision"])
                for sid, q in scores.items():
                    table.add_data(sid, q, decisions.get(sid, ""))
                wandb.log({f"review_table/epoch_{epoch}": table}, step=epoch)

        except Exception as e:
            logger.warning("wandb logging failed: %s", e)

    def _save_config(self) -> None:
        import dataclasses
        path = self.output_dir / "config.json"

        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return str(obj)

        path.write_text(json.dumps(dataclasses.asdict(self.config), indent=2, default=_to_dict))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, start_epoch: int = 0) -> Dict[str, Any]:
        """Run the full self-improving pipeline for K epochs."""
        logger.info("Starting self-improving loop: %d epochs", self.config.num_epochs)

        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info("=" * 60)
            logger.info("EPOCH %d / %d", epoch + 1, self.config.num_epochs)
            logger.info("=" * 60)

            epoch_result = self._run_epoch(epoch)
            self._epoch_results.append(epoch_result)

            # Snapshot library
            self.library_manager.snapshot(epoch)

            # Save epoch summary
            self._save_epoch_summary(epoch, epoch_result)

            # Log to wandb
            self._log_epoch_wandb(epoch, epoch_result)

        # Final summary
        summary = self._build_final_summary()
        self._save_final_summary(summary)

        # Close wandb
        if self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        return summary

    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run a single epoch of the self-improving loop."""
        epoch_dir = self.output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        epoch_start = time.time()

        result = {"epoch": epoch, "phases": {}}

        # ============================================================
        # Phase A: Seed execution (run ReAct with current skills)
        # ============================================================
        logger.info("[Phase A] Seed execution...")
        traj_logger = TrajectoryLogger(output_dir=str(self.output_dir), epoch=epoch)
        trajectories = self._phase_a_seed_execution(epoch, traj_logger)
        traj_logger.save()
        result["phases"]["seed_execution"] = {
            "total_episodes": len(trajectories),
            "success_rate": sum(1 for t in trajectories if t.exact_match) / max(len(trajectories), 1),
            "total_pf_activations": sum(t.total_pf_activations for t in trajectories),
        }
        logger.info("Phase A complete: %d episodes, %.1f%% success",
                     len(trajectories),
                     result["phases"]["seed_execution"]["success_rate"] * 100)

        # ============================================================
        # Phase B: Failure analysis
        # ============================================================
        logger.info("[Phase B] Failure analysis...")
        analyzer = FailureAnalyzer(
            existing_skill_ids=self.library_manager.skill_ids,
            min_cluster_size=self.config.proposal.min_cluster_size,
            output_dir=str(epoch_dir / "analysis"),
            mode=getattr(self.config.proposal, "analyzer_mode", "heuristic"),
            teacher_model=self._teacher_model,
            llm_concurrency=getattr(self.config.proposal, "llm_analyzer_concurrency", 8),
            llm_dedup_threshold=getattr(self.config.proposal, "llm_dedup_threshold", 0.5),
        )
        clusters = analyzer.analyze(trajectories)
        result["phases"]["failure_analysis"] = {
            "patterns_found": len(analyzer.patterns),
            "clusters_found": len(clusters),
            "cluster_categories": [c.suggested_category for c in clusters],
        }
        logger.info("Phase B complete: %d patterns, %d clusters",
                     len(analyzer.patterns), len(clusters))

        if not clusters:
            logger.info("No failure clusters found, skipping proposal phases")
            result["phases"]["skill_proposal"] = {"candidates": 0}
            result["phases"]["library_update"] = {"accepted": 0}
            return self._finalize_epoch_result(result, epoch, epoch_start, trajectories)

        # ============================================================
        # Phase C: Skill proposal (student generates candidates)
        # ============================================================
        logger.info("[Phase C] Skill proposal...")
        proposer = SkillProposer(
            student_model=self._student_model,
            existing_skill_ids=self.library_manager.skill_ids,
            max_candidates=self.config.proposal.max_candidates_per_epoch,
            temperature=self.config.proposal.temperature,
            output_dir=str(epoch_dir / "proposals"),
        )
        candidates = proposer.propose(clusters)
        result["phases"]["skill_proposal"] = {
            "candidates": len(candidates),
            "skill_ids": [c.skill_id for c in candidates],
        }
        logger.info("Phase C complete: %d candidates proposed", len(candidates))

        if not candidates:
            result["phases"]["library_update"] = {"accepted": 0}
            return self._finalize_epoch_result(result, epoch, epoch_start, trajectories)

        # ============================================================
        # Phase D: Skill validation (automated executable checks)
        # Scores are recorded but do NOT gate acceptance.
        # ============================================================
        logger.info("[Phase D] Skill validation (scoring only, no filtering)...")
        validator = SkillValidator(output_dir=str(epoch_dir / "validation"))
        validations = validator.validate(candidates)

        passed_count = sum(1 for v in validations if v.passed)
        result["phases"]["skill_validation"] = {
            "total": len(candidates),
            "passed": passed_count,
            "failed": len(candidates) - passed_count,
        }
        logger.info("Phase D complete: %d/%d passed validation (all proceed regardless)",
                     passed_count, len(candidates))

        # ============================================================
        # Phase E: Skill review (PF helper 5-dim quality scoring)
        # Scores are recorded but do NOT gate acceptance.
        # ============================================================
        logger.info("[Phase E] Skill review (scoring only, no filtering)...")
        reviewer = SkillReviewer(
            teacher_model=self._teacher_model,
            config=self.config.review,
            existing_skill_ids=self.library_manager.skill_ids,
            output_dir=str(epoch_dir / "reviews"),
        )
        reviews = reviewer.review(candidates)
        result["phases"]["skill_review"] = {
            "reviewed": len(reviews),
            "scores": {r.skill_id: r.q_skill for r in reviews},
            "decisions": {r.skill_id: r.decision for r in reviews},
        }
        logger.info("Phase E complete: %s",
                     {r.skill_id: f"{r.q_skill:.2f}/{r.decision}" for r in reviews})

        # ============================================================
        # Phase F: Library update — accept ALL candidates
        # Validation/review scores are stored for analysis but do not
        # block acceptance. This lets us first verify the framework
        # produces quality improvements before adding hard gates.
        # ============================================================
        logger.info("[Phase F] Library update (accept all)...")
        for candidate in candidates:
            self.library_manager._accept_skill(candidate, epoch)
            self.library_manager._history.append({
                "epoch": epoch,
                "skill_id": candidate.skill_id,
                "decision": "accept",
                "q_skill": next((r.q_skill for r in reviews if r.skill_id == candidate.skill_id), None),
                "validation_passed": next((v.passed for v in validations if v.skill_id == candidate.skill_id), None),
            })
        self.library_manager._save_history()

        decisions = {c.skill_id: "accept" for c in candidates}
        result["phases"]["library_update"] = {
            "decisions": decisions,
            "accepted": len(candidates),
            "library_size": len(self.library_manager.skill_ids),
        }
        logger.info("Phase F complete: %d accepted, library size = %d",
                     len(candidates), len(self.library_manager.skill_ids))

        # ============================================================
        # Phase G & H: Pseudo-gradients + Training data
        # ============================================================
        return self._finalize_epoch_result(result, epoch, epoch_start, trajectories,
                                           candidates=candidates, reviews=reviews)

    def _finalize_epoch_result(
        self,
        result: Dict[str, Any],
        epoch: int,
        epoch_start: float,
        trajectories: List[EpisodeTrajectory],
        candidates: Optional[List[CandidateSkill]] = None,
        reviews: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Run pseudo-gradient and training data phases, finalize result."""
        epoch_dir = self.output_dir / f"epoch_{epoch}"

        # Phase G: Pseudo-gradients
        logger.info("[Phase G] Pseudo-gradient computation...")
        pg_computer = PseudoGradientComputer(
            config=self.config.pseudo_gradient,
            output_dir=str(epoch_dir / "gradients"),
        )

        skill_quality = {}
        if candidates and reviews:
            review_map = {r.skill_id: r for r in reviews}
            skill_quality = {c.skill_id: review_map[c.skill_id].q_skill
                            for c in candidates if c.skill_id in review_map}

        student_grad = pg_computer.compute_student_gradient(trajectories, skill_quality)
        teacher_grad = pg_computer.compute_teacher_gradient(trajectories)

        result["phases"]["pseudo_gradient"] = {
            "student_action_corrections": len(student_grad.action_corrections),
            "student_risk_signals": len(student_grad.risk_signals),
            "teacher_selection_signals": len(teacher_grad.selection_signals),
        }

        # Phase H: Training data
        if self.config.save_training_data:
            logger.info("[Phase H] Training data generation...")
            td_outputs = {}
            td_outputs.update(
                self.training_builder.build_action_correction_data(student_grad, epoch)
            )
            td_outputs.update(
                self.training_builder.build_selection_data(teacher_grad, epoch)
            )
            if candidates and reviews:
                td_outputs.update(
                    self.training_builder.build_skillgen_data(candidates, reviews, epoch)
                )
            result["phases"]["training_data"] = {
                "files": {str(k): str(v) for k, v in td_outputs.items()},
            }

        result["elapsed_seconds"] = time.time() - epoch_start
        return result

    # ------------------------------------------------------------------
    # Phase A: Seed execution
    # ------------------------------------------------------------------

    def _phase_a_seed_execution(
        self,
        epoch: int,
        traj_logger: TrajectoryLogger,
    ) -> List[EpisodeTrajectory]:
        """Run ReAct episodes with current skill library on seed data.

        This phase imports from the existing inference system (src/skills_agent/)
        to reuse the ReAct loop, PF pipeline, and tool environment.
        """
        trajectories = []
        seed_data = self.val_manager.get_seed_flat()

        # Import evaluation components from the existing system
        try:
            from src.skills_agent.eval.agent_runner import AgentRunner, RunnerConfig
            from src.skills_agent.eval.tools import ToolEnvironment
            from src.skills_agent.eval.metrics import compute_metrics
            from src.skills_agent.agent.skill_agent_runner import SkillAgentRunner
            from src.skills_agent.agent.config import SkillAgentConfig
            from src.skills_agent.skills.skill import SkillLibrary
        except ImportError as e:
            logger.error("Cannot import inference system: %s", e)
            logger.info("Falling back to simulated seed execution")
            return self._simulated_seed_execution(seed_data, traj_logger, epoch=epoch)

        # Build skill config pointing to the seed directory as primary library path.
        # Generated skills are loaded separately and merged below.
        skill_config = SkillAgentConfig(
            skill_library_path=str(self.library_manager.seed_dir),
            skill_source_format="markdown",
            skills_enabled=True,
            pf_only_mode=True,
            enable_program_functions=True,
            enable_pf_selection=self.config.enable_pf_selection,
            pf_selection_model=self.config.pf_selection_model,
            pf_top_k=self.config.pf_top_k,
            enable_difficulty_gating=self.config.enable_difficulty_gating,
            difficulty_threshold=self.config.difficulty_threshold,
            teacher_api_provider=self._teacher_model.provider if self._teacher_model else None,
            teacher_api_model=self._teacher_model.model_name if self._teacher_model else None,
            teacher_api_key=self._teacher_model.api_key if self._teacher_model else None,
        )

        # Build runner config
        runner_config = RunnerConfig(
            max_steps=self.config.max_steps,
            max_search_calls=self.config.max_search_calls,
            max_read_calls=self.config.max_read_calls,
            timeout_seconds=self.config.timeout_seconds,
            model_type="base",
            serpapi_key=self.config.api_keys.get("serpapi_key") or os.environ.get("SERPAPI_API_KEY", ""),
            openai_key=self.config.api_keys.get("openai_key") or os.environ.get("OPENAI_API_KEY", ""),
            domain=getattr(self.config, "domain", "web_search"),
        )

        # Build tool environment
        env = ToolEnvironment(
            serpapi_key=runner_config.serpapi_key,
            openai_key=runner_config.openai_key,
        )

        # Load dynamic_program_functions.py (e.g. 12 code seed PFs) so the
        # @register_pf side-effects populate the global PF registry. No-op
        # for libraries that don't ship one.
        try:
            from training.common.skill_rollout import _load_dynamic_pfs
            from pathlib import Path as _Path
            _load_dynamic_pfs(_Path(str(self.library_manager.seed_dir)))
        except Exception as _e:
            logger.warning("[pipeline] dynamic PF loader skipped: %s", _e)

        # Load skill library from seed + generated directories.
        # Only the latest version of each base skill_id is kept live —
        # older versions stay on disk for audit but are never selected
        # or fired at runtime (see LibraryManager.skill_ids_latest).
        skill_library = SkillLibrary.load_from_directory(str(self.library_manager.seed_dir))
        if self.library_manager.generated_skill_ids:
            generated_lib = SkillLibrary.load_from_directory(str(self.library_manager.generated_dir))
            for sid, skill in generated_lib._skills.items():
                skill_library._skills[sid] = skill

        latest = set(self.library_manager.skill_ids_latest)
        for sid in list(skill_library._skills.keys()):
            if sid not in latest:
                skill_library._skills.pop(sid, None)

        # Load or initialize the base model.
        # If student is a VLLMStudentWrapper, reuse its engine (no double-load).
        if isinstance(self._student_model, VLLMStudentWrapper):
            # Reuse vLLM student model for ReAct execution
            model = self._student_model.vllm_model
            tokenizer = self._student_model.tokenizer
        elif self.config.base_model_backend:
            # API-based execution
            model = self._student_model
            tokenizer = None
        else:
            # Local model with separate path — attempt vLLM loading
            try:
                from src.skills_agent.eval.model_loader import load_model_vllm
                model, tokenizer = load_model_vllm(
                    self.config.base_model_path,
                    gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                    max_model_len=self.config.vllm_max_model_len,
                    quantization=self.config.vllm_quantization,
                    max_num_seqs=self.config.vllm_max_num_seqs,
                )
            except Exception as e:
                logger.warning("Cannot load local model: %s. Using simulated execution.", e)
                return self._simulated_seed_execution(seed_data, traj_logger, epoch=epoch)

        # ── Two-stage Phase A prefilter ─────────────────────────────────
        # Stage A0 (cheap): raw vLLM rollout, PFs/skill-handlers disabled,
        # used only to identify which pool samples the CURRENT student fails
        # on. We then random-sample `prefilter_cap_k` from those failures
        # and run the expensive Stage A1 only on those. Concentrates compute
        # on the questions skills are actually needed for.
        if (
            getattr(self.config, "prefilter_baseline_failures", False)
            and getattr(self.config, "prefilter_cap_k", 0) > 0
        ):
            prefilter_skill_config = SkillAgentConfig(
                skill_library_path=str(self.library_manager.seed_dir),
                skill_source_format="markdown",
                skills_enabled=False,
                pf_only_mode=False,
                enable_program_functions=False,
                enable_pf_selection=False,
            )
            prefilter_runner = SkillAgentRunner(
                model=model,
                tokenizer=tokenizer,
                config=runner_config,
                env=env,
                skill_library=skill_library,
                skill_config=prefilter_skill_config,
            )
            pf_batch = []
            for i, sample in enumerate(seed_data):
                gold = sample.get("answer", "")
                gold_answers = [gold] if isinstance(gold, str) else gold
                pf_batch.append({
                    "sample_id": sample.get("sample_id", sample.get("id", str(i))),
                    "question": sample["question"],
                    "gold_answers": gold_answers,
                })
            parallel_n_pf = max(1, self.config.vllm_parallel_episodes) if prefilter_runner.use_vllm else 1
            logger.info(
                "[Phase A-prefilter] Running cheap rollout over %d pool samples (no PFs, parallel=%d)",
                len(pf_batch), parallel_n_pf,
            )
            if parallel_n_pf > 1:
                pf_episodes = prefilter_runner.run_batch(
                    pf_batch, mode="clean",
                    parallel_episodes=parallel_n_pf, verbose=False,
                )
            else:
                pf_episodes = [
                    prefilter_runner.run_episode(
                        question=b["question"], gold_answers=b["gold_answers"],
                        sample_id=b["sample_id"], mode="clean",
                    )
                    for b in pf_batch
                ]
            failing_indices: List[int] = []
            for i, (sample, ep) in enumerate(zip(seed_data, pf_episodes)):
                answer = ""
                if ep.final:
                    answer = ep.final.get("answer", "") or ""
                gold = sample.get("answer", "")
                gold_list = [gold] if isinstance(gold, str) else gold
                try:
                    m = compute_metrics(answer, gold_list) or {}
                    em = bool(m.get("exact_match", False))
                except Exception:
                    em = False
                if not em:
                    failing_indices.append(i)
            logger.info(
                "[Phase A-prefilter] %d/%d failed; sampling %d for main pass",
                len(failing_indices), len(seed_data),
                min(self.config.prefilter_cap_k, len(failing_indices)),
            )
            import random as _random
            rng = _random.Random(epoch * 1000 + 42)
            k = self.config.prefilter_cap_k
            if len(failing_indices) > k:
                failing_indices = rng.sample(failing_indices, k)
            seed_data = [seed_data[i] for i in failing_indices]
            # Free prefilter runner reference (model/tokenizer kept alive for main pass)
            del prefilter_runner, pf_episodes

        # Create SkillAgentRunner
        runner = SkillAgentRunner(
            model=model,
            tokenizer=tokenizer,
            config=runner_config,
            env=env,
            skill_library=skill_library,
            skill_config=skill_config,
        )

        # Build batch samples and dispatch in parallel (GPU stays hot across
        # SerpAPI/read latencies instead of idling one episode at a time).
        batch_samples = []
        for i, sample in enumerate(seed_data):
            gold = sample.get("answer", "")
            gold_answers = [gold] if isinstance(gold, str) else gold
            batch_samples.append({
                "sample_id": sample.get("sample_id", sample.get("id", str(i))),
                "question": sample["question"],
                "gold_answers": gold_answers,
            })

        parallel_n = max(1, self.config.vllm_parallel_episodes) if runner.use_vllm else 1
        if parallel_n > 1:
            episodes = runner.run_batch(
                batch_samples,
                mode="clean",
                parallel_episodes=parallel_n,
                verbose=False,
            )
        else:
            episodes = [
                runner.run_episode(
                    question=s["question"],
                    gold_answers=s["gold_answers"],
                    sample_id=s["sample_id"],
                    mode="clean",
                )
                for s in batch_samples
            ]

        for i, (sample, episode) in enumerate(zip(seed_data, episodes)):
            sample_id = batch_samples[i]["sample_id"]
            question = sample["question"]
            gold_answers = batch_samples[i]["gold_answers"]
            dataset_name = sample.get("dataset_name", sample.get("benchmark", ""))

            try:
                # Compute metrics — domain-aware EM/F1
                _domain = getattr(self.config, "domain", "web_search")
                metrics = compute_metrics(episode, gold_answers).to_dict()
                if _domain == "math":
                    from src.skills_agent.eval.metrics import MathAnswerEvaluator
                    answer = episode.get_answer()
                    metrics["exact_match"] = bool(
                        MathAnswerEvaluator.exact_match(answer, gold_answers)
                    )
                    metrics["f1_score"] = float(
                        MathAnswerEvaluator.f1_score(answer, gold_answers)
                    )
                elif _domain == "code":
                    from src.skills_agent.eval.metrics import CodeAnswerEvaluator
                    answer = episode.get_answer()
                    tests = (sample.get("public_tests") or []) + (sample.get("private_tests") or [])
                    func_name = (sample.get("metadata") or {}).get("func_name") if isinstance(sample.get("metadata"), dict) else None
                    metrics["exact_match"] = bool(
                        CodeAnswerEvaluator.exact_match(answer, tests, func_name=func_name)
                    )
                    metrics["f1_score"] = float(
                        CodeAnswerEvaluator.f1_score(answer, tests, func_name=func_name)
                    )

                # Per-episode active skills: recover from episode.pf_records
                # (runner instance state is not per-episode in parallel mode).
                episode_pf_ids = sorted({
                    r["skill_id"] for r in (episode.pf_records or [])
                    if r.get("skill_id")
                })

                # Build trajectory record
                traj = traj_logger.new_episode(
                    sample_id=sample_id,
                    question=question,
                    gold_answers=gold_answers,
                    dataset_name=dataset_name,
                    difficulty_score=0,
                    skills_enabled=skill_config.skills_enabled,
                    selected_pf_ids=episode_pf_ids,
                )

                # Record steps from episode trace. Reconstruct pre-action
                # counters from the trace so the snapshot is never empty
                # (downstream SFT/DPO prompts render '?' otherwise).
                prior_search = 0
                prior_read = 0
                for step_idx, step in enumerate(episode.trace):
                    pf_recs = [r for r in episode.pf_records if r.get("step", -1) == step_idx] \
                        if episode.pf_records else []

                    # episode.trace stores POST-intervention actions. Recover the
                    # original (proposed) action from any activated MODIFY_ACTION
                    # PF record; fall back to the final action when nothing fired.
                    final_type = step.action.type
                    final_arg = step.action.query or step.action.doc_id or ""

                    proposed_type = final_type
                    proposed_arg = final_arg
                    ctx_injections = []
                    for r in pf_recs:
                        if not r.get("activated"):
                            continue
                        itype = r.get("intervention_type")
                        if itype == "modify_action" and r.get("original_action") is not None:
                            proposed_type = r["original_action"]
                            proposed_arg = r.get("original_arg") or ""
                            break  # first activated modifier wins
                    for r in pf_recs:
                        if r.get("activated") and r.get("intervention_type") == "inject_context":
                            txt = r.get("context_text")
                            if txt:
                                ctx_injections.append(txt)

                    step_context = {
                        "step_count": step_idx,
                        "search_count": prior_search,
                        "read_count": prior_read,
                        "has_read": prior_read > 0,
                        "empty_results": bool(
                            step.observation
                            and "No results found" in (step.observation.content or "")
                        ),
                    }

                    traj_logger.record_step(
                        trajectory=traj,
                        step_index=step_idx,
                        proposed_action_type=proposed_type,
                        proposed_action_arg=proposed_arg,
                        proposed_reasoning=step.thought or "",
                        final_action_type=final_type,
                        final_action_arg=final_arg,
                        pf_records=pf_recs,
                        context_injections=ctx_injections,
                        observation_summary=(step.observation.content or "")[:300] if step.observation else "",
                        step_context=step_context,
                    )

                    if final_type == "SEARCH":
                        prior_search += 1
                    elif final_type == "READ":
                        prior_read += 1

                traj_logger.finalize_episode(
                    trajectory=traj,
                    final_answer=episode.get_answer(),
                    exact_match=metrics.get("exact_match", False),
                    f1_score=metrics.get("f1_score", 0.0),
                )
                trajectories.append(traj)

            except Exception as e:
                logger.error("Episode %s failed: %s", sample_id, e)
                continue

            if (i + 1) % 50 == 0:
                logger.info("  Completed %d / %d episodes", i + 1, len(seed_data))

        return trajectories

    def _simulated_seed_execution(
        self,
        seed_data: List[Dict[str, Any]],
        traj_logger: TrajectoryLogger,
        epoch: int = 0,
    ) -> List[EpisodeTrajectory]:
        """Simulated seed execution when the real inference system is unavailable.

        Generates synthetic trajectories with realistic failure patterns for testing
        the downstream pipeline (failure analysis → skill proposal → etc.).
        """
        import random
        rng = random.Random(42 + epoch * 1000)  # Different seed per epoch
        trajectories = []

        for i, sample in enumerate(seed_data):
            sample_id = sample.get("sample_id", sample.get("id", str(i)))
            question = sample["question"]
            gold = sample.get("answer", "")
            gold_answers = [gold] if isinstance(gold, str) else gold

            traj = traj_logger.new_episode(
                sample_id=sample_id,
                question=question,
                gold_answers=gold_answers,
                dataset_name=sample.get("dataset_name", ""),
                difficulty_score=rng.randint(1, 5),
                skills_enabled=True,
                selected_pf_ids=rng.sample(
                    self.library_manager.skill_ids,
                    min(5, len(self.library_manager.skill_ids)),
                ),
            )

            # Simulate 3-8 steps
            n_steps = rng.randint(3, 8)
            for step_idx in range(n_steps):
                if step_idx < n_steps - 1:
                    action_type = rng.choice(["SEARCH", "READ"])
                else:
                    action_type = "FINAL"
                arg = question[:50] if action_type == "SEARCH" else f"doc_{step_idx}"

                traj_logger.record_step(
                    trajectory=traj,
                    step_index=step_idx,
                    proposed_action_type=action_type,
                    proposed_action_arg=arg,
                    proposed_reasoning=f"Step {step_idx} reasoning...",
                    final_action_type=action_type,
                    final_action_arg=arg,
                    pf_records=[],
                    context_injections=[],
                    observation_summary=f"Observation for step {step_idx}",
                    step_context={
                        "step_count": step_idx,
                        "has_read": step_idx > 1,
                        "search_count": min(step_idx, 3),
                        "read_count": max(0, step_idx - 2),
                        "empty_results": rng.random() < 0.2,
                        "max_steps": 25,
                    },
                )

            success = rng.random() < 0.5
            answer = gold_answers[0] if success and gold_answers else "wrong answer"
            traj_logger.finalize_episode(
                trajectory=traj,
                final_answer=answer,
                exact_match=success,
                f1_score=1.0 if success else rng.random() * 0.5,
            )
            trajectories.append(traj)

        logger.info("Simulated %d episodes (%d success)",
                     len(trajectories),
                     sum(1 for t in trajectories if t.exact_match))
        return trajectories

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _save_epoch_summary(self, epoch: int, result: Dict[str, Any]) -> None:
        path = self.output_dir / f"epoch_{epoch}" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))

    def _build_final_summary(self) -> Dict[str, Any]:
        """Build a summary of the full self-improving run."""
        return {
            "experiment_name": self.config.experiment_name,
            "num_epochs": self.config.num_epochs,
            "final_library": self.library_manager.get_summary(),
            "epoch_results": self._epoch_results,
            "training_data": (
                self.training_builder.get_summary()
                if self.training_builder else {}
            ),
        }

    def _save_final_summary(self, summary: Dict[str, Any]) -> None:
        path = self.output_dir / "final_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Final summary saved to %s", path)
        logger.info("Library evolved: %d skills", summary["final_library"]["total_skills"])
