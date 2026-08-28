"""
SkillEvaluator — extends Evaluator with skill-based A/B comparison support.

Supports running multiple ablation configurations:
- baseline: No skills (original AgentRunner)
- skills_top3: Inject top-3 skills
- skills_all: Inject all 10 skills
- skills_no_step: System prompt only, no per-step reminders
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import csv
import time

from .evaluator import Evaluator, EvalConfig
from .agent_runner import AgentRunner, RunnerConfig
from .tools import ToolEnvironment
from .episode import Episode
from .metrics import (
    compute_metrics,
    aggregate_metrics,
    EpisodeMetrics,
    AggregatedMetrics,
)
from .skill_metrics import (
    SkillEpisodeMetrics,
    SkillAggregatedMetrics,
    compute_skill_metrics,
    aggregate_skill_metrics,
    compute_skill_effectiveness_report,
)
from .domain_metrics import compute_domain_metrics, aggregate_domain_metrics
from .skill_episode import SkillEpisode
from ..skills.skill import SkillLibrary
from ..skills.selector import SkillSelector
from ..agent.skill_agent_runner import SkillAgentRunner
from ..agent.config import SkillAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation."""
    name: str
    skills_enabled: bool = True
    max_skills: int = 3
    enable_step_reminders: bool = True
    compact_format: bool = False
    pf_only_mode: bool = False
    enable_prompt_only_skills: Optional[bool] = None  # None = inherit from skill_config


DEFAULT_ABLATIONS = [
    AblationConfig(name="baseline", skills_enabled=False),
    AblationConfig(name="skills_top3", skills_enabled=True, max_skills=3),
    AblationConfig(name="skills_all", skills_enabled=True, max_skills=10),
    AblationConfig(
        name="skills_no_step",
        skills_enabled=True,
        max_skills=3,
        enable_step_reminders=False,
    ),
]


class SkillEvaluator:
    """
    Evaluator with skill-based A/B comparison support.

    Extends the evaluation framework to:
    1. Run multiple ablation configurations
    2. Track skill-specific metrics
    3. Generate comparative reports
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        skill_config: SkillAgentConfig,
        ablations: Optional[List[AblationConfig]] = None,
        preloaded_models: Optional[Dict[str, Tuple[Any, Any]]] = None,
    ):
        self.eval_config = eval_config
        self.skill_config = skill_config
        self.ablations = ablations or DEFAULT_ABLATIONS
        self.preloaded_models = preloaded_models
        self._owns_models = False  # True if we auto-preloaded (need cleanup)

        self.output_dir = Path(eval_config.output_dir) / eval_config.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # {ablation_name: {model_name: {mode: metrics}}}

    def run(
        self,
        test_data: List[Dict[str, Any]],
        ablations: Optional[List[str]] = None,
        verbose: bool = True,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Run evaluation across all ablation configurations.

        Args:
            test_data: List of samples
            ablations: Optional list of ablation names to run (None = all)
            verbose: Print progress
            resume: Skip ablation/model/mode combinations whose episode files already exist

        Returns:
            Dictionary with all results
        """
        start_time = time.time()

        # Auto-preload models if caller didn't provide them
        if self.preloaded_models is None:
            logger.info("No preloaded models provided — auto-preloading...")
            self.preloaded_models = Evaluator.preload_models(self.eval_config)
            self._owns_models = True
            logger.info(f"Preloaded {len(self.preloaded_models)} model(s)")

        # Filter ablations if specified
        to_run = self.ablations
        if ablations:
            to_run = [a for a in self.ablations if a.name in ablations]

        logger.info(f"Running {len(to_run)} ablation(s): {[a.name for a in to_run]}")
        logger.info(f"Models: {list(self.eval_config.models.keys())}")
        logger.info(f"Modes: {self.eval_config.modes}")
        logger.info(f"Samples: {len(test_data)}")

        if self.eval_config.num_samples:
            test_data = test_data[: self.eval_config.num_samples]

        try:
            for ablation in to_run:
                logger.info(f"\n{'='*60}")
                logger.info(f"Ablation: {ablation.name}")
                logger.info(f"{'='*60}")
                self._run_ablation(ablation, test_data, verbose, resume=resume)
        finally:
            if self._owns_models:
                logger.info("Cleaning up auto-preloaded models...")
                Evaluator.cleanup_models(self.preloaded_models)
                self.preloaded_models = None
                self._owns_models = False

        # Generate comparative report
        report = self._generate_comparison_report()

        elapsed = time.time() - start_time
        logger.info(f"\nAll ablations complete in {elapsed:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")

        return report

    # ------------------------------------------------------------------
    # Multi-round retry with API-based judge (parallel)
    # ------------------------------------------------------------------

    _JUDGE_PROMPT = (
        "You are evaluating a question-answering agent's output.\n\n"
        "Question: {question}\n"
        "Agent's answer: {answer}\n\n"
        "Judge as INSUFFICIENT if:\n"
        "- The answer says 'unable to determine' or similar\n"
        "- The answer contains reasoning/meta-text instead of a concrete answer\n"
        "- The answer is clearly a guess without evidence\n"
        "- The answer is clearly incomplete for what the question asks\n\n"
        "Judge as SUFFICIENT if the answer is a concrete entity/value "
        "that directly addresses the question.\n\n"
        "Respond: SUFFICIENT or INSUFFICIENT\n"
        "If INSUFFICIENT, add a second line with a brief hint for what to search."
    )

    def _get_judge_model(self):
        """Lazily load the API judge model (reuses PF helper config)."""
        if hasattr(self, "_judge_model") and self._judge_model is not None:
            return self._judge_model

        from ..agent.config import SkillAgentConfig
        from ..eval.model_loader import load_model_api

        # Use PF helper config for the judge
        provider = self.skill_config.teacher_api_provider
        model_name = self.skill_config.teacher_api_model
        api_key = self.skill_config.teacher_api_key

        # Fallback: resolve from roles.PF helper via eval_config
        if not provider:
            # Try to get from the already-resolved skill_config fields
            provider = getattr(self.skill_config, 'pf_selection_provider', None) or 'openai'
            model_name = model_name or ''
            if not api_key:
                import os
                api_key = os.environ.get("OPENAI_API_KEY")

        if not provider or not api_key:
            self._judge_model = None
            return None

        try:
            model, _ = load_model_api(provider, model_name, api_key)
            self._judge_model = model
            logger.info(f"[retry] Loaded judge model: {provider}/{model_name}")
            return model
        except Exception as e:
            logger.warning(f"[retry] Failed to load judge model: {e}")
            self._judge_model = None
            return None

    def _judge_single(self, judge_model, question: str, answer: str) -> Tuple[bool, str]:
        """Judge a single answer. Returns (is_sufficient, feedback)."""
        if not answer or not answer.strip():
            return False, "No answer was produced. Try different search queries."
        lower = answer.lower().strip()
        if lower.startswith("unable to determine") or lower.startswith("[error]"):
            return False, "Agent failed to find an answer. Try more specific search queries."

        if judge_model is None:
            return True, ""  # No judge → accept all

        try:
            result = judge_model.generate(
                messages=[{"role": "user", "content": self._JUDGE_PROMPT.format(
                    question=question, answer=answer,
                )}],
                max_tokens=150,
                temperature=0.0,
            )
            if result:
                lines = result.strip().split("\n", 1)
                verdict = lines[0].strip().upper()
                feedback = lines[1].strip() if len(lines) > 1 else ""
                if "INSUFFICIENT" in verdict:
                    return False, feedback
                return True, ""
        except Exception as e:
            logger.warning(f"[retry] Judge call failed: {e}")
        return True, ""

    def _judge_batch(self, episodes: List[Episode]) -> List[Tuple[bool, str]]:
        """Judge all episodes in parallel using API model."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        judge_model = self._get_judge_model()
        results = [None] * len(episodes)

        def _judge_one(idx, ep):
            answer = ep.get_answer() if ep.final else ""
            return idx, self._judge_single(judge_model, ep.question, answer)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_judge_one, i, ep) for i, ep in enumerate(episodes)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def _retry_loop(
        self,
        new_episodes: List[Episode],
        remaining_data: List[Dict[str, Any]],
        max_rounds: int,
        runner, mode, model_name, verbose, parallel,
    ) -> List[Episode]:
        """Multi-round retry: judge answers with API model, retry insufficient ones."""
        # Build sample lookup
        sample_lookup = {}
        for s in remaining_data:
            sid = s.get("sample_id", s.get("id", ""))
            sample_lookup[sid] = s

        for retry_round in range(1, max_rounds):
            # Judge all episodes in parallel
            judgments = self._judge_batch(new_episodes)

            # Collect insufficient episodes
            retry_samples = []
            for ep, (is_ok, feedback) in zip(new_episodes, judgments):
                if not is_ok:
                    orig_sample = sample_lookup.get(ep.sample_id, {})
                    retry_s = dict(orig_sample)
                    answer = ep.get_answer() if ep.final else ""
                    retry_s["question"] = (
                        f"{ep.question}\n\n"
                        f"[RETRY round {retry_round}/{max_rounds - 1}] "
                        f"Your previous answer \"{answer}\" was judged insufficient. "
                        f"{feedback} "
                        f"Please try a different search strategy."
                    )
                    retry_s["_original_question"] = ep.question
                    retry_samples.append(retry_s)

            if not retry_samples:
                logger.info(f"  [retry] Round {retry_round}: all {len(new_episodes)} answers accepted")
                break

            logger.info(
                f"  [retry] Round {retry_round}/{max_rounds - 1}: "
                f"retrying {len(retry_samples)}/{len(new_episodes)} episodes"
            )

            retry_episodes = runner.run_batch(
                samples=retry_samples,
                mode=mode,
                model_name=model_name,
                verbose=verbose,
                parallel_episodes=parallel,
                min_batch_size=self.eval_config.min_batch_size,
                poll_timeout=self.eval_config.poll_timeout,
            )

            # Restore original question and merge
            retry_map = {}
            for ep in retry_episodes:
                orig_q = None
                for rs in retry_samples:
                    if rs.get("sample_id", rs.get("id", "")) == ep.sample_id:
                        orig_q = rs.get("_original_question")
                        break
                if orig_q:
                    ep.question = orig_q
                retry_map[ep.sample_id] = ep

            for i, ep in enumerate(new_episodes):
                if ep.sample_id in retry_map:
                    new_episodes[i] = retry_map[ep.sample_id]

        return new_episodes

    def _run_ablation(
        self,
        ablation: AblationConfig,
        test_data: List[Dict[str, Any]],
        verbose: bool,
        resume: bool = False,
    ) -> None:
        """Run a single ablation configuration across all models and modes."""
        ablation_dir = self.output_dir / ablation.name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Create skill config for this ablation (inherit from base skill_config)
        abl_skill_config = SkillAgentConfig(
            skill_library_path=self.skill_config.skill_library_path,
            skill_source_format=self.skill_config.skill_source_format,
            max_skills_in_prompt=ablation.max_skills,
            enable_step_reminders=ablation.enable_step_reminders,
            skills_enabled=ablation.skills_enabled,
            compact_format=ablation.compact_format,
            pf_only_mode=ablation.pf_only_mode or self.skill_config.pf_only_mode,
            enable_prompt_only_skills=(
                ablation.enable_prompt_only_skills
                if ablation.enable_prompt_only_skills is not None
                else self.skill_config.enable_prompt_only_skills
            ),
            enable_program_functions=self.skill_config.enable_program_functions,
            disabled_program_functions=self.skill_config.disabled_program_functions,
            enable_skill_handlers=self.skill_config.enable_skill_handlers,
            teacher_api_provider=self.skill_config.teacher_api_provider,
            teacher_api_model=self.skill_config.teacher_api_model,
            teacher_api_key=self.skill_config.teacher_api_key,
            handler_vote_threshold=self.skill_config.handler_vote_threshold,
            enable_multi_teacher=self.skill_config.enable_multi_teacher,
            teacher_models=self.skill_config.teacher_models,
            deliberation_strategy=self.skill_config.deliberation_strategy,
            # PF selection (PF helper/heuristic selects top-K PFs per question)
            enable_pf_selection=self.skill_config.enable_pf_selection,
            pf_selection_model=self.skill_config.pf_selection_model,
            pf_selection_provider=self.skill_config.pf_selection_provider,
            pf_selection_model_name=self.skill_config.pf_selection_model_name,
            pf_selection_api_key=self.skill_config.pf_selection_api_key,
            pf_top_k=self.skill_config.pf_top_k,
            # Difficulty gating
            enable_difficulty_gating=self.skill_config.enable_difficulty_gating,
            difficulty_model=self.skill_config.difficulty_model,
            difficulty_threshold=self.skill_config.difficulty_threshold,
            # PF helper format postprocessing (runs on ALL questions)
            enable_teacher_format_postprocess=self.skill_config.enable_teacher_format_postprocess,
            format_postprocess_val_dir=self.skill_config.format_postprocess_val_dir,
            format_postprocess_test_dir=self.skill_config.format_postprocess_test_dir,
            format_postprocess_dataset_name=self.skill_config.format_postprocess_dataset_name,
            # Multi-round PF helper retry
            enable_teacher_retry=self.skill_config.enable_teacher_retry,
            max_retry_rounds=self.skill_config.max_retry_rounds,
        )

        ablation_results = {}

        for model_name, model_cfg in self.eval_config.models.items():
            model_results = {}

            # Models are guaranteed preloaded by run()
            model, tokenizer, model_type = self.preloaded_models[model_name]

            # Create base runner config
            runner_config = RunnerConfig(
                max_steps=self.eval_config.max_steps,
                max_search_calls=self.eval_config.max_search_calls,
                max_read_calls=self.eval_config.max_read_calls,
                timeout_seconds=self.eval_config.timeout_seconds,
                model_type=model_type if isinstance(model_type, str) else model_cfg["type"],
                serpapi_key=self.eval_config.serpapi_key,
                openai_key=self.eval_config.openai_key,
                summary_model=self.eval_config.summary_model,
                summary_provider=self.eval_config.summary_provider,
                summary_api_key=self.eval_config.summary_api_key,
                domain=self.eval_config.domain,
                # Forward sampling settings — code domain runs with greedy
                # decode for deterministic pass@1 (matches paper convention).
                temperature=self.eval_config.temperature,
                top_p=self.eval_config.top_p,
                do_sample=self.eval_config.do_sample,
                max_new_tokens=self.eval_config.max_new_tokens,
            )

            # Create tool environment
            env = ToolEnvironment(
                serpapi_key=self.eval_config.serpapi_key,
                openai_key=self.eval_config.openai_key,
                summary_model=self.eval_config.summary_model,
                summary_provider=self.eval_config.summary_provider,
                summary_api_key=self.eval_config.summary_api_key,
            )

            # Create runner
            if ablation.skills_enabled:
                # Auto-detect: directory → Markdown loader, file → JSON loader
                _lib_path = Path(abl_skill_config.skill_library_path)
                _fmt = abl_skill_config.skill_source_format
                if _fmt == "auto":
                    _fmt = "markdown" if _lib_path.is_dir() else "json"
                if _fmt == "markdown":
                    skill_library = SkillLibrary.load_from_directory(
                        abl_skill_config.skill_library_path
                    )
                else:
                    skill_library = SkillLibrary(abl_skill_config.skill_library_path)
                runner = SkillAgentRunner(
                    model=model,
                    tokenizer=tokenizer,
                    config=runner_config,
                    env=env,
                    skill_library=skill_library,
                    skill_config=abl_skill_config,
                )
            else:
                runner = AgentRunner(
                    model=model,
                    tokenizer=tokenizer,
                    config=runner_config,
                    env=env,
                )

            # Run for each mode
            for mode in self.eval_config.modes:
                ep_file = ablation_dir / f"{model_name}_{mode}_episodes.jsonl"

                # ── Resume: load existing episodes, determine remaining samples ──
                existing_ep_map: Dict[str, Episode] = {}  # sample_id → Episode
                if resume and ep_file.exists() and ep_file.stat().st_size > 0:
                    try:
                        with open(ep_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    ep_data = json.loads(line)
                                    ep = Episode.from_dict(ep_data)
                                    existing_ep_map[ep.sample_id] = ep
                    except Exception as e:
                        logger.warning(
                            f"  Failed to load existing episodes from {ep_file} "
                            f"({e}), will re-run all"
                        )
                        existing_ep_map = {}

                # Determine which samples still need running.
                # sample_id fallback mirrors run_batch: sample_id > id > str(index).
                # We inject sample_id into remaining samples so that run_batch
                # produces consistent IDs even when the batch is a subset.
                remaining_data = []
                for i, sample in enumerate(test_data):
                    sid = sample.get("sample_id", sample.get("id", str(i)))
                    if sid not in existing_ep_map:
                        s = dict(sample)
                        s.setdefault("sample_id", sid)
                        remaining_data.append(s)

                if existing_ep_map and not remaining_data:
                    # All samples already completed — skip entirely
                    episodes = []
                    for i, s in enumerate(test_data):
                        sid = s.get("sample_id", s.get("id", str(i)))
                        if sid in existing_ep_map:
                            episodes.append(existing_ep_map[sid])
                    metrics_list = self._compute_episode_metrics(
                        episodes, ablation, runner,
                    )
                    agg = aggregate_skill_metrics(metrics_list)
                    model_results[mode] = {
                        "metrics": agg,
                        "episodes": episodes,
                        "per_episode_metrics": metrics_list,
                    }
                    logger.info(
                        f"  Skipping {ablation.name}/{model_name}/{mode} "
                        f"({len(episodes)} episodes already complete): "
                        f"EM={agg.answer_em:.4f} F1={agg.answer_f1:.4f}"
                    )
                    continue

                if existing_ep_map:
                    logger.info(
                        f"  Resuming {ablation.name}/{model_name}/{mode}: "
                        f"{len(existing_ep_map)} done, "
                        f"{len(remaining_data)} remaining..."
                    )
                else:
                    logger.info(f"  Running {ablation.name}/{model_name}/{mode}...")

                # ── Run remaining samples ────────────────────────────────────
                parallel = self.eval_config.parallel_episodes if self.eval_config.use_vllm else 1
                new_episodes = runner.run_batch(
                    samples=remaining_data,
                    mode=mode,
                    model_name=model_name,
                    verbose=verbose,
                    parallel_episodes=parallel,
                    min_batch_size=self.eval_config.min_batch_size,
                    poll_timeout=self.eval_config.poll_timeout,
                )

                # ── Multi-round retry with API judge (parallel) ──────────
                enable_retry = (
                    hasattr(self.skill_config, "enable_teacher_retry")
                    and self.skill_config.enable_teacher_retry
                )
                max_rounds = getattr(self.skill_config, "max_retry_rounds", 5)

                if enable_retry and max_rounds > 1:
                    new_episodes = self._retry_loop(
                        new_episodes, remaining_data, max_rounds,
                        runner, mode, model_name, verbose, parallel,
                    )

                # Merge: add new episodes to map
                for ep in new_episodes:
                    existing_ep_map[ep.sample_id] = ep

                # Reconstruct full episode list in original sample order
                episodes = []
                for i, sample in enumerate(test_data):
                    sid = sample.get("sample_id", sample.get("id", str(i)))
                    if sid in existing_ep_map:
                        episodes.append(existing_ep_map[sid])

                # ── Compute metrics ──────────────────────────────────────────
                metrics_list = self._compute_episode_metrics(
                    episodes, ablation, runner,
                )
                agg = aggregate_skill_metrics(metrics_list)

                domain = self.eval_config.domain
                model_results[mode] = {
                    "metrics": agg,
                    "episodes": episodes,
                    "per_episode_metrics": metrics_list,
                }

                # ── Save episodes (full set, overwrite) ─────────────────────
                with open(ep_file, "w", encoding="utf-8") as f:
                    for ep in episodes:
                        f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")

                # Save trajectories
                if self.eval_config.save_trajectories:
                    traj_file = ablation_dir / f"{model_name}_{mode}_trajectories.jsonl"
                    with open(traj_file, "w", encoding="utf-8") as f:
                        for ep in episodes:
                            traj_data = {
                                "sample_id": ep.sample_id,
                                "question": ep.question,
                                "steps": ep.get_trajectory(),
                            }
                            if ep.handler_records:
                                traj_data["handler_records"] = ep.handler_records
                            if hasattr(ep, "pf_records") and ep.pf_records:
                                traj_data["pf_records"] = ep.pf_records
                            f.write(json.dumps(traj_data, ensure_ascii=False) + "\n")
                    logger.info(f"    Saved {len(episodes)} trajectories to {traj_file}")

                logger.info(
                    f"    EM={agg.answer_em:.4f} F1={agg.answer_f1:.4f} "
                    f"CEM={agg.answer_cem:.4f} HasRead={agg.has_read_rate:.4f}"
                )

            ablation_results[model_name] = model_results

            # Save skill library snapshot if applicable
            if ablation.skills_enabled and isinstance(runner, SkillAgentRunner):
                snapshot_file = ablation_dir / f"{model_name}_skill_snapshot.json"
                with open(snapshot_file, "w", encoding="utf-8") as f:
                    f.write(runner.skill_library.snapshot())

        self.results[ablation.name] = ablation_results

    def _compute_episode_metrics(
        self,
        episodes: List[Episode],
        ablation: AblationConfig,
        runner,
    ) -> List[SkillEpisodeMetrics]:
        """Compute per-episode metrics for a list of episodes."""
        metrics_list = []
        for ep in episodes:
            gold = ep.gold_answers
            if ablation.skills_enabled and isinstance(runner, SkillAgentRunner):
                m = compute_skill_metrics(
                    episode=ep,
                    gold_answers=gold,
                    active_skill_ids=runner.get_active_skill_ids(),
                )
            else:
                base_m = compute_metrics(ep, gold)
                m = SkillEpisodeMetrics(
                    exact_match=base_m.exact_match,
                    f1_score=base_m.f1_score,
                    cover_exact_match=base_m.cover_exact_match,
                    has_read=base_m.has_read,
                    step_count=base_m.step_count,
                    search_count=base_m.search_count,
                    read_count=base_m.read_count,
                    valid_structure=base_m.valid_structure,
                )
            metrics_list.append(m)
        return metrics_list

    def _load_existing_ablation_metrics(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Scan output_dir for existing ablation episode files not in self.results.

        Returns dict with same structure as self.results but with metrics only
        (no episodes list, to save memory).
        """
        extra = {}
        if not self.output_dir.exists():
            return extra

        for ablation_dir in sorted(self.output_dir.iterdir()):
            if not ablation_dir.is_dir():
                continue
            abl_name = ablation_dir.name
            if abl_name in self.results:
                continue  # Already have fresh results

            # Look for episode files: {model}_{mode}_episodes.jsonl
            for ep_file in sorted(ablation_dir.glob("*_episodes.jsonl")):
                if ep_file.stat().st_size == 0:
                    continue
                # Parse model and mode from filename
                stem = ep_file.stem  # e.g. "base_clean_episodes"
                parts = stem.rsplit("_episodes", 1)[0]  # "base_clean"
                # Find the mode suffix (last segment after last _)
                segments = parts.rsplit("_", 1)
                if len(segments) == 2:
                    model_name, mode = segments
                else:
                    model_name, mode = parts, "clean"

                try:
                    episodes = []
                    with open(ep_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                episodes.append(Episode.from_dict(json.loads(line)))
                    if not episodes:
                        continue

                    # Compute metrics
                    metrics_list = []
                    for ep in episodes:
                        base_m = compute_metrics(ep, ep.gold_answers)
                        m = SkillEpisodeMetrics(
                            exact_match=base_m.exact_match,
                            f1_score=base_m.f1_score,
                            cover_exact_match=base_m.cover_exact_match,
                            has_read=base_m.has_read,
                            step_count=base_m.step_count,
                            search_count=base_m.search_count,
                            read_count=base_m.read_count,
                            valid_structure=base_m.valid_structure,
                        )
                        metrics_list.append(m)
                    agg = aggregate_skill_metrics(metrics_list)

                    extra.setdefault(abl_name, {}).setdefault(model_name, {})[mode] = {
                        "metrics": agg,
                    }
                    logger.info(
                        f"  Loaded existing {abl_name}/{model_name}/{mode}: "
                        f"EM={agg.answer_em:.4f} F1={agg.answer_f1:.4f} "
                        f"({len(episodes)} episodes)"
                    )
                except Exception as e:
                    logger.warning(f"  Failed to load {ep_file}: {e}")

        return extra

    def _generate_comparison_report(self) -> Dict[str, Any]:
        """Generate a comparative report across all ablations (run + existing)."""
        # Merge current results with any existing ablation results on disk
        all_results = dict(self.results)
        existing = self._load_existing_ablation_metrics()
        for abl_name, abl_data in existing.items():
            if abl_name not in all_results:
                all_results[abl_name] = abl_data

        report = {"ablations": {}}

        for abl_name, abl_results in all_results.items():
            report["ablations"][abl_name] = {}
            for model_name, model_results in abl_results.items():
                report["ablations"][abl_name][model_name] = {}
                for mode, data in model_results.items():
                    metrics = data["metrics"]
                    report["ablations"][abl_name][model_name][mode] = metrics.to_dict()

        # Write comparison CSV
        csv_rows = []
        for abl_name, abl_results in all_results.items():
            for model_name, model_results in abl_results.items():
                for mode, data in model_results.items():
                    m = data["metrics"]
                    row = {
                        "ablation": abl_name,
                        "model": model_name,
                        "mode": mode,
                        "answer_em": f"{m.answer_em:.4f}",
                        "answer_f1": f"{m.answer_f1:.4f}",
                        "answer_cem": f"{m.answer_cem:.4f}",
                        "has_read_rate": f"{m.has_read_rate:.4f}",
                        "avg_steps": f"{m.avg_steps:.2f}",
                        "num_samples": m.num_samples,
                    }
                    # Add PF metrics if available
                    if hasattr(m, "avg_pf_activations"):
                        row["avg_pf_activations"] = f"{m.avg_pf_activations:.2f}"
                        row["avg_pf_modify_actions"] = f"{m.avg_pf_modify_actions:.2f}"
                        row["avg_pf_inject_contexts"] = f"{m.avg_pf_inject_contexts:.2f}"
                    csv_rows.append(row)

        csv_file = self.output_dir / "comparison.csv"
        if csv_rows:
            # Preserve extra columns (e.g., answer_mbe from run_llm_judge_eval.py)
            # from any existing comparison.csv — re-running eval should only
            # refresh the core metrics it computes, not wipe downstream columns.
            builtin_keys = set(csv_rows[0].keys())
            existing_extra: Dict[tuple, Dict[str, str]] = {}
            extra_cols: List[str] = []
            if csv_file.exists():
                try:
                    with open(csv_file, "r", encoding="utf-8") as rf:
                        reader = csv.DictReader(rf)
                        for extra_col in reader.fieldnames or []:
                            if extra_col not in builtin_keys and extra_col not in extra_cols:
                                extra_cols.append(extra_col)
                        for r in reader:
                            key = (r.get("ablation"), r.get("model"), r.get("mode"))
                            existing_extra[key] = {
                                k: v for k, v in r.items() if k not in builtin_keys
                            }
                except Exception as e:
                    logger.warning(f"  Could not re-read {csv_file} to preserve extra cols: {e}")

            for row in csv_rows:
                key = (row["ablation"], row["model"], row["mode"])
                for c in extra_cols:
                    row.setdefault(c, existing_extra.get(key, {}).get(c, ""))

            fieldnames = list(csv_rows[0].keys())
            for c in extra_cols:
                if c not in fieldnames:
                    fieldnames.append(c)
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        # Save JSON report
        report_file = self.output_dir / "comparison_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Comparison report saved to {report_file}")
        return report
