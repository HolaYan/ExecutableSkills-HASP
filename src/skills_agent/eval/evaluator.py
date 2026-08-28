"""
Main Evaluator for ASearcher.

Minimal evaluation flow:
1. Load test data (question + gold_answers)
2. For each model and mode: run Agent Runner, save episode.jsonl
3. Compute metrics and generate summary table
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import csv
import time

from transformers import PreTrainedModel, PreTrainedTokenizer

from .model_loader import load_model_for_eval, load_model_vllm, load_model_api, is_vllm_available, ModelType, VLLMModelWrapper, APIModelWrapper
from .agent_runner import AgentRunner, RunnerConfig, create_adversarial_wrapper, create_pregenerated_adversarial_wrapper
from .tools import ToolEnvironment
from .episode import Episode
from .metrics import (
    compute_metrics,
    aggregate_metrics,
    aggregate_pass_at_k,
    compute_robustness_drop,
    create_comparison_table,
    EpisodeMetrics,
    AggregatedMetrics,
)
from .domain_metrics import compute_domain_metrics, aggregate_domain_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Experiment
    exp_name: str = "eval"
    output_dir: str = "./outputs/eval"

    # Models to evaluate
    models: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "base": {"path": "Qwen/Qwen3-30B-A3B", "type": "base"},
    })

    # Modes to run
    modes: List[str] = field(default_factory=lambda: ["clean"])

    # Budget
    max_steps: int = 10
    max_search_calls: int = 10
    max_read_calls: int = 10
    timeout_seconds: int = 300

    # Adversarial settings
    adv_attack_type: str = "noise"
    adv_strength: float = 0.3

    # Pre-generated adversarial data directory
    adv_data_dir: str = "./data/adv"

    # Data
    num_samples: Optional[int] = None
    domain: str = "web_search"

    # Model loading
    load_in_4bit: bool = True
    use_flash_attention_2: bool = True  # Enable Flash Attention 2 for faster inference

    # vLLM settings for high-performance inference
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.95
    vllm_max_model_len: int = 8192
    vllm_tensor_parallel_size: Optional[int] = None  # None = auto-detect all available GPUs
    vllm_max_num_seqs: int = 512  # Max concurrent sequences (higher = better GPU util)
    vllm_quantization: Optional[str] = "fp8"  # Use FP8 for higher throughput
    parallel_episodes: int = 384  # Number of episodes to process in parallel with vLLM

    # Async batch settings (GPU-tool decoupling)
    async_batch: bool = True  # Kept for backward compat; async always used with vLLM
    min_batch_size: int = 4  # Min episodes before GPU fires (keep small — recv() drain grabs more)
    poll_timeout: float = 0.3  # Max seconds GPU waits for more episodes (short = less idle)

    # Summarization settings (avoids vLLM lock contention in async mode)
    use_gpt_summary: bool = True  # Route SUMMARY to API when using vLLM async
    summary_model: str = ""  # Model for summarization
    summary_provider: str = "openai"  # Provider for summarization ("openai" | "anthropic")
    summary_api_key: Optional[str] = None  # API key for summary provider

    # API model backend (Claude, GPT-5, etc.)
    use_api_model: bool = False
    api_provider: Optional[str] = None  # "anthropic" | "openai"
    api_model_name: Optional[str] = None
    api_max_concurrent: int = 16

    # API keys (for real search and summarization)
    serpapi_key: Optional[str] = None
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None

    # Use local model for summarization instead of GPT
    use_local_model_for_summary: bool = True

    # Trajectory saving
    save_trajectories: bool = True  # Save full message trajectories per episode

    # Pass@k settings
    pass_at_k: int = 1  # Number of seeds (1 = single run, >1 = Pass@k with Max@k)
    base_seed: int = 42  # Base seed for generating sampling seeds

    # Sampling — forwarded into RunnerConfig. Defaults match RunnerConfig
    # (T=0.1, top_p=0.9, sampled). For greedy decode set
    # `do_sample: false` in YAML; the runner will then use T=0.
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    max_new_tokens: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "output_dir": self.output_dir,
            "models": self.models,
            "modes": self.modes,
            "max_steps": self.max_steps,
            "max_search_calls": self.max_search_calls,
            "max_read_calls": self.max_read_calls,
            "timeout_seconds": self.timeout_seconds,
            "adv_attack_type": self.adv_attack_type,
            "adv_strength": self.adv_strength,
            "adv_data_dir": self.adv_data_dir,
            "num_samples": self.num_samples,
            "domain": self.domain,
            "load_in_4bit": self.load_in_4bit,
            "use_flash_attention_2": self.use_flash_attention_2,
            "use_vllm": self.use_vllm,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_max_model_len": self.vllm_max_model_len,
            "vllm_tensor_parallel_size": self.vllm_tensor_parallel_size,
            "vllm_max_num_seqs": self.vllm_max_num_seqs,
            "vllm_quantization": self.vllm_quantization,
            "parallel_episodes": self.parallel_episodes,
            "async_batch": self.async_batch,
            "min_batch_size": self.min_batch_size,
            "poll_timeout": self.poll_timeout,
            "use_api_model": self.use_api_model,
            "api_provider": self.api_provider,
            "api_model_name": self.api_model_name,
            "api_max_concurrent": self.api_max_concurrent,
            "serpapi_key": "***" if self.serpapi_key else None,
            "openai_key": "***" if self.openai_key else None,
            "anthropic_key": "***" if self.anthropic_key else None,
            "use_local_model_for_summary": self.use_local_model_for_summary,
            "use_gpt_summary": self.use_gpt_summary,
            "summary_model": self.summary_model,
            "save_trajectories": self.save_trajectories,
            "pass_at_k": self.pass_at_k,
            "base_seed": self.base_seed,
        }


class Evaluator:
    """
    Main evaluator class.

    Usage:
        config = EvalConfig(
            models={
                "base": {"path": "Qwen/Qwen3-30B-A3B", "type": "base"},
                "sft": {"path": "./ckpt/sft/best", "type": "sft"},
                "rl": {"path": "./ckpt/rl/best", "type": "rl"},
            },
            modes=["clean", "adv"],
        )

        evaluator = Evaluator(config)
        results = evaluator.run(test_data)

        # For multi-dataset evaluation with model reuse:
        preloaded = Evaluator.preload_models(config)
        for dataset in datasets:
            evaluator = Evaluator(config, preloaded_models=preloaded)
            evaluator.run(dataset)
        Evaluator.cleanup_models(preloaded)
    """

    def __init__(self, config: EvalConfig, preloaded_models: Optional[Dict[str, Tuple[Any, Any]]] = None):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Storage for results
        self.all_episodes: Dict[str, Dict[str, List[Episode]]] = {}
        self.all_metrics: Dict[str, Dict[str, AggregatedMetrics]] = {}

        # Preloaded models (to avoid reloading for multi-dataset evaluation)
        self._preloaded_models = preloaded_models

    @staticmethod
    def preload_models(config: "EvalConfig") -> Dict[str, Tuple[Any, Any]]:
        """
        Preload all models specified in config.

        Use this for multi-dataset evaluation to avoid reloading models for each dataset.

        Returns:
            Dict mapping model_name -> (model, tokenizer)
        """
        preloaded = {}
        for model_name, model_cfg in config.models.items():
            model_path = model_cfg["path"]
            model_type = model_cfg["type"]

            logger.info(f"Preloading model: {model_name} from {model_path}")

            if config.use_api_model:
                # Use API model backend (Claude, GPT-5, etc.)
                api_key = None
                if config.api_provider == "anthropic":
                    api_key = config.anthropic_key
                elif config.api_provider == "openai":
                    api_key = config.openai_key
                model, tokenizer = load_model_api(
                    provider=config.api_provider,
                    model_name=config.api_model_name or model_path,
                    api_key=api_key,
                )
                preloaded[model_name] = (model, tokenizer, "api")
                continue

            if config.use_vllm:
                if not is_vllm_available():
                    logger.warning("vLLM not available, falling back to HuggingFace")
                    model, tokenizer = load_model_for_eval(
                        model_name_or_path=model_path,
                        model_type=model_type,
                        load_in_4bit=config.load_in_4bit,
                        use_flash_attention_2=config.use_flash_attention_2,
                    )
                else:
                    logger.info("Using vLLM for high-performance inference")
                    logger.info(f"  Quantization: {config.vllm_quantization or 'none'}")
                    logger.info(f"  Max sequences: {config.vllm_max_num_seqs}")
                    logger.info(f"  Parallel episodes: {config.parallel_episodes}")
                    model, tokenizer = load_model_vllm(
                        model_name_or_path=model_path,
                        model_type=model_type,
                        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                        max_model_len=config.vllm_max_model_len,
                        tensor_parallel_size=config.vllm_tensor_parallel_size,
                        max_num_seqs=config.vllm_max_num_seqs,
                        quantization=config.vllm_quantization,
                    )
            else:
                model, tokenizer = load_model_for_eval(
                    model_name_or_path=model_path,
                    model_type=model_type,
                    load_in_4bit=config.load_in_4bit,
                    use_flash_attention_2=config.use_flash_attention_2,
                )

            preloaded[model_name] = (model, tokenizer, model_type)

        return preloaded

    @staticmethod
    def cleanup_models(preloaded_models: Dict[str, Tuple[Any, Any]]) -> None:
        """Clean up preloaded models to free GPU memory."""
        import gc
        import torch

        for model_name, (model, tokenizer, _) in preloaded_models.items():
            logger.info(f"Cleaning up model: {model_name}")
            del model
            del tokenizer

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def run(
        self,
        test_data: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.

        Args:
            test_data: List of samples with 'question', 'gold_answers', etc.
            verbose: Print progress

        Returns:
            Dictionary with all results
        """
        start_time = time.time()
        logger.info(f"Starting evaluation: {self.config.exp_name}")
        logger.info(f"Models: {list(self.config.models.keys())}")
        logger.info(f"Modes: {self.config.modes}")
        logger.info(f"Samples: {len(test_data)}")

        # Limit samples if configured
        if self.config.num_samples:
            test_data = test_data[:self.config.num_samples]

        # Run evaluation for each model
        for model_name, model_cfg in self.config.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating model: {model_name}")
            logger.info(f"{'='*50}")

            self._evaluate_model(
                model_name=model_name,
                model_path=model_cfg["path"],
                model_type=model_cfg["type"],
                test_data=test_data,
                verbose=verbose,
            )

        # Generate reports
        self._generate_reports()

        elapsed = time.time() - start_time
        logger.info(f"\nEvaluation complete in {elapsed:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")

        return {
            "metrics": {
                model: {mode: m.to_dict() for mode, m in modes.items()}
                for model, modes in self.all_metrics.items()
            },
            "comparison_table": create_comparison_table(self.all_metrics),
        }

    def _evaluate_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        test_data: List[Dict[str, Any]],
        verbose: bool,
    ) -> None:
        """Evaluate a single model on all modes."""

        # Check if model is preloaded
        using_preloaded = False
        if self._preloaded_models and model_name in self._preloaded_models:
            logger.info(f"Using preloaded model: {model_name}")
            model, tokenizer, model_type = self._preloaded_models[model_name]
            using_preloaded = True
        else:
            # Load model (use vLLM if configured and available)
            logger.info(f"Loading model: {model_path} (type: {model_type})")

            if self.config.use_vllm:
                if not is_vllm_available():
                    logger.warning("vLLM not available, falling back to HuggingFace")
                    model, tokenizer = load_model_for_eval(
                        model_name_or_path=model_path,
                        model_type=model_type,
                        load_in_4bit=self.config.load_in_4bit,
                        use_flash_attention_2=self.config.use_flash_attention_2,
                    )
                else:
                    logger.info("Using vLLM for high-performance inference")
                    logger.info(f"  Quantization: {self.config.vllm_quantization or 'none'}")
                    logger.info(f"  Max sequences: {self.config.vllm_max_num_seqs}")
                    logger.info(f"  Parallel episodes: {self.config.parallel_episodes}")
                    model, tokenizer = load_model_vllm(
                        model_name_or_path=model_path,
                        model_type=model_type,
                        gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                        max_model_len=self.config.vllm_max_model_len,
                        tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                        max_num_seqs=self.config.vllm_max_num_seqs,
                        quantization=self.config.vllm_quantization,
                    )
            else:
                model, tokenizer = load_model_for_eval(
                    model_name_or_path=model_path,
                    model_type=model_type,
                    load_in_4bit=self.config.load_in_4bit,
                    use_flash_attention_2=self.config.use_flash_attention_2,
                )

        # Initialize storage
        self.all_episodes[model_name] = {}
        self.all_metrics[model_name] = {}

        # Run each mode
        for mode in self.config.modes:
            logger.info(f"\nRunning mode: {mode}")

            if self.config.pass_at_k > 1:
                self._evaluate_mode_pass_at_k(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_type=model_type,
                    mode=mode,
                    test_data=test_data,
                    verbose=verbose,
                )
            else:
                episodes = self._run_mode(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    model_type=model_type,
                    mode=mode,
                    test_data=test_data,
                    verbose=verbose,
                )

                # Store episodes
                self.all_episodes[model_name][mode] = episodes

                # Compute metrics
                episode_metrics = [compute_metrics(ep) for ep in episodes]
                agg_metrics = aggregate_metrics(episode_metrics)
                self.all_metrics[model_name][mode] = agg_metrics

                # Log metrics
                logger.info(f"  Answer EM: {agg_metrics.answer_em:.4f}")
                logger.info(f"  Answer F1: {agg_metrics.answer_f1:.4f}")
                logger.info(f"  Answer CEM: {agg_metrics.answer_cem:.4f}")
                logger.info(f"  Has Read Rate: {agg_metrics.has_read_rate:.4f}")
                logger.info(f"  Avg Steps: {agg_metrics.avg_steps:.2f}")

                # Save episodes
                self._save_episodes(episodes, mode, model_name)

                # Save trajectories
                if self.config.save_trajectories:
                    self._save_trajectories(episodes, mode, model_name)

        # Clean up model to free memory (only if not preloaded)
        if not using_preloaded:
            del model
            del tokenizer
            import gc
            import torch
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _generate_seeds(base_seed: int, k: int) -> List[int]:
        """Generate k deterministic seeds from a base seed."""
        import random as _random
        rng = _random.Random(base_seed)
        seeds = []
        seen = set()
        while len(seeds) < k:
            s = rng.randint(0, 2**31 - 1)
            if s not in seen:
                seen.add(s)
                seeds.append(s)
        return seeds

    def _evaluate_mode_pass_at_k(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        model_type: str,
        mode: str,
        test_data: List[Dict[str, Any]],
        verbose: bool,
    ) -> None:
        """Run Pass@k evaluation: multiple seeds with Max@k aggregation."""
        k = self.config.pass_at_k
        seeds = self._generate_seeds(self.config.base_seed, k)
        logger.info(f"Pass@{k} evaluation with seeds: {seeds}")

        per_seed_metrics: Dict[int, List[EpisodeMetrics]] = {}
        per_seed_agg: Dict[int, AggregatedMetrics] = {}

        for seed in seeds:
            logger.info(f"\n  Seed {seed}:")
            episodes = self._run_mode(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                model_type=model_type,
                mode=mode,
                test_data=test_data,
                verbose=verbose,
                seed=seed,
            )

            # Save per-seed episodes
            self._save_episodes(episodes, mode, model_name, seed_suffix=f"seed_{seed}")

            # Save per-seed trajectories
            if self.config.save_trajectories:
                self._save_trajectories(episodes, mode, model_name, seed_suffix=f"seed_{seed}")

            # Compute per-seed metrics
            ep_metrics = [compute_metrics(ep) for ep in episodes]
            agg = aggregate_metrics(ep_metrics)
            per_seed_metrics[seed] = ep_metrics
            per_seed_agg[seed] = agg

            logger.info(f"    EM: {agg.answer_em:.4f}, F1: {agg.answer_f1:.4f}, CEM: {agg.answer_cem:.4f}")

        # Compute Max@k aggregation
        max_k_metrics = aggregate_pass_at_k(per_seed_metrics)
        self.all_metrics[model_name][mode] = max_k_metrics

        # Store per-seed metrics for reports
        for seed, agg in per_seed_agg.items():
            self.all_metrics[model_name][f"{mode}_seed{seed}"] = agg

        # Store episodes from last seed as representative
        self.all_episodes[model_name][mode] = episodes

        logger.info(f"\n  Max@{k} Results:")
        logger.info(f"    Answer EM: {max_k_metrics.answer_em:.4f}")
        logger.info(f"    Answer F1: {max_k_metrics.answer_f1:.4f}")
        logger.info(f"    Answer CEM: {max_k_metrics.answer_cem:.4f}")
        logger.info(f"    Has Read Rate: {max_k_metrics.has_read_rate:.4f}")
        logger.info(f"    Avg Steps: {max_k_metrics.avg_steps:.2f}")

    def _run_mode(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        model_type: str,
        mode: str,
        test_data: List[Dict[str, Any]],
        verbose: bool,
        seed: Optional[int] = None,
    ) -> List[Episode]:
        """Run evaluation for a specific mode."""

        # Create runner config
        # Base models get few-shot examples; SFT/RL models don't need them
        runner_config = RunnerConfig(
            max_steps=self.config.max_steps,
            max_search_calls=self.config.max_search_calls,
            max_read_calls=self.config.max_read_calls,
            timeout_seconds=self.config.timeout_seconds,
            model_type=model_type,
            serpapi_key=self.config.serpapi_key,
            openai_key=self.config.openai_key,
            seed=seed,
            summary_model=self.config.summary_model,
            summary_provider=self.config.summary_provider,
            summary_api_key=self.config.summary_api_key,
            # Forward domain so the runner emits the right system prompt and
            # so SkillAgentRunner.step_context["domain"] reflects "code" /
            # "math" / "web_search" — code-domain PFs gate on this.
            domain=getattr(self.config, "domain", "web_search"),
        )

        # Create environment with API keys and adversarial wrapper
        adv_wrapper = None
        pregenerated_modes = {"adv_conflict_l1", "adv_conflict_l2", "adv_conflict_l3", "adv_outdated"}

        if mode in pregenerated_modes:
            # Load pre-generated adversarial data
            distractor_data = self._load_adv_data(mode, test_data)
            if distractor_data:
                attack_type = mode.replace("adv_", "")
                adv_wrapper = create_pregenerated_adversarial_wrapper(
                    distractor_data=distractor_data,
                    attack_type=attack_type,
                )
            else:
                logger.warning(f"No pre-generated adversarial data found for mode {mode}, running without adversarial perturbations")
        elif mode == "adv":
            # Legacy mode: use config's adv_attack_type
            adv_wrapper = create_adversarial_wrapper(
                attack_type=self.config.adv_attack_type,
                strength=self.config.adv_strength,
            )
        elif mode.startswith("adv_"):
            # Built-in adv modes: adv_noise, adv_reorder, adv_irrelevant
            attack_type = mode.replace("adv_", "")
            adv_wrapper = create_adversarial_wrapper(
                attack_type=attack_type,
                strength=self.config.adv_strength,
            )

        # Use local model for summarization if configured
        local_model = model if self.config.use_local_model_for_summary else None
        local_tokenizer = tokenizer if self.config.use_local_model_for_summary else None

        env = ToolEnvironment(
            serpapi_key=self.config.serpapi_key,
            openai_key=self.config.openai_key,
            adv_wrapper=adv_wrapper,
            local_model=local_model,
            local_tokenizer=local_tokenizer,
            summary_model=self.config.summary_model,
            summary_provider=self.config.summary_provider,
            summary_api_key=self.config.summary_api_key,
        )

        # Create runner
        runner = AgentRunner(
            model=model,
            tokenizer=tokenizer,
            config=runner_config,
            env=env,
        )

        # Run episodes (use parallel processing with vLLM)
        parallel = self.config.parallel_episodes if self.config.use_vllm else 1
        episodes = runner.run_batch(
            samples=test_data,
            mode=mode,
            model_name=model_name,
            verbose=verbose,
            parallel_episodes=parallel,
            async_batch=self.config.async_batch,
            min_batch_size=self.config.min_batch_size,
            poll_timeout=self.config.poll_timeout,
        )

        return episodes

    def _load_adv_data(self, mode: str, test_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Load pre-generated adversarial data for conflict/outdated modes.

        Returns:
            Dict mapping sample_id -> distractor info, or None if file not found.
        """
        adv_dir = Path(self.config.adv_data_dir)

        # Determine which file to load based on mode
        # Extract dataset name from exp_name (may contain path like "exp/DatasetName")
        dataset_name = self.config.exp_name.split("/")[-1] if "/" in self.config.exp_name else self.config.exp_name

        suffix = "conflict" if mode.startswith("adv_conflict") else ("outdated" if mode == "adv_outdated" else None)
        if suffix is None:
            return None

        adv_file = adv_dir / f"{dataset_name}_{suffix}.jsonl"

        # Fallback: try dataset name from test_data's "benchmark" field
        if not adv_file.exists() and test_data:
            benchmark = test_data[0].get("benchmark")
            if benchmark:
                adv_file = adv_dir / f"{benchmark}_{suffix}.jsonl"

        if not adv_file.exists():
            logger.warning(f"Adversarial data file not found: {adv_file}")
            return None

        logger.info(f"Loading adversarial data from {adv_file}")
        distractor_data = {}
        with open(adv_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sample_id = entry.get("sample_id")
                if sample_id is not None:
                    distractor_data[str(sample_id)] = entry

        logger.info(f"Loaded adversarial data for {len(distractor_data)} samples")
        return distractor_data

    def _save_episodes(
        self,
        episodes: List[Episode],
        mode: str,
        model_name: str,
        seed_suffix: Optional[str] = None,
    ) -> None:
        """Save episodes to JSONL file."""
        if seed_suffix:
            episodes_dir = self.output_dir / "episodes" / mode / seed_suffix
        else:
            episodes_dir = self.output_dir / "episodes" / mode
        episodes_dir.mkdir(parents=True, exist_ok=True)

        output_path = episodes_dir / f"{model_name}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for episode in episodes:
                f.write(episode.to_json() + "\n")

        logger.info(f"  Saved {len(episodes)} episodes to {output_path}")

    def _save_trajectories(
        self,
        episodes: List[Episode],
        mode: str,
        model_name: str,
        seed_suffix: Optional[str] = None,
    ) -> None:
        """Save full message trajectories to JSONL file."""
        if seed_suffix:
            traj_dir = self.output_dir / "trajectories" / mode / seed_suffix
        else:
            traj_dir = self.output_dir / "trajectories" / mode
        traj_dir.mkdir(parents=True, exist_ok=True)

        output_path = traj_dir / f"{model_name}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for ep in episodes:
                traj_data = {
                    "sample_id": ep.sample_id,
                    "question": ep.question,
                    "steps": ep.get_trajectory(),
                }
                f.write(json.dumps(traj_data, ensure_ascii=False) + "\n")

        logger.info(f"  Saved {len(episodes)} trajectories to {output_path}")

    def _generate_reports(self) -> None:
        """Generate summary reports."""
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        def _metrics_row(model_name, metrics):
            return {
                "model": model_name,
                "answer_em": f"{metrics.answer_em:.4f}",
                "answer_f1": f"{metrics.answer_f1:.4f}",
                "answer_cem": f"{metrics.answer_cem:.4f}",
                "has_read_rate": f"{metrics.has_read_rate:.4f}",
                "avg_steps": f"{metrics.avg_steps:.2f}",
                "avg_search_calls": f"{metrics.avg_search_calls:.2f}",
                "avg_read_calls": f"{metrics.avg_read_calls:.2f}",
                "valid_structure_rate": f"{metrics.valid_structure_rate:.4f}",
                "num_samples": metrics.num_samples,
            }

        # Generate CSV for each mode (including per-seed and max@k)
        all_mode_keys = set()
        for model_name in self.all_metrics:
            all_mode_keys.update(self.all_metrics[model_name].keys())

        for mode_key in sorted(all_mode_keys):
            rows = []
            for model_name in self.all_metrics:
                if mode_key in self.all_metrics[model_name]:
                    rows.append(_metrics_row(model_name, self.all_metrics[model_name][mode_key]))

            if rows:
                # Sanitize mode_key for filename
                safe_name = mode_key.replace("/", "_")
                csv_path = metrics_dir / f"{safe_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info(f"Saved metrics to {csv_path}")

        # Generate comparison table (only uses base modes, not per-seed)
        comparison = create_comparison_table(self.all_metrics)
        if comparison:
            csv_path = metrics_dir / "comparison.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
                writer.writeheader()
                writer.writerows(comparison)
            logger.info(f"Saved comparison table to {csv_path}")

        # Print summary table
        self._print_summary_table()

    def _print_summary_table(self) -> None:
        """Print summary table to console."""
        k = self.config.pass_at_k
        title = f"EVALUATION SUMMARY (Max@{k})" if k > 1 else "EVALUATION SUMMARY"

        logger.info("\n" + "=" * 95)
        logger.info(title)
        logger.info("=" * 95)

        # Header
        header = f"{'Model':<15} | {'Mode':<20} | {'EM':<8} | {'F1':<8} | {'CEM':<8} | {'Read%':<8} | {'Steps':<8}"
        logger.info(header)
        logger.info("-" * 95)

        # Rows
        for model_name in self.all_metrics:
            for mode in sorted(self.all_metrics[model_name].keys()):
                m = self.all_metrics[model_name][mode]
                # Annotate Max@k rows
                display_mode = mode
                if k > 1 and mode in self.config.modes:
                    display_mode = f"{mode} (Max@{k})"
                row = f"{model_name:<15} | {display_mode:<20} | {m.answer_em:.4f} | {m.answer_f1:.4f} | {m.answer_cem:.4f} | {m.has_read_rate:.4f} | {m.avg_steps:.2f}"
                logger.info(row)

        # Robustness drop if applicable
        if "clean" in self.config.modes and "adv" in self.config.modes:
            logger.info("\n" + "-" * 95)
            logger.info("ROBUSTNESS DROP (clean - adv)")
            logger.info("-" * 95)
            for model_name in self.all_metrics:
                if "clean" in self.all_metrics[model_name] and "adv" in self.all_metrics[model_name]:
                    drop = compute_robustness_drop(
                        self.all_metrics[model_name]["clean"],
                        self.all_metrics[model_name]["adv"],
                    )
                    logger.info(f"{model_name:<15} | EM drop: {drop['em_drop']:.4f} | F1 drop: {drop['f1_drop']:.4f} | CEM drop: {drop['cem_drop']:.4f}")

        logger.info("=" * 95)


def run_quick_eval(
    model_path: str,
    test_data: List[Dict[str, Any]],
    model_type: str = "base",
    mode: str = "clean",
    output_dir: str = "./outputs/eval/quick",
    num_samples: Optional[int] = None,
    load_in_4bit: bool = True,
) -> Dict[str, Any]:
    """
    Quick evaluation of a single model.

    Args:
        model_path: Path to model or HuggingFace model name
        test_data: Test samples
        model_type: "base", "sft", or "rl"
        mode: "clean" or "adv"
        output_dir: Output directory
        num_samples: Limit number of samples
        load_in_4bit: Use 4-bit quantization

    Returns:
        Evaluation results
    """
    config = EvalConfig(
        exp_name="quick_eval",
        output_dir=output_dir,
        models={"model": {"path": model_path, "type": model_type}},
        modes=[mode],
        num_samples=num_samples,
        load_in_4bit=load_in_4bit,
    )

    evaluator = Evaluator(config)
    return evaluator.run(test_data)


# Re-export load_test_data from data_loader
from .data_loader import load_test_data
