"""Skill-aware full-episode rollouts for rejection sampling (E2 / E5).

Replaces the previous single-step vLLM-only rollout: each sample now goes
through `SkillAgentRunner` (the production inference framework), producing
a complete ReAct trajectory where PFs fire live and skills actually get
used. Output rows are per-step so downstream SFT/filter tooling still
consumes the same schema.

Rejection-sampling diversity comes from running `group_size` full
episodes per input sample (temperature > 0 yields varied trajectories);
filtering operates per episode based on final-answer correctness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.skill_rollout import (
    SkillRolloutRunner,
    flatten_to_per_step,
    load_training_samples,
)

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    model_path: str
    prompts_path: str
    output_dir: str
    skill_library_dir: str                  # REQUIRED: seed (+ generated, merged upstream)
    raw_data_dir: str = "data/web_search"
    group_size: int = 4                     # N episodes per sample (diversity via temperature)
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 128
    parallel_episodes: int = 32             # concurrent episodes in SkillAgentRunner (higher → more GPU-hot while some episodes wait on API)
    max_steps: int = 10
    max_search_calls: int = 8
    max_read_calls: int = 8
    timeout_seconds: int = 300
    pf_top_k: int = 10
    enable_pf_selection: bool = True
    pf_selection_model: str = ""
    mode: str = "clean"                     # "clean" | "adv"
    domain: str = "web_search"              # "web_search" | "math"


class Rollouter:
    def __init__(self, cfg: RolloutConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._runner: Optional[SkillRolloutRunner] = None

    def _build_runner(self) -> SkillRolloutRunner:
        if self._runner is None:
            self._runner = SkillRolloutRunner(
                model_path=self.cfg.model_path,
                skill_library_dir=self.cfg.skill_library_dir,
                backend="vllm",
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
                max_model_len=self.cfg.max_model_len,
                max_num_seqs=self.cfg.max_num_seqs,
                parallel_episodes=self.cfg.parallel_episodes,
                max_steps=self.cfg.max_steps,
                max_search_calls=self.cfg.max_search_calls,
                max_read_calls=self.cfg.max_read_calls,
                timeout_seconds=self.cfg.timeout_seconds,
                pf_top_k=self.cfg.pf_top_k,
                enable_pf_selection=self.cfg.enable_pf_selection,
                pf_selection_model=self.cfg.pf_selection_model,
                domain=self.cfg.domain,
            )
            self._runner.setup()
        return self._runner

    def run(self) -> Path:
        samples = load_training_samples(
            self.cfg.prompts_path, raw_data_dir=self.cfg.raw_data_dir,
        )
        logger.info(
            "RS rollout: %d unique samples × group_size=%d = %d full episodes (skill-aware)",
            len(samples), self.cfg.group_size, len(samples) * self.cfg.group_size,
        )
        runner = self._build_runner()

        out_path = self.out_dir / f"rollouts_{self.cfg.mode}.jsonl"
        n_rows = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for g in range(self.cfg.group_size):
                logger.info("  group %d/%d", g + 1, self.cfg.group_size)
                episodes = runner.run(samples, mode=self.cfg.mode)
                rows = flatten_to_per_step(episodes, samples, domain=self.cfg.domain)
                for r in rows:
                    r["group_index"] = g
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_rows += 1
        logger.info("Saved %d per-step rollout rows → %s", n_rows, out_path)
        return out_path
