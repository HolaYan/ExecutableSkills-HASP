"""Objective A data builder — 'Learn to use PFs/Skills'.

Converts PF-aware trajectories (+ `student_gradient.json`) into SFT,
preference (DPO/IPO), and prompt-only (for on-policy) datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..signals import SignalAggregator
from .prompt_templates import (
    SYSTEM_USE_PFS,
    build_react_step_prompt,
    build_action_target,
    to_chat,
    to_chat_prompt_only,
)
from .signal_filter import FilterConfig, SignalFilter

logger = logging.getLogger(__name__)


@dataclass
class UsePFsBuilderConfig:
    output_dir: str
    enabled_signals: List[str]
    signal_weights: Optional[Dict[str, float]] = None
    threshold: float = 0.3
    formats: List[str] = None        # ["sft", "dpo", "prompt"]
    top_k_per_episode: Optional[int] = None


class UsePFsBuilder:
    """Builds training data for Objective A from trajectories."""

    def __init__(self, config: UsePFsBuilderConfig, aggregator: SignalAggregator):
        self.cfg = config
        self.aggregator = aggregator
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.formats = config.formats or ["sft"]

    # ------------------------------------------------------------------

    def build(
        self,
        trajectories: Iterable,
        teacher_step_scores: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        context = {"teacher_step_scores": teacher_step_scores or {}}
        flt = SignalFilter(self.aggregator, FilterConfig(
            enabled_signals=self.cfg.enabled_signals,
            threshold=self.cfg.threshold,
            top_k_per_episode=self.cfg.top_k_per_episode,
        ))
        kept = flt.filter_steps(trajectories, context=context)
        logger.info("Objective A: kept %d step samples after signal filter", len(kept))

        outputs = {}
        if "sft" in self.formats:
            outputs["sft"] = self._write_sft(kept)
        if "dpo" in self.formats:
            outputs["dpo"] = self._write_dpo(kept)
        if "prompt" in self.formats:
            outputs["prompt"] = self._write_prompt_only(kept)

        self._write_manifest(kept)
        return outputs

    # ------------------------------------------------------------------

    def _write_sft(self, kept: List[Dict[str, Any]]) -> Path:
        path = self.out_dir / "objA_sft.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in kept:
                step = item["step"]
                traj = item["trajectory"]
                sample = {
                    "question": traj.question,
                    "step_index": step.step_index,
                    "step_context": step.step_context_snapshot,
                    "proposed_reasoning": step.proposed_reasoning,
                }
                user = build_react_step_prompt(sample)
                target = build_action_target(step.final_action_type, step.final_action_arg)
                chat = to_chat(SYSTEM_USE_PFS, user, target)
                chat["sample_weight"] = item["aggregate_reward"]
                chat["signal_breakdown"] = item["breakdown"]
                chat["signal_breakdown_4"] = item.get("breakdown_coarse", {})
                chat["sample_id"] = item["sample_id"]
                f.write(json.dumps(chat, ensure_ascii=False) + "\n")
        logger.info("Wrote SFT samples → %s", path)
        return path

    def _write_dpo(self, kept: List[Dict[str, Any]]) -> Path:
        """Preference pairs: chosen = final (PF-corrected), rejected = proposed."""
        path = self.out_dir / "objA_dpo.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in kept:
                step = item["step"]
                if not step.was_modified:
                    continue  # No preference signal when PF didn't modify
                traj = item["trajectory"]
                sample = {
                    "question": traj.question,
                    "step_index": step.step_index,
                    "step_context": step.step_context_snapshot,
                    "proposed_reasoning": step.proposed_reasoning,
                }
                user = build_react_step_prompt(sample)
                chosen = build_action_target(step.final_action_type, step.final_action_arg)
                rejected = build_action_target(step.proposed_action_type, step.proposed_action_arg)
                coarse = item.get("breakdown_coarse", {})
                row = {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_USE_PFS},
                        {"role": "user", "content": user},
                    ],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    # sample_weight scales the DPO loss per pair; high-quality
                    # PF corrections (high correctness+outcome in coarse view)
                    # contribute more gradient.
                    "sample_weight": item["aggregate_reward"],
                    "signal_breakdown_4": coarse,
                    "sample_id": item["sample_id"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote DPO pairs → %s", path)
        return path

    def _write_prompt_only(self, kept: List[Dict[str, Any]]) -> Path:
        """For GRPO / RS rollouts — prompts only (no targets)."""
        path = self.out_dir / "objA_prompts.jsonl"
        seen = set()
        with open(path, "w", encoding="utf-8") as f:
            for item in kept:
                traj = item["trajectory"]
                step = item["step"]
                key = (item["sample_id"], step.step_index)
                if key in seen:
                    continue
                seen.add(key)
                sample = {
                    "question": traj.question,
                    "step_index": step.step_index,
                    "step_context": step.step_context_snapshot,
                    "proposed_reasoning": step.proposed_reasoning,
                }
                user = build_react_step_prompt(sample)
                row = to_chat_prompt_only(SYSTEM_USE_PFS, user)
                row["sample_id"] = item["sample_id"]
                row["step_index"] = step.step_index
                # Metadata columns — required by grpo/reward.py at training time
                row["question"] = traj.question
                row["step_context"] = step.step_context_snapshot or {}
                # Also propagate coarse 4-scalar so downstream RS / GRPO can
                # weigh trajectory quality without re-computing from trajectory.
                row["signal_breakdown_4"] = item.get("breakdown_coarse", {})
                row["sample_weight"] = item["aggregate_reward"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote prompt-only → %s", path)
        return path

    def _write_manifest(self, kept: List[Dict[str, Any]]) -> None:
        manifest = {
            "objective": "A_use_pfs",
            "n_samples": len(kept),
            "enabled_signals": self.cfg.enabled_signals,
            "threshold": self.cfg.threshold,
            "formats": self.formats,
            "signal_weights": self.cfg.signal_weights or {},
        }
        with open(self.out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
