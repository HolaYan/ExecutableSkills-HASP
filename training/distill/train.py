"""E3/E6 on-policy distillation via skill-aware full-episode rollouts.

PF helper (GPT-4o) and student (local vLLM) both run through
`SkillAgentRunner` so PFs fire live and skill library is actually used.
PF helper trajectories become SFT supervision; disagreement with student
at the same step type is up-weighted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..common.skill_rollout import (
    SkillRolloutRunner,
    flatten_to_per_step,
    load_training_samples,
)
from ..sft.trainer import SFTRunner, TrainerHyperparams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

_ACTION_PAT = re.compile(r"Action:\s*(SEARCH|READ|SUMMARY|FINAL)\s*\(", re.IGNORECASE)


def _action_type(text: str) -> str:
    m = _ACTION_PAT.search(text or "")
    return m.group(1).upper() if m else ""


def _run_rollouts(
    samples: List[Dict[str, Any]],
    model_path: str,
    skill_library_dir: str,
    backend: str,
    api_provider: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    tp_size: int = 2,
    parallel_episodes: int = 16,
    max_steps: int = 10,
    max_search_calls: int = 8,
    max_read_calls: int = 8,
    domain: str = "web_search",
    max_model_len: int = 8192,
) -> List[Dict[str, Any]]:
    runner = SkillRolloutRunner(
        model_path=model_path,
        skill_library_dir=skill_library_dir,
        backend=backend,
        api_provider=api_provider,
        api_model=api_model,
        api_key=api_key,
        tensor_parallel_size=tp_size,
        parallel_episodes=parallel_episodes,
        max_steps=max_steps,
        max_search_calls=max_search_calls,
        max_read_calls=max_read_calls,
        max_model_len=max_model_len,
        domain=domain,
    )
    runner.setup()
    episodes = runner.run(samples, mode="clean")
    return flatten_to_per_step(episodes, samples, domain=domain)


def build_distill_sft(cfg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = cfg["data"]["prompts_path"]
    raw_data_dir = cfg.get("data", {}).get("raw_data_dir", "data/web_search")

    samples = load_training_samples(prompts_path, raw_data_dir=raw_data_dir)
    logger.info("Distill: %d unique training samples", len(samples))

    # Default path: closed-loop experiments have {exp_id}/library (with evolved
    # skills). Flat experiments (E3) don't — fall back to the seed library.
    _default_lib = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["id"] / "library"
    if not _default_lib.is_dir():
        # Flat experiment: use domain-appropriate seed library
        domain = cfg.get("domain", "web_search")
        _default_lib = Path("skills") / domain if domain in ("math", "code", "web") else Path("skills")
    skill_library_dir = cfg.get("rollout", {}).get("skill_library_dir", str(_default_lib))
    parallel_episodes = cfg.get("rollout", {}).get("parallel_episodes", 16)
    max_steps = cfg.get("rollout", {}).get("max_steps", 10)
    max_search_calls = cfg.get("rollout", {}).get("max_search_calls", 8)
    max_read_calls = cfg.get("rollout", {}).get("max_read_calls", 8)
    tp_size = cfg.get("rollout", {}).get("tensor_parallel_size", 2)
    max_model_len = cfg.get("rollout", {}).get("max_model_len", 8192)

    # ------ PF helper rollout (API, skill-aware) ------
    teacher_cfg = cfg["teacher"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for the reference rollout")
    logger.info("PF helper rollouts via API: model=%s", teacher_cfg.get("model", ""))
    teacher_rows = _run_rollouts(
        samples,
        model_path=teacher_cfg.get("model", ""),     # unused by api backend, kept for logging
        skill_library_dir=skill_library_dir,
        backend="api",
        api_provider="openai",
        api_model=teacher_cfg.get("model", ""),
        api_key=api_key,
        tp_size=tp_size,
        parallel_episodes=teacher_cfg.get("concurrency", parallel_episodes),
        max_steps=max_steps,
        max_search_calls=max_search_calls,
        max_read_calls=max_read_calls,
        domain=cfg.get("domain", "web_search"),
        max_model_len=max_model_len,
    )
    logger.info("PF helper emitted %d per-step rows", len(teacher_rows))

    # ------ Student rollout (vLLM, skill-aware) for on-policy weighting ------
    student_action_at: Dict[tuple, str] = {}
    if cfg.get("on_policy", False):
        logger.info("Student rollouts via vLLM: %s", cfg["model"]["path"])
        student_rows = _run_rollouts(
            samples,
            model_path=cfg["model"]["path"],
            skill_library_dir=skill_library_dir,
            backend="vllm",
            tp_size=tp_size,
            parallel_episodes=parallel_episodes,
            max_steps=max_steps,
            max_search_calls=max_search_calls,
            max_read_calls=max_read_calls,
            domain=cfg.get("domain", "web_search"),
            max_model_len=max_model_len,
        )
        for r in student_rows:
            student_action_at[(r["sample_id"], r["step_index"])] = _action_type(r["generation"])

    # ------ Build SFT rows from PF helper trajectory ------
    sft_path = out_dir / "distill_sft.jsonl"
    n_kept = 0
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in teacher_rows:
            t_type = _action_type(r["generation"])
            if not t_type:
                continue  # malformed PF helper output
            s_type = student_action_at.get((r["sample_id"], r["step_index"]), "")
            weight = 2.0 if (s_type and s_type != t_type) else 1.0
            msgs = r.get("messages", [])
            if not msgs:
                continue
            assistant = {"role": "assistant", "content": r.get("generation", "")}
            row = {
                "messages": list(msgs) + [assistant],
                "sample_weight": weight,
                "sample_id": r["sample_id"],
                "step_index": r["step_index"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_kept += 1
    logger.info("Distill SFT: %d rows → %s", n_kept, sft_path)
    return sft_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    exp = cfg["experiment"]["id"]
    base = Path(cfg["experiment"]["output_dir"]) / exp
    data_dir = base / "data"
    ckpt_dir = base / "ckpt"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sft_path = build_distill_sft(cfg, data_dir)

    t = cfg["trainer"]
    hp = TrainerHyperparams(
        model_path=cfg["model"]["path"],
        output_dir=str(ckpt_dir),
        data_path=str(sft_path),
        num_train_epochs=t.get("num_train_epochs", 2.0),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
        learning_rate=t.get("learning_rate", 5e-6),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_seq_length=t.get("max_seq_length", 4096),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        save_every_n_epochs=t.get("save_every_n_epochs"),
        deepspeed=t.get("deepspeed"),
        use_lora=t.get("use_lora", True),
        lora_r=t.get("lora_r", 16),
        lora_alpha=t.get("lora_alpha", 32),
        report_to=t.get("report_to", "wandb"),
        run_name=exp,
        seed=cfg["experiment"].get("seed", 42),
    )
    SFTRunner(hp).run()


if __name__ == "__main__":
    main()
