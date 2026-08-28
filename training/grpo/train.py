"""GRPO training entrypoint — wraps `trl.GRPOTrainer`.

Reward = combination of S1..S4 signals (full-mix) or a single-signal
subset for ablation. Configure it in the training config's `signals:` block.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from ..data.signal_filter import resolve_enabled_signals
from ..rejection_sampling.verifier import TeacherVerifier
from .reward import build_reward_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def _load_prompts_dataset(prompts_path: str):
    """Load prompt-only jsonl into a HF dataset, keeping metadata."""
    from datasets import Dataset
    import json
    rows = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # GRPOTrainer expects "prompt" column as str (chat template applied)
    # but we also keep metadata for reward fn
    ds = Dataset.from_list(rows)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from trl import GRPOTrainer, GRPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    exp_id = cfg["experiment"]["id"]
    base_out = Path(cfg["experiment"]["output_dir"]) / exp_id
    ckpt_out = base_out / "ckpt"
    ckpt_out.mkdir(parents=True, exist_ok=True)

    model_path = cfg["model"]["path"]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = False

    ds = _load_prompts_dataset(cfg["data"]["prompts_path"])

    # Build reward fn
    enabled = resolve_enabled_signals(cfg["signals"]["enabled"])
    weights = cfg["signals"].get("weights") or {}
    mode = cfg["reward"].get("mode", "action")
    teacher = None
    if mode == "skill":
        teacher = TeacherVerifier(model_name=cfg["reward"].get("teacher_model", ""))
    reward_fn = build_reward_fn(enabled_signals=enabled, weights=weights, mode=mode, teacher_verifier=teacher)

    t = cfg["trainer"]
    gcfg = GRPOConfig(
        output_dir=str(ckpt_out),
        num_train_epochs=t.get("num_train_epochs", 1.0),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
        learning_rate=t.get("learning_rate", 5e-6),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_prompt_length=t.get("max_prompt_length", 2048),
        max_completion_length=t.get("max_completion_length", 1024),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        logging_steps=t.get("logging_steps", 5),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 2),
        deepspeed=t.get("deepspeed"),
        report_to=t.get("report_to", "wandb"),
        run_name=exp_id,
        seed=cfg["experiment"].get("seed", 42),
        beta=t.get("kl_beta", 0.04),
        num_generations=t.get("num_generations", 8),
        temperature=t.get("temperature", 0.9),
        top_p=t.get("top_p", 0.95),
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        reward_funcs=[reward_fn],
        args=gcfg,
    )
    trainer.train()
    trainer.save_model(str(ckpt_out))
    tok.save_pretrained(str(ckpt_out))
    logger.info("GRPO done → %s", ckpt_out)


if __name__ == "__main__":
    main()
