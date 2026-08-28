"""Thin wrappers around TRL `SFTTrainer` / `DPOTrainer`.

We deliberately keep training logic in this file so that every experiment
YAML under `configs/training/` flows through the same entry point.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to the highest-numbered `checkpoint-N` subdir, or None.

    Used for auto-resume: if a prior SLURM run was killed, the
    HuggingFace Trainer can pick up from the last saved checkpoint.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    cands = []
    for p in root.glob("checkpoint-*"):
        name = p.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except ValueError:
            continue
        cands.append((step, p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return str(cands[0][1])


@dataclass
class TrainerHyperparams:
    model_path: str
    output_dir: str
    data_path: str
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.03
    max_seq_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    save_every_n_epochs: Optional[int] = None  # if set, save every N epochs (overrides save_steps)
    deepspeed: Optional[str] = None
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    report_to: str = "wandb"
    run_name: str = ""
    seed: int = 42
    sample_weight_field: Optional[str] = None  # name of per-sample weight, if any
    # `evolving/`: an HF TrainerCallback built by evolving.callback.build_callback.
    # Left as Any so importing the trainer never pulls in transformers' callback
    # machinery, and so this stays None on every path that does not use it.
    evolve_callback: Any = None


class _BaseRunner:
    def __init__(self, hp: TrainerHyperparams):
        self.hp = hp
        Path(hp.output_dir).mkdir(parents=True, exist_ok=True)

    def _build_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained(self.hp.model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dtype = torch.bfloat16 if self.hp.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.hp.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.config.use_cache = False

        if self.hp.use_lora:
            from peft import LoraConfig, get_peft_model
            lc = LoraConfig(
                r=self.hp.lora_r,
                lora_alpha=self.hp.lora_alpha,
                lora_dropout=self.hp.lora_dropout,
                target_modules=self.hp.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lc)
            model.print_trainable_parameters()
            if self.hp.gradient_checkpointing:
                model.enable_input_require_grads()

        return model, tok

    def _load_dataset(self) -> "Any":
        from datasets import load_dataset
        return load_dataset("json", data_files=self.hp.data_path, split="train")

    def _resolve_save_cadence(self, ds_len: int) -> Dict[str, Any]:
        if not self.hp.save_every_n_epochs:
            return {"save_strategy": "steps", "save_steps": self.hp.save_steps}
        effective_batch = max(
            1, self.hp.per_device_train_batch_size * self.hp.gradient_accumulation_steps
        )
        steps_per_epoch = max(1, ds_len // effective_batch)
        save_steps_override = max(1, self.hp.save_every_n_epochs * steps_per_epoch)
        logger.info(
            "save_every_n_epochs=%d → save_steps=%d (steps_per_epoch=%d, ds_len=%d, eff_batch=%d)",
            self.hp.save_every_n_epochs, save_steps_override, steps_per_epoch, ds_len, effective_batch,
        )
        return {"save_strategy": "steps", "save_steps": save_steps_override}


class SFTRunner(_BaseRunner):
    def run(self) -> str:
        from trl import SFTTrainer, SFTConfig

        model, tok = self._build_model_and_tokenizer()
        ds = self._load_dataset()
        save_kwargs = self._resolve_save_cadence(len(ds))

        cfg = SFTConfig(
            output_dir=self.hp.output_dir,
            num_train_epochs=self.hp.num_train_epochs,
            per_device_train_batch_size=self.hp.per_device_train_batch_size,
            gradient_accumulation_steps=self.hp.gradient_accumulation_steps,
            learning_rate=self.hp.learning_rate,
            warmup_ratio=self.hp.warmup_ratio,
            max_length=self.hp.max_seq_length,
            bf16=self.hp.bf16,
            gradient_checkpointing=self.hp.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=self.hp.logging_steps,
            save_total_limit=self.hp.save_total_limit,
            deepspeed=self.hp.deepspeed,
            report_to=self.hp.report_to,
            run_name=self.hp.run_name or os.path.basename(self.hp.output_dir),
            seed=self.hp.seed,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            **save_kwargs,
        )
        cfg._n_gpu = 1  # pin to cuda:0; avoids HF Trainer's DataParallel OOM when N GPUs are visible

        trainer = SFTTrainer(
            model=model,
            processing_class=tok,
            train_dataset=ds,
            args=cfg,
            # `evolving/`: every N steps, pause → eval the live weights →
            # distill PFs from the failures → grow the run-scoped library.
            # None when `evolve.enabled` is false, which is the default.
            callbacks=[cb] if (cb := getattr(self.hp, "evolve_callback", None)) else None,
        )
        # Auto-resume: if output_dir already holds `checkpoint-N` subdirs
        # (from a prior killed run), HuggingFace Trainer will pick up the
        # latest one and continue from there.
        _last_ckpt = _latest_checkpoint(self.hp.output_dir)
        if _last_ckpt:
            logger.info("Auto-resume: continuing SFT from %s", _last_ckpt)
            trainer.train(resume_from_checkpoint=_last_ckpt)
        else:
            trainer.train()
        trainer.save_model(self.hp.output_dir)
        tok.save_pretrained(self.hp.output_dir)
        logger.info("SFT done → %s", self.hp.output_dir)
        return self.hp.output_dir


class DPORunner(_BaseRunner):
    def __init__(self, hp: TrainerHyperparams, beta: float = 0.1):
        super().__init__(hp)
        self.beta = beta

    def run(self) -> str:
        from trl import DPOTrainer, DPOConfig

        model, tok = self._build_model_and_tokenizer()
        ds = self._load_dataset()
        save_kwargs = self._resolve_save_cadence(len(ds))

        cfg = DPOConfig(
            output_dir=self.hp.output_dir,
            num_train_epochs=self.hp.num_train_epochs,
            per_device_train_batch_size=self.hp.per_device_train_batch_size,
            gradient_accumulation_steps=self.hp.gradient_accumulation_steps,
            learning_rate=self.hp.learning_rate,
            warmup_ratio=self.hp.warmup_ratio,
            max_length=self.hp.max_seq_length,
            bf16=self.hp.bf16,
            gradient_checkpointing=self.hp.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=self.hp.logging_steps,
            save_total_limit=self.hp.save_total_limit,
            deepspeed=self.hp.deepspeed,
            report_to=self.hp.report_to,
            run_name=self.hp.run_name or os.path.basename(self.hp.output_dir),
            seed=self.hp.seed,
            beta=self.beta,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            **save_kwargs,
        )
        cfg._n_gpu = 1

        trainer = DPOTrainer(
            model=model,
            tokenizer=tok,
            train_dataset=ds,
            args=cfg,
        )
        _last_ckpt = _latest_checkpoint(self.hp.output_dir)
        if _last_ckpt:
            logger.info("Auto-resume: continuing DPO from %s", _last_ckpt)
            trainer.train(resume_from_checkpoint=_last_ckpt)
        else:
            trainer.train()
        trainer.save_model(self.hp.output_dir)
        tok.save_pretrained(self.hp.output_dir)
        logger.info("DPO done → %s", self.hp.output_dir)
        return self.hp.output_dir
