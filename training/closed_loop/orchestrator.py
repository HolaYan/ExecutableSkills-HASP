"""V2 closed-loop orchestrator (E4 SFT, E5 RS, E6 Distill).

Alternates between training phases and library-evolve phases:

  for it in range(max_evolve_iterations):
      (a) Train for `evolve_every_epochs` epochs (resume from last ckpt)
      (b) Run one self-improving epoch against the latest ckpt → skills merged
      (c) Refresh Obj-A/Obj-B training data from new trajectories
  # final train pass (optional) after last evolve

Each iteration writes to:
  {output_dir}/{exp_id}/iter_{it}/
    data/   — refreshed SFT/prompts for this iter
    ckpt/   — model checkpoint at end of this iter's train phase
    evolve/ — self-improving outputs (trajectories, proposals, reviews)
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

from ..sft.trainer import SFTRunner, TrainerHyperparams
from .data_refresh import refresh_objA
from .evolve_step import run_evolve_step


def _merge_lora_if_needed(ckpt_dir: Path, base_model_path: str) -> str:
    """If `ckpt_dir` contains a LoRA adapter (no `model.safetensors`), merge
    into the base model and write the full weights to `{ckpt_dir}/merged/`.
    Returns the path to a directory loadable via AutoModelForCausalLM.

    Merge runs on CPU to avoid competing with the HF trainer / vLLM for GPU.
    """
    has_full_weights = any(ckpt_dir.glob("model*.safetensors")) or \
                       any(ckpt_dir.glob("pytorch_model*.bin"))
    if has_full_weights:
        return str(ckpt_dir)

    has_adapter = (ckpt_dir / "adapter_config.json").exists()
    if not has_adapter:
        logger.warning("%s has neither full weights nor LoRA adapter — using as-is", ckpt_dir)
        return str(ckpt_dir)

    merged = ckpt_dir / "merged"
    if merged.exists() and any(merged.glob("model*.safetensors")):
        return str(merged)

    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Merging LoRA adapter %s into base %s (CPU)", ckpt_dir, base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, str(ckpt_dir))
    merged_model = peft_model.merge_and_unload()
    merged.mkdir(exist_ok=True)
    merged_model.save_pretrained(str(merged), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    tok.save_pretrained(str(merged))
    del base, peft_model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(merged)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


# ----------------------------------------------------------------------
# Method-specific train step dispatchers
# ----------------------------------------------------------------------

def _train_sft(cfg: dict, data_path: Path, ckpt_dir: Path, resume_from: str = None, run_suffix: str = "") -> str:
    t = cfg["trainer"]
    hp = TrainerHyperparams(
        model_path=resume_from or cfg["model"]["path"],
        output_dir=str(ckpt_dir),
        data_path=str(data_path),
        num_train_epochs=cfg["closed_loop"]["evolve_every_epochs"],
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
        learning_rate=t.get("learning_rate", 1e-5),
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
        run_name=f"{cfg['experiment']['id']}_{run_suffix}",
        seed=cfg["experiment"].get("seed", 42),
    )
    SFTRunner(hp).run()
    return str(ckpt_dir)


def _train_rs(cfg: dict, data_path: Path, ckpt_dir: Path, resume_from: str = None, run_suffix: str = "") -> str:
    """One-iteration RS loop (skill-aware rollout → episode filter → SFT).

    Keeps `experiment.id` / `experiment.output_dir` untouched so that
    `run_rs_iteration` writes to `{output_dir}/{exp_id}/iter_{it}/{ckpt,merged}/`
    — matching the path layout the orchestrator + eval scripts expect.
    """
    from ..rejection_sampling.train import run_rs_iteration

    # ckpt_dir is `{output_dir}/{exp_id}/iter_{it}/ckpt` → recover it from the dir name.
    iter_idx = int(ckpt_dir.parent.name.split("_")[-1])

    rs_cfg = copy.deepcopy(cfg)
    # RS's prompts file is the refreshed prompts from this iter
    rs_cfg["data"]["prompts_path"] = str(data_path.parent / "objA_prompts.jsonl")
    rs_cfg.setdefault("rollout", {})
    # Point rollouts at this experiment's evolved skill library (seed + generated)
    rs_cfg["rollout"]["skill_library_dir"] = str(
        Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["id"] / "library"
    )
    # Override num_train_epochs per-iter to evolve_every_epochs. Without this,
    # closed-loop iters use trainer.num_train_epochs (8) instead of the
    # closed-loop cadence (3) → each iter is 8-epoch SFT (~30+ min) and the
    # watchdog kills the job for sustained low GPU util.
    cl = cfg.get("closed_loop", {})
    if "evolve_every_epochs" in cl:
        rs_cfg.setdefault("trainer", {})
        rs_cfg["trainer"]["num_train_epochs"] = float(cl["evolve_every_epochs"])
    return run_rs_iteration(rs_cfg, iter_idx=iter_idx, current_model_path=resume_from or cfg["model"]["path"])


def _train_distill(cfg: dict, data_path: Path, ckpt_dir: Path, resume_from: str = None, run_suffix: str = "") -> str:
    from ..distill.train import build_distill_sft

    distill_cfg = copy.deepcopy(cfg)
    distill_cfg["data"]["prompts_path"] = str(data_path.parent / "objA_prompts.jsonl")
    distill_cfg["model"]["path"] = resume_from or cfg["model"]["path"]
    distill_cfg["experiment"]["id"] = f"{cfg['experiment']['id']}_{run_suffix}"
    distill_cfg.setdefault("rollout", {})
    distill_cfg["rollout"]["skill_library_dir"] = str(
        Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["id"] / "library"
    )

    sft_path = build_distill_sft(distill_cfg, ckpt_dir.parent / "data")
    return _train_sft(cfg, sft_path, ckpt_dir, resume_from=resume_from, run_suffix=run_suffix)


_DISPATCH = {
    "sft": _train_sft,
    "rs": _train_rs,
    "distill": _train_distill,
}


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def _setup_library_dir(cfg: dict, base: Path) -> Path:
    """Copy seed skills into an experiment-scoped library dir so evolve step
    can mutate it without touching the canonical `self_improving/skills/`."""
    lib_dir = base / "library"
    lib_dir.mkdir(parents=True, exist_ok=True)
    seed_src = Path(cfg["closed_loop"].get("seed_library_dir", "skills"))
    if not any(lib_dir.iterdir()):
        if seed_src.exists():
            shutil.copytree(seed_src, lib_dir / "seed", dirs_exist_ok=True)
        (lib_dir / "generated").mkdir(exist_ok=True)
        logger.info("Initialised iter library at %s (from %s)", lib_dir, seed_src)
    return lib_dir


def run(cfg: dict) -> None:
    exp_id = cfg["experiment"]["id"]
    method = cfg["method"]                                  # sft | rs | distill
    base = Path(cfg["experiment"]["output_dir"]) / exp_id
    base.mkdir(parents=True, exist_ok=True)

    train_fn = _DISPATCH[method]
    cl = cfg["closed_loop"]
    max_iters = cl["max_evolve_iterations"]
    lib_dir = _setup_library_dir(cfg, base)

    # Initial data: use pre-materialised shared data or build it from canonical SI dir
    init_data = Path(cl.get("initial_sft_path",
                            "./training/outputs/_shared_data/objA_sft.jsonl"))
    if not init_data.exists():
        raise FileNotFoundError(
            f"{init_data} not found — run `python -m training.prepare_data` first")

    current_ckpt = cfg["model"]["path"]     # starts from base model
    current_data = init_data

    for it in range(max_iters):
        iter_dir = base / f"iter_{it}"
        iter_dir.mkdir(exist_ok=True)
        data_dir = iter_dir / "data"
        data_dir.mkdir(exist_ok=True)
        ckpt_dir = iter_dir / "ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        evolve_dir = iter_dir / "evolve"

        logger.info("========== Iteration %d / %d (method=%s) ==========", it + 1, max_iters, method)

        # Resume: if iter already has a usable merged/full checkpoint, skip train+merge.
        # Checked locations: iter_N/merged/ (RS), iter_N/ckpt/merged/ (SFT/Distill).
        resumed_ckpt = None
        for cand in (iter_dir / "merged", ckpt_dir / "merged"):
            if (cand / "config.json").exists() and (
                any(cand.glob("model*.safetensors"))
                or any(cand.glob("pytorch_model*.bin"))
            ):
                resumed_ckpt = str(cand)
                break
        if resumed_ckpt is not None:
            logger.info("[iter %d] Resume: found merged ckpt at %s — skipping train", it, resumed_ckpt)
            current_ckpt = resumed_ckpt
        else:
            # 1. Train
            logger.info("[iter %d] Training for %d epochs", it, cl["evolve_every_epochs"])
            train_out = train_fn(
                cfg, data_path=current_data, ckpt_dir=ckpt_dir,
                resume_from=current_ckpt, run_suffix=f"iter{it}",
            )
            # LoRA adapter → merged full weights so next iter + evolve step can
            # load the student via AutoModelForCausalLM / vLLM.
            current_ckpt = _merge_lora_if_needed(Path(train_out), base_model_path=cfg["model"]["path"])

        # 2. Evolve library (skip on last iter if no-op requested, or always if disabled)
        if cl.get("disable_evolve", False):
            logger.info("[iter %d] disable_evolve=true — skipping evolve step", it)
        elif it < max_iters - 1 or cl.get("evolve_after_final", False):
            evolve_mode = cl.get("evolve_mode", "full")
            if evolve_mode == "lite":
                # Cheap path: skips Phase A/B/D/E/F/G/H. One student-vLLM
                # generation against bootstrap-pool failures → propose + write.
                # No PF helper API. No data refresh (no Phase A trajectories).
                from .evolve_step_lite import lite_evolve_step
                logger.info("[iter %d] Running LITE library-evolve step", it)

                teacher_call = None
                if cl.get("enable_teacher_review", False):
                    import os
                    from src.skills_agent.eval.model_loader import APIModelWrapper
                    api_cfg = cfg.get("api_models", {}).get(
                        cl.get("teacher_model_key", "gpt"),
                        {"provider": "openai", "model_name": ""},
                    )
                    _key_env = {
                        "openai": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY",
                        "google": "GOOGLE_API_KEY",
                    }[api_cfg["provider"]]
                    _teacher = APIModelWrapper(
                        provider=api_cfg["provider"],
                        model_name=api_cfg["model_name"],
                        api_key=os.environ.get(_key_env),
                        max_tokens=64,
                        temperature=0.0,
                    )
                    def teacher_call(messages, _t=_teacher):
                        return _t.generate_from_messages(messages, max_tokens=64, temperature=0.0)

                lite_evolve_step(
                    student_ckpt_path=current_ckpt,
                    library_dir=str(lib_dir),
                    bootstrap_trajectories_path=cl.get(
                        "bootstrap_trajectories_path",
                        "outputs/bootstrap_rollouts/epoch_0/trajectories/trajectories.jsonl",
                    ),
                    epoch=it,
                    n_failures=cl.get("lite_evolve_n_failures", 20),
                    max_new_skills=cl.get("lite_evolve_max_skills", 3),
                    temperature=cl.get("lite_evolve_temperature", 0.3),
                    tensor_parallel_size=cl.get(
                        "lite_evolve_tp_size",
                        cfg.get("rollout", {}).get("tensor_parallel_size", 2),
                    ),
                    gpu_memory_utilization=cl.get(
                        "lite_evolve_gpu_memory_utilization", 0.85,
                    ),
                    max_model_len=cl.get(
                        "lite_evolve_max_model_len",
                        cfg.get("rollout", {}).get("max_model_len", 8192),
                    ),
                    enable_compile_check=cl.get("enable_compile_check", True),
                    enable_teacher_review=cl.get("enable_teacher_review", False),
                    teacher_review_threshold=cl.get("teacher_review_threshold", 0.5),
                    teacher_call=teacher_call,
                )
                # Lite mode does not refresh objA data — initial _shared_data
                # stays in use for all iters. Library accumulates new skills
                # across iters and the next rollout setup picks them up.
            else:
                logger.info("[iter %d] Running library-evolve step", it)
                run_evolve_step(
                    base_config_path=cl["self_improving_config"],
                    student_ckpt_path=current_ckpt,
                    iter_output_dir=str(evolve_dir),
                    library_dir=str(lib_dir),
                    num_epochs=1,
                    seed_samples=cl.get("evolve_val_samples", 80),
                    prefilter_baseline_failures=cl.get("prefilter_baseline_failures", False),
                    prefilter_cap_k=cl.get("prefilter_cap_k", 0),
                )

                # 3. Refresh training data from the new trajectories
                if cl.get("refresh_data_mode", "immediate") == "immediate":
                    refresh_objA(
                        self_improving_dir=str(evolve_dir),
                        out_dir=str(data_dir),
                        enabled_signals=cfg.get("signals", {}).get("enabled", "all"),
                        threshold=cfg.get("signals", {}).get("threshold", 0.25),
                        formats=["sft", "prompt"],
                        signal_mode=cfg.get("signals", {}).get("mode", "coarse"),
                    )
                    current_data = data_dir / "objA_sft.jsonl"

    logger.info("Closed-loop complete. Final ckpt: %s", current_ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
