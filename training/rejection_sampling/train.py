"""Rejection-sampling / on-policy-distillation training loop.

Single iteration:
  (1) Roll out current policy with vLLM (`rollout.py`)
  (2) Score rollouts with PFVerifier or TeacherVerifier (`verifier.py`)
  (3) Keep top-k rollouts per prompt → SFT-style training file
  (4) Fine-tune current policy with TRL SFTTrainer
  (5) Optionally iterate N rounds (new ckpt becomes the rollout model)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..sft.trainer import SFTRunner, TrainerHyperparams
from .rollout import Rollouter, RolloutConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def _merge_lora_into_base(base_model_path: str, adapter_dir: str, out_path: str) -> str:
    """Merge a LoRA adapter onto its base model and write a standalone HF dir.

    vLLM rejects adapter-only dirs (no config.json); next iter's rollout needs a
    full model. Runs on CPU to avoid competing with the HF trainer / vLLM for GPU.
    """
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Merging LoRA adapter %s onto base %s → %s", adapter_dir, base_model_path, out_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    Path(out_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tok.save_pretrained(out_path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Merged model saved → %s", out_path)
    return out_path


def _episode_trajectory_score(rows: List[Dict[str, Any]], domain: str = "web_search") -> float:
    """Signal-aware trajectory quality score in [0, 1].

    Combines three quick proxies attached to the already-flattened rows:
      - EM     (episode-level exact_match, binary)            — outcome
      - F1     (token F1 of final answer vs gold)             — correctness
      - Econ   (step-count economy, penalize beyond 6 steps)  — timing/cost
    Weights: 0.70 / 0.20 / 0.10 — outcome dominates but F1 breaks ties
    between equally-EM trajectories, and Econ prefers shorter solutions.

    `domain="math"` uses MathAnswerEvaluator for F1 (numeric / LaTeX-aware)
    so e.g. "27" vs "27.0" still scores 1.0 instead of being tokenized into
    a 0-overlap mismatch by the string evaluator.

    For math single-step rollouts (max_steps=1) the Econ term is degenerate
    (always 1.0) — fine, EM dominates anyway.
    """
    if not rows:
        return 0.0
    last = rows[-1]
    em = 1.0 if last.get("exact_match") else 0.0
    final = last.get("final_answer", "") or ""
    gold = last.get("gold_answers", []) or []
    f1 = 0.0
    try:
        from src.skills_agent.eval.metrics import AnswerEvaluator, MathAnswerEvaluator
        if final and gold:
            if domain == "math":
                f1 = float(MathAnswerEvaluator.f1_score(final, gold))
            else:
                f1 = float(AnswerEvaluator().f1_score(final, gold))
    except Exception:
        f1 = 0.0
    n_steps = len(rows)
    econ = max(0.0, 1.0 - max(0, n_steps - 6) / 4.0)  # 1.0 up to 6 steps, 0.0 at 10+
    return 0.70 * em + 0.20 * f1 + 0.10 * econ


def _filter_by_episode_correctness(
    rollout_path: str,
    out_path: str,
    top_k: int,
    require_exact_match: bool = True,
    min_score: float = 0.0,
    domain: str = "web_search",
    spec_example_gate: bool = False,
) -> str:
    """Episode-level filter: rank candidate episodes per sample by
    `0.7*EM + 0.2*F1 + 0.1*StepEconomy`, keep top-k, emit every step of
    each kept episode as SFT rows.

    `require_exact_match=True` still hard-filters to EM-correct episodes
    (F1/Econ only break ties among successes); set False to keep a
    ranked top-k regardless of EM (useful when success rate is low).
    `min_score` is a floor on the composite score — useful when EM-required
    is False but you still want to drop pure garbage (e.g., 0.15 keeps
    F1>0.25 partial-credit episodes; 0.0 keeps everything top-k.

    `spec_example_gate` (code domain) additionally runs the spec's own
    `>>>` / `assert` examples on each kept episode's answer via
    `SpecExampleVerifier` and drops the ones that fail. This is the
    training-time form of the same acceptance rule used at inference: keep the
    first sample that passes the spec's own examples.
    Episodes whose spec carries no runnable examples score 0.5 and are kept —
    the gate only removes solutions the spec itself contradicts.
    """
    from collections import defaultdict

    verifier = None
    if spec_example_gate:
        try:
            from .verifier import SpecExampleVerifier
            verifier = SpecExampleVerifier()
        except Exception as e:  # sandbox/library unavailable — do not fail the run
            logger.warning("spec_example_gate requested but unavailable (%s); skipping", e)
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    with open(rollout_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row.get("sample_id"), row.get("group_index", 0))
            groups[key].append(row)

    per_sample: Dict[str, List[tuple]] = defaultdict(list)
    for (sid, gidx), rows in groups.items():
        rows.sort(key=lambda r: int(r.get("step_index", 0)))
        em = bool(rows[-1].get("exact_match", False)) if rows else False
        traj_score = _episode_trajectory_score(rows, domain=domain)
        per_sample[sid].append((traj_score, em, gidx, rows))

    n_written = 0
    n_kept_episodes = 0
    n_kept_em = 0
    n_spec_rejected = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, ep_list in per_sample.items():
            ep_list.sort(key=lambda t: t[0], reverse=True)
            kept = 0
            for traj_score, em, gidx, rows in ep_list:
                if require_exact_match and not em:
                    continue
                if traj_score < min_score:
                    continue
                if kept >= top_k:
                    break
                if verifier is not None and rows:
                    last = rows[-1]
                    res = verifier.score(
                        {"question": last.get("question", ""),
                         "entry_point": last.get("entry_point", ""),
                         "public_test_code": last.get("public_test_code", "")},
                        last.get("final_answer") or last.get("generation", ""),
                    )
                    if res.score == 0.0:   # the spec's own examples contradict it
                        n_spec_rejected += 1
                        continue
                for r in rows:
                    msgs = r.get("messages", [])
                    if not msgs:
                        continue
                    assistant_msg = {"role": "assistant", "content": r.get("generation", "")}
                    sft_row = {
                        "messages": list(msgs) + [assistant_msg],
                        # sample_weight now reflects full trajectory quality
                        # (EM + F1 + step-economy), not just EM binary. Downstream
                        # SFT loss will weigh high-quality trajectories more.
                        "sample_weight": traj_score,
                        "trajectory_em": 1.0 if em else 0.0,
                        "sample_id": sid,
                        "step_index": r.get("step_index"),
                        "group_index": gidx,
                    }
                    f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
                    n_written += 1
                kept += 1
                n_kept_episodes += 1
                if em:
                    n_kept_em += 1
    logger.info(
        "Episode-filter: kept %d episodes (top-%d per sample, EM-required=%s, min_score=%.2f, EM-correct=%d) → %d SFT rows at %s",
        n_kept_episodes, top_k, require_exact_match, min_score, n_kept_em, n_written, out_path,
    )
    if verifier is not None:
        logger.info("spec_example_gate: rejected %d episodes whose answer failed "
                    "the spec's own examples", n_spec_rejected)
    return out_path


def _build_evolve_callback(cfg: dict, out_dir: str):
    """In-training evolution (`evolving/`), or None when it is not enabled.

    The run-scoped library it grows lives at `{experiment.output_dir}/{id}/library`
    — one library per experiment, shared by every RS iteration, so PFs admitted
    in iteration k are live for the rollouts of iteration k+1. `_resolve_library_dir`
    points `rollout.skill_library_dir` at it.
    """
    try:
        from evolving.config import EvolveConfig
        from evolving.callback import build_callback
    except Exception as e:
        logger.warning("evolving/ unavailable (%s); training without it", e)
        return None
    ec = EvolveConfig.from_cfg(cfg)
    if not ec.enabled:
        return None
    ec.library_dir = ec.library_dir or _evolving_library_dir(cfg)
    return build_callback(ec, out_dir)


def _evolving_library_dir(cfg: dict) -> str:
    return str(Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["id"] / "library")


def _resolve_library_dir(cfg: dict) -> str:
    """The library rollouts should load: the evolving copy when evolution is on
    (so grown PFs are actually used), otherwise the configured one."""
    configured = cfg["rollout"].get("skill_library_dir", "skills")
    if not (cfg.get("evolve") or {}).get("enabled"):
        return configured
    lib = Path(_evolving_library_dir(cfg))
    if lib.is_dir():
        logger.info("evolving: rollouts load the grown library at %s", lib)
        return str(lib)
    return configured   # generation 0 has not been seeded yet


def run_rs_iteration(cfg: dict, iter_idx: int, current_model_path: str) -> str:
    """One round of rejection sampling + SFT. Returns updated model path."""
    exp = cfg["experiment"]["id"]
    base = Path(cfg["experiment"]["output_dir"]) / exp / f"iter_{iter_idx}"
    base.mkdir(parents=True, exist_ok=True)

    # 1. Rollout (skill-aware full episodes via inference framework)
    rc = RolloutConfig(
        model_path=current_model_path,
        prompts_path=cfg["data"]["prompts_path"],
        output_dir=str(base / "rollouts"),
        skill_library_dir=_resolve_library_dir(cfg),
        raw_data_dir=cfg["rollout"].get("raw_data_dir", "data/web_search"),
        group_size=cfg["rollout"].get("group_size", 4),
        tensor_parallel_size=cfg["rollout"].get("tensor_parallel_size", 2),
        gpu_memory_utilization=cfg["rollout"].get("gpu_memory_utilization", 0.90),
        max_model_len=cfg["rollout"].get("max_model_len", 8192),
        max_num_seqs=cfg["rollout"].get("max_num_seqs", 128),
        parallel_episodes=cfg["rollout"].get("parallel_episodes", 16),
        max_steps=cfg["rollout"].get("max_steps", 10),
        max_search_calls=cfg["rollout"].get("max_search_calls", 8),
        max_read_calls=cfg["rollout"].get("max_read_calls", 8),
        timeout_seconds=cfg["rollout"].get("timeout_seconds", 300),
        pf_top_k=cfg["rollout"].get("pf_top_k", 10),
        enable_pf_selection=cfg["rollout"].get("enable_pf_selection", True),
        pf_selection_model=cfg["rollout"].get("pf_selection_model", ""),
        mode=cfg["rollout"].get("mode", "clean"),
        domain=cfg.get("domain", "web_search"),
    )
    rollout_path = Rollouter(rc).run()

    # 2. Episode-level filter: keep successful trajectories only.
    sft_path = str(base / "train.jsonl")
    _filter_by_episode_correctness(
        str(rollout_path),
        sft_path,
        top_k=cfg["filter"].get("top_k", 2),
        require_exact_match=cfg["filter"].get("require_exact_match", True),
        min_score=cfg["filter"].get("min_score", 0.0),
        domain=cfg.get("domain", "web_search"),
        # code domain: default on — the spec's own examples are free to run
        spec_example_gate=cfg["filter"].get(
            "spec_example_gate", cfg.get("domain") == "code"),
    )

    # 4. SFT
    t = cfg["trainer"]
    ckpt_dir = str(base / "ckpt")
    hp = TrainerHyperparams(
        model_path=current_model_path,
        output_dir=ckpt_dir,
        data_path=sft_path,
        num_train_epochs=t.get("num_train_epochs", 1.0),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 5e-6),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_seq_length=t.get("max_seq_length", 4096),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        save_every_n_epochs=t.get("save_every_n_epochs"),
        deepspeed=t.get("deepspeed"),
        use_lora=t.get("use_lora", False),
        lora_r=t.get("lora_r", 16),
        lora_alpha=t.get("lora_alpha", 32),
        report_to=t.get("report_to", "wandb"),
        run_name=f"{exp}_iter{iter_idx}",
        seed=cfg["experiment"].get("seed", 42) + iter_idx,
        evolve_callback=_build_evolve_callback(cfg, str(base)),
    )
    SFTRunner(hp).run()

    if t.get("use_lora", False):
        merged_dir = str(base / "merged")
        _merge_lora_into_base(current_model_path, ckpt_dir, merged_dir)
        return merged_dir
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    n_iters = cfg.get("n_iterations", 1)
    exp = cfg["experiment"]["id"]
    exp_base = Path(cfg["experiment"]["output_dir"]) / exp
    use_lora = cfg.get("trainer", {}).get("use_lora", False)

    model_path = cfg["model"]["path"]
    for i in range(n_iters):
        iter_base = exp_base / f"iter_{i}"
        merged_dir = iter_base / "merged"
        ckpt_dir = iter_base / "ckpt"

        if merged_dir.exists() and (merged_dir / "config.json").exists():
            logger.info("===== RS iteration %d / %d — resume: found merged at %s =====", i + 1, n_iters, merged_dir)
            model_path = str(merged_dir)
            continue
        if use_lora and ckpt_dir.exists() and (ckpt_dir / "adapter_model.safetensors").exists():
            logger.info("===== RS iteration %d / %d — resume: merging existing adapter =====", i + 1, n_iters)
            _merge_lora_into_base(model_path, str(ckpt_dir), str(merged_dir))
            model_path = str(merged_dir)
            continue

        logger.info("===== RS iteration %d / %d =====", i + 1, n_iters)
        model_path = run_rs_iteration(cfg, i, model_path)
    logger.info("Final model: %s", model_path)


if __name__ == "__main__":
    main()
