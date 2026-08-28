"""MATH-domain bootstrap: rollout `data/math/{AMC23,GameOf24}.jsonl` training
pool with Qwen/Qwen2.5-7B-Instruct through SkillAgentRunner (math PFs
enabled, teacher = gpt-4o) and build `_shared_data_math/objA_*.jsonl`.

Active math datasets (replaced the old MATH-500 setup):
  - AIME24:    30 total — ALL 30 reserved as test, NOT in bootstrap pool
  - AMC23:     ~40 total — first 20 test, remaining ~20 → training pool
  - GameOf24:  1362 total — first 100 test, then capped to 400 → training pool

Per-dataset test boundaries match `self_improving/configs/self_improving_math.yaml::
validation.test_samples_overrides`. AIME24 is excluded entirely from --datasets.
Rollout uses the math domain: no SEARCH/READ, one FINAL step per problem.

Run via SLURM:  sbatch training/scripts/sbatch_math_bootstrap.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

# Reuse helpers from web-search bootstrap
from training.scripts.build_bootstrap_sft import (
    _load_done_sample_ids,
    build_sft,
    episode_to_trajectory,
    load_training_pool,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def run_math_rollouts(
    model_path: str,
    pool: List[Dict[str, Any]],
    traj_out_dir: str,
    max_steps: int,
    parallel_episodes: int,
    tp_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    teacher_model: str = "",
    pf_top_k: int = 6,
    chunk_size: int = 100,
    epoch: int = 0,
) -> int:
    """Skill-aware MATH rollout. Sets RunnerConfig.domain='math' so
    _build_math_system_prompt is used, and skill_library points at
    self_improving/skills_math/seed."""
    from src.skills_agent.eval.agent_runner import RunnerConfig
    from src.skills_agent.eval.tools import ToolEnvironment
    from src.skills_agent.eval.model_loader import load_model_vllm
    from src.skills_agent.agent.skill_agent_runner import SkillAgentRunner
    from src.skills_agent.agent.config import SkillAgentConfig
    from src.skills_agent.skills.skill import SkillLibrary

    oai = os.environ.get("OPENAI_API_KEY", "")
    if not oai:
        raise RuntimeError("OPENAI_API_KEY unset — required for teacher")

    traj_path = Path(traj_out_dir) / f"epoch_{epoch}" / "trajectories" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done_sample_ids(traj_path)
    if done_ids:
        logger.info("Resume: %d sample_ids in %s — skipping", len(done_ids), traj_path)
    pending = [s for s in pool if str(s["sample_id"]) not in done_ids]
    logger.info("Pending math rollouts: %d / %d", len(pending), len(pool))
    if not pending:
        return len(pool)

    logger.info("Loading vLLM student %s (TP=%d, fp8)", model_path, tp_size)
    model, tokenizer = load_model_vllm(
        model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=256,
        quantization="fp8",
    )

    skill_cfg = SkillAgentConfig(
        skill_library_path="skills/math",
        skill_source_format="markdown",
        skills_enabled=True,
        pf_only_mode=True,
        enable_program_functions=True,
        enable_skill_handlers=False,     # vote-cascade handlers not relevant to math
        handler_vote_threshold=4,
        enable_pf_selection=True,
        pf_selection_model=teacher_model,
        pf_top_k=pf_top_k,
        teacher_api_provider="openai",
        teacher_api_model=teacher_model,
        teacher_api_key=oai,
    )
    runner_cfg = RunnerConfig(
        max_steps=max_steps,
        max_search_calls=0,
        max_read_calls=0,
        timeout_seconds=300,
        model_type="base",
        serpapi_key="",              # math doesn't search
        openai_key=oai,
        domain="math",               # critical: switches system prompt
    )
    env = ToolEnvironment(serpapi_key="", openai_key=oai)
    skill_library = SkillLibrary.load_from_directory("skills/textual/math")

    runner = SkillAgentRunner(
        model=model, tokenizer=tokenizer,
        config=runner_cfg, env=env,
        skill_library=skill_library,
        skill_config=skill_cfg,
    )

    n_done = len(done_ids)
    n_em = 0
    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start: chunk_start + chunk_size]
        logger.info("Chunk %d-%d / %d — running %d episodes (parallel=%d)",
                    chunk_start, chunk_start + len(chunk) - 1, len(pending),
                    len(chunk), parallel_episodes)
        batch = [
            {"sample_id": s["sample_id"], "question": s["question"],
             "gold_answers": s["gold_answers"]}
            for s in chunk
        ]
        episodes = runner.run_batch(
            batch, mode="clean",
            parallel_episodes=parallel_episodes, verbose=False,
        )
        with open(traj_path, "a", encoding="utf-8") as f:
            for s, ep in zip(chunk, episodes):
                traj = episode_to_trajectory(s, ep, max_steps=max_steps, epoch=epoch, domain="math")
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                n_done += 1
                if traj.exact_match:
                    n_em += 1
        logger.info("  → saved. progress %d/%d  (EM %d)",
                    n_done, len(pool), n_em)

    logger.info("Math bootstrap done: %d trajectories in %s", n_done, traj_path)
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    # Default: AMC23 + GameOf24 (AIME24 excluded — all 30 rows are test).
    ap.add_argument("--datasets", nargs="+", default=["AMC23", "GameOf24"])
    ap.add_argument("--raw-dir", default="data/math")
    ap.add_argument("--test-reserve", type=int, default=20,
                    help="Default first N samples reserved as test "
                         "(per-dataset overrides via --test-reserve-overrides).")
    ap.add_argument("--per-dataset-cap", type=int, default=None,
                    help="Default max training samples per dataset after test-reserve "
                         "(per-dataset overrides via --per-dataset-cap-overrides).")
    ap.add_argument("--test-reserve-overrides", type=str, default='{"AMC23":20,"GameOf24":100}',
                    help="JSON dict, per-dataset test_reserve override. "
                         "Default: AMC23=20 (first 20 are test), GameOf24=100. "
                         "Matches validation.test_samples_overrides in self_improving_math.yaml.")
    ap.add_argument("--per-dataset-cap-overrides", type=str, default='{"GameOf24":400}',
                    help="JSON dict, per-dataset training-pool cap override. "
                         "Default: GameOf24=400 (cap to keep gpt-4o cost reasonable). "
                         "AMC23 has no cap → uses all ~20 remaining rows after test-reserve.")
    ap.add_argument("--traj-out", default="outputs/bootstrap_rollouts_math/")
    ap.add_argument("--sft-out", default="training/outputs/_shared_data_math/")
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--parallel-episodes", type=int, default=24)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--pf-top-k", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--signals", default="all")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Signal aggregate-reward threshold for SFT filtering. "
                         "0.0 = keep all steps with non-negative score (math pool is small, "
                         "be generous).")
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--force-fresh", action="store_true")
    args = ap.parse_args()

    test_reserve_overrides = json.loads(args.test_reserve_overrides) if args.test_reserve_overrides else {}
    per_dataset_cap_overrides = json.loads(args.per_dataset_cap_overrides) if args.per_dataset_cap_overrides else {}

    pool = load_training_pool(
        args.datasets, args.raw_dir, args.test_reserve,
        per_dataset_cap=args.per_dataset_cap,
        test_reserve_overrides=test_reserve_overrides,
        per_dataset_cap_overrides=per_dataset_cap_overrides,
    )

    if not args.skip_rollout:
        traj_path = Path(args.traj_out) / "epoch_0" / "trajectories" / "trajectories.jsonl"
        if args.force_fresh and traj_path.exists():
            logger.info("--force-fresh: removing %s", traj_path)
            traj_path.unlink()
        run_math_rollouts(
            model_path=args.model_path,
            pool=pool,
            traj_out_dir=args.traj_out,
            max_steps=args.max_steps,
            parallel_episodes=args.parallel_episodes,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            teacher_model=args.teacher_model,
            pf_top_k=args.pf_top_k,
            chunk_size=args.chunk_size,
        )
    else:
        logger.info("--skip-rollout: reusing existing trajectories")

    build_sft(
        traj_out_dir=args.traj_out,
        sft_out_dir=args.sft_out,
        enabled_signals=args.signals,
        threshold=args.threshold,
    )
    logger.info("Done. trajectories=%s, SFT=%s", args.traj_out, args.sft_out)


if __name__ == "__main__":
    main()
