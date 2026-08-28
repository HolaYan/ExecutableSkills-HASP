"""CODE-domain bootstrap: rollout `data/code/{humaneval_plus,mbpp_plus,bigcodebench}.jsonl`
training pool with Qwen/Qwen2.5-7B-Instruct through SkillAgentRunner (code PFs
enabled, teacher = gpt-4o) and build `_shared_data_code/objA_*.jsonl`.

Per-dataset layout (set by `data/code/build_eval_plus_datasets.py`,
testset rows first then train pool):
  humaneval_plus:  328 total → 100 test + 228 train pool (114 problems × 2 variants)
  mbpp_plus:       756 total → 100 test + 656 train pool (328 problems × 2 variants)
  bigcodebench:   1140 total → 100 test + 1040 train pool

`test_reserve_overrides` skips both test AND SI portions (= test_n + si_n) so
bootstrap only sees the train block. Default per-dataset cap of 150 keeps
gpt-4o PF helper cost manageable (3 × 150 = 450 episodes ≤ ~$5 at single-step).

Rollout uses the code domain: no SEARCH/READ, single FINAL action containing
a Python solution. Evaluator runs the solution in a subprocess sandbox
against (public + private) test cases for pass@1.

Run via SLURM:  sbatch training/scripts/sbatch_build_bootstrap_sft_code.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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


def run_code_rollouts(
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
    """Skill-aware CODE rollout. Sets RunnerConfig.domain='code' so
    `_build_code_system_prompt` is used; skill_library points at
    `self_improving/skills_code/seed` and dynamic_program_functions.py
    is exec-loaded so the 12 code PFs register globally.
    """
    from src.skills_agent.eval.agent_runner import RunnerConfig
    from src.skills_agent.eval.tools import ToolEnvironment
    from src.skills_agent.eval.model_loader import load_model_vllm
    from src.skills_agent.agent.skill_agent_runner import SkillAgentRunner
    from src.skills_agent.agent.config import SkillAgentConfig
    from src.skills_agent.skills.skill import SkillLibrary
    from training.common.skill_rollout import _load_dynamic_pfs

    oai = os.environ.get("OPENAI_API_KEY", "")
    if not oai:
        raise RuntimeError("OPENAI_API_KEY unset — required for teacher (PFSelector falls back to heuristic anyway)")

    traj_path = Path(traj_out_dir) / f"epoch_{epoch}" / "trajectories" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done_sample_ids(traj_path)
    if done_ids:
        logger.info("Resume: %d sample_ids in %s — skipping", len(done_ids), traj_path)
    pending = [s for s in pool if str(s["sample_id"]) not in done_ids]
    logger.info("Pending code rollouts: %d / %d", len(pending), len(pool))
    if not pending:
        return len(pool)

    # Side-effect: register the 12 code PFs into the global registry.
    seed_lib = Path("skills/code")
    _load_dynamic_pfs(seed_lib)

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
        skill_library_path=str(seed_lib),
        skill_source_format="markdown",
        skills_enabled=True,
        pf_only_mode=True,
        enable_program_functions=True,
        enable_skill_handlers=False,
        handler_vote_threshold=4,
        enable_pf_selection=True,
        pf_selection_model=teacher_model,
        pf_top_k=pf_top_k,
        teacher_api_provider="openai",
        teacher_api_model=teacher_model,
        teacher_api_key=oai,
        # Inject SKILL.md prompt content for the 12 prompt-only-style code skills.
        # PFs still fire (audit-only mostly) but the prompt is what actually steers.
        enable_prompt_only_skills=True,
    )
    runner_cfg = RunnerConfig(
        max_steps=max_steps,
        max_search_calls=0,
        max_read_calls=0,
        timeout_seconds=600,
        model_type="base",
        serpapi_key="",              # code doesn't search
        openai_key=oai,
        domain="code",               # critical: switches system prompt to code
    )
    env = ToolEnvironment(serpapi_key="", openai_key=oai)
    skill_library = SkillLibrary.load_from_directory(str(seed_lib))

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
             "gold_answers": s.get("gold_answers", [])}
            for s in chunk
        ]
        episodes = runner.run_batch(
            batch, mode="clean",
            parallel_episodes=parallel_episodes, verbose=False,
        )
        with open(traj_path, "a", encoding="utf-8") as f:
            for s, ep in zip(chunk, episodes):
                # Pass the FULL `s` (with public_tests / private_tests / metadata)
                # into episode_to_trajectory so the code branch can run the sandbox.
                traj = episode_to_trajectory(s, ep, max_steps=max_steps, epoch=epoch, domain="code")
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                n_done += 1
                if traj.exact_match:
                    n_em += 1
        logger.info("  → saved. progress %d/%d  (EM %d, %.1f%%)",
                    n_done, len(pool), n_em, 100.0 * n_em / max(1, n_done))

    logger.info("Code bootstrap done: %d trajectories in %s", n_done, traj_path)
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    # Default: all three difficulty buckets
    ap.add_argument("--datasets", nargs="+",
                    default=["humaneval_plus", "mbpp_plus", "bigcodebench"])
    ap.add_argument("--raw-dir", default="data/code")
    ap.add_argument("--test-reserve", type=int, default=100,
                    help="Default first-N test reserve (per-dataset overrides override). "
                         "100 = testset entries written by build_eval_plus_datasets.py.")
    ap.add_argument("--per-dataset-cap", type=int, default=None,
                    help="Default training-pool cap (per-dataset overrides override).")
    # test_reserve = testset entries; everything after is training pool.
    ap.add_argument("--test-reserve-overrides", type=str,
                    default='{"humaneval_plus":100,"mbpp_plus":100,"bigcodebench":100}',
                    help="JSON dict; matches the testset entry count used by "
                         "self_improving/configs/self_improving_code.yaml.")
    ap.add_argument("--per-dataset-cap-overrides", type=str,
                    default='{"humaneval_plus":228,"mbpp_plus":300,"bigcodebench":300}',
                    help="JSON dict, per-dataset training-pool cap. HumanEval pool only "
                         "has 228 rows so cap matches; MBPP+/BCB are capped to keep "
                         "single-step gpt-4o teacher cost in the $5-10 range.")
    ap.add_argument("--traj-out", default="outputs/bootstrap_rollouts_code/")
    ap.add_argument("--sft-out",  default="training/outputs/_shared_data_code/")
    ap.add_argument("--max-steps", type=int, default=2,
                    help="2 = one shot + one PF-driven RETRY for code domain. "
                         "Set to 1 to disable retry (faster, no PF intervention).")
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
                         "0.0 = keep all non-negative; code pool already small.")
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
        run_code_rollouts(
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
