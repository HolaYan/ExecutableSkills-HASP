"""Rebuild the shared bootstrap SFT data (`_shared_data/objA_*.jsonl`).

One-shot rollout job. Directly loads the 3 active datasets from
`data/web_search/`, skips the first `--test-reserve` (200) samples per
dataset as test, and runs a **PF-aware** ReAct rollout (skills + PFs
ENABLED, teacher = gpt-4o) over the training pool to produce full
trajectories. Signals are then scored in **coarse** 4-family mode and
written to `training/outputs/_shared_data/`.

NOTE on naming: this is NOT the full self-improving pipeline. It only
runs its Phase A (seed execution with current skill library) + Phase H
(training data builder). Skill evolution (Phase B-F) is skipped entirely.
Trajectories land under `outputs/bootstrap_rollouts/` — NOT
`outputs/self_improving/` — to avoid confusion.

Student model defaults to `openai/gpt-oss-20b` (vLLM, TP=2). Downstream
E1–E6 all read from `_shared_data/` so they pick up the new bootstrap
automatically without config changes.

Budget knobs:
  max_steps=8, max_search_calls=4  → SerpAPI usage capped, shorter tails.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def load_training_pool(
    datasets: List[str],
    raw_dir: str,
    test_reserve: int,
    per_dataset_cap: Optional[int] = None,
    test_reserve_overrides: Optional[Dict[str, int]] = None,
    per_dataset_cap_overrides: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Return [{dataset_name, sample_id, question, gold_answers}, ...] across
    all datasets, each sliced `[test_reserve : test_reserve + per_dataset_cap]`
    (no cap if None).

    Per-dataset overrides take precedence over the global `test_reserve` /
    `per_dataset_cap`. Used by the math bootstrap where AMC23 and GameOf24
    have different test-set sizes (20 vs 100) defined in
    `self_improving/configs/self_improving_math.yaml::test_samples_overrides`.
    """
    pool: List[Dict[str, Any]] = []
    root = Path(raw_dir)
    test_reserve_overrides = test_reserve_overrides or {}
    per_dataset_cap_overrides = per_dataset_cap_overrides or {}
    for ds in datasets:
        ds_test_reserve = test_reserve_overrides.get(ds, test_reserve)
        ds_cap = per_dataset_cap_overrides.get(ds, per_dataset_cap)
        path = root / f"{ds}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        rows = [json.loads(l) for l in open(path) if l.strip()]
        start = ds_test_reserve
        stop = (start + ds_cap) if ds_cap else len(rows)
        training_rows = rows[start:stop]
        logger.info(
            "%s: total=%d, test_reserved=%d, cap=%s, training_pool=%d",
            ds, len(rows), ds_test_reserve, ds_cap, len(training_rows),
        )
        for i, r in enumerate(training_rows):
            gold = r.get("answer", "")
            gold_list = [gold] if isinstance(gold, str) else gold
            entry = {
                "dataset_name": ds,
                "sample_id": f"{ds}_{i + ds_test_reserve}",
                "question": r["question"],
                "gold_answers": gold_list,
            }
            # Code-domain rows carry test cases + metadata; harmless for math/web.
            # `eval_test_code` / `entry_point` / `variant` are EvalPlus + BCB
            # additions used by the new sandbox-script-runner path.
            for k in ("public_tests", "private_tests", "metadata", "difficulty",
                      "starter_code", "eval_test_code", "entry_point", "variant"):
                if k in r:
                    entry[k] = r[k]
            pool.append(entry)
    logger.info("Combined training pool: %d samples across %d datasets", len(pool), len(datasets))
    return pool


def episode_to_trajectory(
    sample: Dict[str, Any],
    episode,
    max_steps: int,
    epoch: int,
    domain: str = "web_search",
):
    """Convert a SkillAgentRunner `Episode` into an `EpisodeTrajectory`.

    Mirrors the extraction logic in `self_improving/pipeline.py::_phase_a_seed_execution`
    so PF activations / proposed-vs-final actions / context injections are all
    preserved for downstream signal scoring and DPO pair building.

    Key insight: `episode.trace[i]` stores the POST-intervention action. The
    original (student-proposed) action lives in `episode.pf_records` under
    each activated MODIFY_ACTION entry's `original_action` / `original_arg`.

    `domain="math"` selects MathAnswerEvaluator (numeric+LaTeX equivalence,
    so "7" matches "7.0", "\\boxed{1/2}" matches "0.5", etc.) instead of
    the string-EM AnswerEvaluator used for web_search.
    """
    from training.signals.trajectory import (
        EpisodeTrajectory, StepRecord, PFActivationRecord,
    )
    from src.skills_agent.eval.metrics import (
        AnswerEvaluator, MathAnswerEvaluator, CodeAnswerEvaluator,
    )

    pf_records = list(getattr(episode, "pf_records", None) or [])

    # Episode-level selected PFs = union over step-level skill_ids
    selected_pf_ids = sorted({
        r["skill_id"] for r in pf_records
        if r.get("skill_id")
    })

    traj = EpisodeTrajectory(
        sample_id=str(sample["sample_id"]),
        question=sample["question"],
        gold_answers=list(sample["gold_answers"]),
        dataset_name=sample["dataset_name"],
        difficulty_score=0,
        skills_enabled=True,
        selected_pf_ids=selected_pf_ids,
        epoch=epoch,
    )

    trace = getattr(episode, "trace", []) or []
    prior_search = 0
    prior_read = 0
    for step_idx, step in enumerate(trace):
        a = getattr(step, "action", None)
        final_type = getattr(a, "type", "") if a else ""
        final_arg = ""
        if a is not None:
            final_arg = (getattr(a, "query", None) or getattr(a, "doc_id", None) or "") or ""

        # PF records scoped to THIS step
        step_pf_recs = [r for r in pf_records if r.get("step", -1) == step_idx]

        # Recover proposed action from any activated MODIFY_ACTION intervention
        proposed_type = final_type
        proposed_arg = final_arg
        ctx_injections: List[str] = []
        pf_activations_list: List[PFActivationRecord] = []
        for r in step_pf_recs:
            if not r.get("activated"):
                continue
            itype = r.get("intervention_type")
            if itype == "modify_action" and r.get("original_action") is not None:
                proposed_type = r["original_action"]
                proposed_arg = r.get("original_arg") or ""
                break  # first activated modifier wins

        for r in step_pf_recs:
            if r.get("activated") and r.get("intervention_type") == "inject_context":
                txt = r.get("context_text")
                if txt:
                    ctx_injections.append(txt)
            # Build PFActivationRecord for every PF record on this step
            pf_activations_list.append(PFActivationRecord(
                pf_id=r.get("skill_id", ""),
                activated=bool(r.get("activated", False)),
                intervention_type=r.get("intervention_type", "noop"),
                reason=r.get("reason", ""),
                original_action=proposed_type if r.get("activated") else None,
                original_arg=proposed_arg if r.get("activated") else None,
                modified_action=r.get("new_action_type") if r.get("intervention_type") == "modify_action" else None,
                modified_arg=r.get("new_action_arg") if r.get("intervention_type") == "modify_action" else None,
                injected_text=(r.get("context_text") or "")[:200] if r.get("intervention_type") == "inject_context" else None,
            ))

        was_modified = (proposed_type != final_type) or (proposed_arg != final_arg)
        thought = (getattr(step, "thought", "") or "")[:500]

        # Observation summary for signal hooks (e.g. empty_results detection)
        obs = getattr(step, "observation", None)
        obs_content = ""
        if obs is not None:
            obs_content = (getattr(obs, "content", None) or "") or ""

        step_context = {
            "step_count": step_idx,
            "search_count": prior_search,
            "read_count": prior_read,
            "has_read": prior_read > 0,
            "empty_results": bool(obs_content and "No results found" in obs_content),
            "contradictory_sources": False,
            "max_steps": max_steps,
        }

        traj.steps.append(StepRecord(
            step_index=step_idx,
            proposed_action_type=proposed_type,
            proposed_action_arg=str(proposed_arg),
            proposed_reasoning=thought,
            final_action_type=final_type,
            final_action_arg=str(final_arg),
            was_modified=was_modified,
            pf_activations=pf_activations_list,
            context_injections=ctx_injections,
            observation_summary=obs_content[:300],
            step_context_snapshot=step_context,
        ))

        if final_type == "SEARCH":
            prior_search += 1
        elif final_type == "READ":
            prior_read += 1

    final_answer = ""
    if episode.final:
        final_answer = episode.final.get("answer", "") or ""
    try:
        if domain == "math":
            em = bool(MathAnswerEvaluator.exact_match(final_answer, sample["gold_answers"]))
            f1 = float(MathAnswerEvaluator.f1_score(final_answer, sample["gold_answers"]))
        elif domain == "code":
            # Two scoring modes:
            #   • EvalPlus / BigCodeBench rows ship `eval_test_code` (combined
            #     test-driver script). Pass it through to the script-runner
            #     path of `CodeAnswerEvaluator`.
            #   • LCB rows ship `public_tests + private_tests`; legacy path.
            eval_test_code = sample.get("eval_test_code") or ""
            entry_point = sample.get("entry_point") or (
                (sample.get("metadata") or {}).get("entry_point")
            )
            tests = (sample.get("public_tests") or []) + (sample.get("private_tests") or [])
            func_name = (sample.get("metadata") or {}).get("func_name") or entry_point
            em = bool(CodeAnswerEvaluator.exact_match(
                final_answer, tests, func_name=func_name,
                eval_test_code=eval_test_code, entry_point=entry_point,
            ))
            f1 = float(CodeAnswerEvaluator.f1_score(
                final_answer, tests, func_name=func_name,
                eval_test_code=eval_test_code, entry_point=entry_point,
            ))
        else:
            _ev = AnswerEvaluator()
            em = bool(_ev.exact_match(final_answer, sample["gold_answers"]))
            f1 = float(_ev.f1_score(final_answer, sample["gold_answers"]))
    except Exception:
        em, f1 = False, 0.0
    traj.final_answer = final_answer
    traj.exact_match = em
    traj.f1_score = f1
    traj.compute_stats()
    return traj


def _load_done_sample_ids(traj_path: Path) -> set:
    """Read existing trajectories.jsonl (if any), return set of completed sample_ids
    for resume. Tolerant to partial / truncated last line."""
    done = set()
    if not traj_path.exists():
        return done
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sample_id")
                if sid:
                    done.add(str(sid))
            except Exception:
                continue  # last truncated line → skip
    return done


def _trim_trajectories_to_pool(
    traj_path: Path, pool_sample_ids: set,
) -> tuple:
    """Rewrite trajectories.jsonl keeping only entries whose sample_id is in the
    current pool. Used when pool size has been narrowed between runs (e.g.
    per_dataset_cap changed). Returns (kept, dropped) counts."""
    if not traj_path.exists():
        return (0, 0)
    kept_lines: List[str] = []
    dropped = 0
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = str(obj.get("sample_id", ""))
            except Exception:
                continue
            if sid in pool_sample_ids:
                kept_lines.append(line)
            else:
                dropped += 1
    if dropped > 0:
        with open(traj_path, "w", encoding="utf-8") as f:
            for l in kept_lines:
                f.write(l + "\n")
    return (len(kept_lines), dropped)


def run_rollouts(
    model_path: str,
    pool: List[Dict[str, Any]],
    traj_out_dir: str,
    max_steps: int,
    max_search_calls: int,
    max_read_calls: int,
    parallel_episodes: int,
    tp_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    teacher_model: str = "",
    pf_top_k: int = 10,
    chunk_size: int = 200,
    epoch: int = 0,
) -> int:
    """Run SKILL-AWARE ReAct rollout over `pool`, persist trajectories.jsonl.

    - student  : `model_path` via vLLM (e.g. openai/gpt-oss-20b, TP=2)
    - teacher  : `teacher_model` via OpenAI API (e.g. gpt-4o) — handles PF
                 interventions and PF selection
    - skills   : ENABLED (PF-aware trajectories → richer SFT signal)
    - resume   : processes pool in `chunk_size` batches, appends to
                 trajectories.jsonl after each chunk; on restart skips
                 sample_ids already present in the file
    """
    from src.skills_agent.eval.agent_runner import RunnerConfig
    from src.skills_agent.eval.tools import ToolEnvironment
    from src.skills_agent.eval.model_loader import load_model_vllm
    from src.skills_agent.agent.skill_agent_runner import SkillAgentRunner
    from src.skills_agent.agent.config import SkillAgentConfig
    from src.skills_agent.skills.skill import SkillLibrary

    serp = os.environ.get("SERPAPI_API_KEY", "")
    oai = os.environ.get("OPENAI_API_KEY", "")
    if not oai:
        raise RuntimeError("OPENAI_API_KEY unset — required for teacher + PFSelector")

    traj_path = Path(traj_out_dir) / f"epoch_{epoch}" / "trajectories" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: filter pool to pending ──────────────────────────────
    done_ids = _load_done_sample_ids(traj_path)
    if done_ids:
        logger.info("Resume: %d sample_ids already in %s; skipping them", len(done_ids), traj_path)
    pending = [s for s in pool if str(s["sample_id"]) not in done_ids]
    logger.info("Pending rollout count: %d / %d", len(pending), len(pool))
    if not pending:
        logger.info("Nothing to do — trajectories already complete.")
        return len(pool)

    logger.info("Loading vLLM student %s (TP=%d)", model_path, tp_size)
    model, tokenizer = load_model_vllm(
        model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=128,
    )

    # Skills + PFs ENABLED; a hosted PF helper serves the PF handlers and PFSelector.
    # `enable_skill_handlers=False`: matches configs/agent_eval.yaml which explicitly
    # disables the vote-based verify_* handler cascade. Those handlers fire 5-7 times
    # per proposed FINAL, each a serial ~2s PF helper call → 10-15s idle per FINAL,
    # which killed the previous bootstrap run after 4h stuck in chunk 0.
    skill_cfg = SkillAgentConfig(
        skill_library_path="skills/web",
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
    )
    runner_cfg = RunnerConfig(
        max_steps=max_steps,
        max_search_calls=max_search_calls,
        max_read_calls=max_read_calls,
        timeout_seconds=300,
        model_type="base",
        serpapi_key=serp,
        openai_key=oai,
    )
    env = ToolEnvironment(serpapi_key=serp, openai_key=oai)
    skill_library = SkillLibrary.load_from_directory("skills/textual/web")

    runner = SkillAgentRunner(
        model=model,
        tokenizer=tokenizer,
        config=runner_cfg,
        env=env,
        skill_library=skill_library,
        skill_config=skill_cfg,
    )

    # ── Chunked execution with incremental save ─────────────────────
    n_done = len(done_ids)
    n_em = 0
    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start: chunk_start + chunk_size]
        logger.info(
            "Chunk %d-%d / %d — running %d episodes (parallel=%d)",
            chunk_start, chunk_start + len(chunk) - 1, len(pending),
            len(chunk), parallel_episodes,
        )
        batch = [
            {
                "sample_id": s["sample_id"],
                "question": s["question"],
                "gold_answers": s["gold_answers"],
            }
            for s in chunk
        ]
        episodes = runner.run_batch(
            batch, mode="clean",
            parallel_episodes=parallel_episodes,
            verbose=False,
        )
        # Append this chunk's trajectories immediately (resume-friendly)
        with open(traj_path, "a", encoding="utf-8") as f:
            for s, ep in zip(chunk, episodes):
                traj = episode_to_trajectory(s, ep, max_steps=max_steps, epoch=epoch)
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                n_done += 1
                if traj.exact_match:
                    n_em += 1
        logger.info("  → saved. progress %d/%d  (EM-so-far %d)", n_done, len(pool), n_em)

    logger.info("Bootstrap rollout done: %d trajectories in %s", n_done, traj_path)
    return n_done


def build_sft(
    traj_out_dir: str,
    sft_out_dir: str,
    enabled_signals: str = "all",
    threshold: float = 0.25,
) -> Dict[str, Path]:
    """Convert trajectories → objA_sft/dpo/prompts jsonls in coarse mode."""
    from training.data.use_pfs_builder import UsePFsBuilder, UsePFsBuilderConfig
    from training.data.signal_filter import resolve_enabled_signals
    from training.signals.aggregator import AggregatorConfig, SignalAggregator
    from training.prepare_data import _load_trajectories

    enabled = resolve_enabled_signals(enabled_signals)
    agg = SignalAggregator(AggregatorConfig(
        enabled=enabled, normalize=True, mode="coarse",
    ))

    trajs = _load_trajectories(traj_out_dir)
    if not trajs:
        raise RuntimeError(f"No trajectories found under {traj_out_dir}")

    Path(sft_out_dir).mkdir(parents=True, exist_ok=True)
    builder = UsePFsBuilder(
        UsePFsBuilderConfig(
            output_dir=str(sft_out_dir),
            enabled_signals=enabled,
            threshold=threshold,
            formats=["sft", "dpo", "prompt"],
        ),
        agg,
    )
    outputs = builder.build(trajs)
    logger.info("Wrote bootstrap SFT data → %s", sft_out_dir)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
                    help="Student model for bootstrap rollouts (vLLM). "
                         "gpt-oss-20b's harmony format is incompatible with our "
                         "ReAct parser → switched to Qwen3-30B Thinking (FP8).")
    ap.add_argument("--datasets", nargs="+",
                    default=["Musique_rand1000", "HotpotQA_rand1000", "2WikiMultihopQA_rand1000"])
    ap.add_argument("--raw-dir", default="data/web_search")
    ap.add_argument("--test-reserve", type=int, default=200,
                    help="First N samples per dataset reserved as test")
    ap.add_argument("--per-dataset-cap", type=int, default=None,
                    help="Max training samples per dataset (None = use all available)")
    ap.add_argument("--traj-out", default="outputs/bootstrap_rollouts/",
                    help="Where to write trajectories.jsonl (consumed by E1)")
    ap.add_argument("--sft-out", default="training/outputs/_shared_data/",
                    help="Where to write objA_sft/dpo/prompts.jsonl (consumed by E2-E6)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-search-calls", type=int, default=4,
                    help="SerpAPI call cap per episode")
    ap.add_argument("--max-read-calls", type=int, default=4)
    ap.add_argument("--parallel-episodes", type=int, default=16)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--teacher-model", default="",
                    help="PF helper for PF handlers + PFSelector (OpenAI API)")
    ap.add_argument("--pf-top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="Rollout chunk size (trajectories flushed to disk per chunk → resume granularity)")
    ap.add_argument("--signals", default="all")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--skip-rollout", action="store_true",
                    help="Skip rollout (reuse existing trajectories.jsonl), only rebuild SFT")
    ap.add_argument("--force-fresh", action="store_true",
                    help="Ignore existing trajectories.jsonl, roll out from scratch")
    args = ap.parse_args()

    pool = load_training_pool(
        args.datasets, args.raw_dir, args.test_reserve,
        per_dataset_cap=args.per_dataset_cap,
    )

    if not args.skip_rollout:
        traj_path = Path(args.traj_out) / "epoch_0" / "trajectories" / "trajectories.jsonl"
        if args.force_fresh and traj_path.exists():
            logger.info("--force-fresh: removing %s", traj_path)
            traj_path.unlink()
        run_rollouts(
            model_path=args.model_path,
            pool=pool,
            traj_out_dir=args.traj_out,
            max_steps=args.max_steps,
            max_search_calls=args.max_search_calls,
            max_read_calls=args.max_read_calls,
            parallel_episodes=args.parallel_episodes,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            teacher_model=args.teacher_model,
            pf_top_k=args.pf_top_k,
            chunk_size=args.chunk_size,
        )
    else:
        logger.info("--skip-rollout set — using existing trajectories")

    build_sft(
        traj_out_dir=args.traj_out,
        sft_out_dir=args.sft_out,
        enabled_signals=args.signals,
        threshold=args.threshold,
    )

    logger.info("Done. Outputs:")
    logger.info("  trajectories  : %s/epoch_0/trajectories/trajectories.jsonl", args.traj_out)
    logger.info("  SFT bootstrap : %s/objA_sft.jsonl", args.sft_out)


if __name__ == "__main__":
    main()
