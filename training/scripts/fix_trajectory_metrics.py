"""Offline re-score `exact_match` / `f1_score` fields in an existing
trajectories.jsonl and rebuild the downstream `_shared_data/` outputs.

The bootstrap rollout (build_bootstrap_sft.py) incorrectly passed a string
to `compute_metrics(episode, gold)` which expects an Episode object. Every
trajectory got `exact_match=False, f1_score=0.0`. This script walks the
jsonl, re-scores via `AnswerEvaluator` (which DOES take strings), writes
the file back in place, then invokes `UsePFsBuilder` to regenerate
`objA_{sft,dpo,prompts}.jsonl` with correct S4 signal scores.

Usage:
    python training/scripts/fix_trajectory_metrics.py \\
        --traj-out outputs/bootstrap_rollouts/ \\
        --sft-out  training/outputs/_shared_data/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def rescore_trajectories(traj_path: Path) -> dict:
    from src.skills_agent.eval.metrics import AnswerEvaluator
    ev = AnswerEvaluator()

    rows = []
    n_em_old = 0
    n_em_new = 0
    f1_sum_old = 0.0
    f1_sum_new = 0.0

    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_em_old += int(bool(obj.get("exact_match", False)))
            f1_sum_old += float(obj.get("f1_score", 0.0))

            final = obj.get("final_answer", "") or ""
            gold = obj.get("gold_answers", []) or []
            em = bool(ev.exact_match(final, gold))
            f1 = float(ev.f1_score(final, gold))
            obj["exact_match"] = em
            obj["f1_score"] = f1

            n_em_new += int(em)
            f1_sum_new += f1
            rows.append(obj)

    n = len(rows)
    stats = {
        "total": n,
        "em_old": n_em_old,
        "em_new": n_em_new,
        "em_rate_old": n_em_old / max(1, n),
        "em_rate_new": n_em_new / max(1, n),
        "f1_mean_old": f1_sum_old / max(1, n),
        "f1_mean_new": f1_sum_new / max(1, n),
    }

    # Atomic overwrite
    tmp = traj_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(traj_path)
    logger.info("Rescored %d trajectories → %s", n, traj_path)
    logger.info("  EM rate: %d/%d (%.2f%%) → %d/%d (%.2f%%)",
                n_em_old, n, 100 * stats["em_rate_old"],
                n_em_new, n, 100 * stats["em_rate_new"])
    logger.info("  F1 mean: %.4f → %.4f", stats["f1_mean_old"], stats["f1_mean_new"])
    return stats


def rebuild_shared_data(traj_out_dir: str, sft_out_dir: str,
                       enabled_signals: str = "all", threshold: float = 0.25):
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
        raise RuntimeError(f"No trajectories under {traj_out_dir}")

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
    logger.info("Rebuilt shared data → %s", sft_out_dir)
    for fmt, path in outputs.items():
        logger.info("  %s: %s", fmt, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-out", default="outputs/bootstrap_rollouts/")
    ap.add_argument("--sft-out", default="training/outputs/_shared_data/")
    ap.add_argument("--signals", default="all")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--skip-rebuild", action="store_true",
                    help="Only rescore trajectories.jsonl; skip the SFT/DPO/prompts rebuild")
    args = ap.parse_args()

    traj_path = Path(args.traj_out) / "epoch_0" / "trajectories" / "trajectories.jsonl"
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)

    stats = rescore_trajectories(traj_path)

    if args.skip_rebuild:
        logger.info("--skip-rebuild set; stopping after trajectory rescore")
        return

    rebuild_shared_data(
        traj_out_dir=args.traj_out,
        sft_out_dir=args.sft_out,
        enabled_signals=args.signals,
        threshold=args.threshold,
    )

    logger.info("Done. %s", stats)


if __name__ == "__main__":
    main()
