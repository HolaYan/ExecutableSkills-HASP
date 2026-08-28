"""One-shot data preparation.

Builds every variant needed by E1..E15 in a single shared directory:

  training/outputs/_shared_data/
    ├── objA_sft.jsonl
    ├── objA_dpo.jsonl
    ├── objA_prompts.jsonl
    ├── objB_sft.jsonl
    ├── objB_dpo.jsonl
    └── objB_prompts.jsonl

Usage:
    python -m training.prepare_data --self-improving-dir <dir>  \
                                    [--out-dir training/outputs/_shared_data] \
                                    [--signals all] [--threshold 0.25]

The shared jsonls are used by the RS/GRPO yamls (E8..E15) so they can
run independently of any SFT experiment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .sft.train import _load_trajectories, _load_candidates_and_reviews
from .signals import SignalAggregator
from .signals.aggregator import AggregatorConfig
from .data import UsePFsBuilder, EvolveBuilder
from .data.use_pfs_builder import UsePFsBuilderConfig
from .data.evolve_builder import EvolveBuilderConfig
from .data.signal_filter import resolve_enabled_signals

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-improving-dir",
                    default="outputs/self_improving/",
                    help="Root dir of a finished self_improving run")
    ap.add_argument("--out-dir",
                    default="training/outputs/_shared_data/")
    ap.add_argument("--signals", default="all",
                    help="Enabled signals for filtering (default: all)")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--q-skill-threshold", type=float, default=0.5)
    ap.add_argument("--lam-val-gain", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    enabled = resolve_enabled_signals(args.signals)
    agg = SignalAggregator(AggregatorConfig(
        enabled=enabled, normalize=True, mode="coarse",
    ))

    # ---- Objective A ----
    trajs = _load_trajectories(args.self_improving_dir)
    if trajs:
        obj_a = UsePFsBuilder(
            UsePFsBuilderConfig(
                output_dir=str(out),
                enabled_signals=enabled,
                threshold=args.threshold,
                formats=["sft", "dpo", "prompt"],
            ),
            agg,
        )
        obj_a.build(trajs)
    else:
        logger.warning("No trajectories found under %s — skipping Obj A", args.self_improving_dir)

    # ---- Objective B ----
    cands, revs = _load_candidates_and_reviews(args.self_improving_dir)
    if cands and revs:
        obj_b = EvolveBuilder(
            EvolveBuilderConfig(
                output_dir=str(out),
                q_skill_threshold=args.q_skill_threshold,
                lam_val_gain=args.lam_val_gain,
                formats=["sft", "dpo", "prompt"],
            ),
            agg,
        )
        obj_b.build(cands, revs)
    else:
        logger.warning("No candidates/reviews under %s — skipping Obj B", args.self_improving_dir)

    logger.info("Shared data prepared at %s", out)


if __name__ == "__main__":
    main()
