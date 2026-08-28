"""Offline re-score existing Obj-A jsonls under the 4-scalar coarse view.

Reads a jsonl that was produced in "fine" mode (so every row carries
`signal_breakdown`: Dict[sub_id → weighted_value]) and rewrites each
row with:

  - `sample_weight` recomputed from coarse 4-scalar aggregate
  - `signal_breakdown_4`: {correctness, timing, modality, outcome}

The raw `signal_breakdown` is retained for audit.

Usage:
    python training/signals/rescore.py \\
        --in  training/outputs/_shared_data/objA_sft.jsonl \\
        --out training/outputs/_shared_data/objA_sft_coarse.jsonl

Does NOT require running any rollout or loading any trajectory object —
only the pre-stored signal_breakdown is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from .aggregator import AggregatorConfig, SignalAggregator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def rescore_file(
    in_path: str,
    out_path: str,
    family_weights: Dict[str, float] | None = None,
    overwrite_sample_weight: bool = True,
) -> int:
    """Walk jsonl `in_path`, compute 4-scalar coarse view per row from the
    pre-stored `signal_breakdown`, write to `out_path`. Returns row count."""
    cfg = AggregatorConfig(
        enabled=[],   # unused — coarse_from_breakdown doesn't touch signal fns
        mode="coarse",
        family_weights=family_weights,
    )
    agg = SignalAggregator.__new__(SignalAggregator)
    agg.config = cfg

    n = 0
    n_skipped = 0
    sum_old = 0.0
    sum_new = 0.0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("signal_breakdown") or {}
            if not isinstance(raw, dict) or not raw:
                n_skipped += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            coarse = agg.coarse_from_breakdown(raw)
            new_w = agg.scalar_from_coarse(coarse)
            old_w = float(row.get("sample_weight", 0.0))
            sum_old += old_w
            sum_new += new_w
            row["signal_breakdown_4"] = coarse
            if overwrite_sample_weight:
                row["sample_weight_fine"] = old_w
                row["sample_weight"] = new_w
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    if n:
        logger.info(
            "Rescored %d rows (skipped %d without breakdown); mean weight %.4f → %.4f",
            n, n_skipped, sum_old / n, sum_new / n,
        )
    logger.info("Wrote → %s", out_path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input jsonl with per-row `signal_breakdown` (fine, 15-dim)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output jsonl with coarse 4-scalar + new sample_weight")
    ap.add_argument("--w-correctness", type=float, default=None)
    ap.add_argument("--w-timing", type=float, default=None)
    ap.add_argument("--w-modality", type=float, default=None)
    ap.add_argument("--w-outcome", type=float, default=None)
    ap.add_argument("--keep-old-weight", action="store_true",
                    help="Don't overwrite sample_weight; only add breakdown_4")
    args = ap.parse_args()

    fw = {}
    if args.w_correctness is not None: fw["correctness"] = args.w_correctness
    if args.w_timing      is not None: fw["timing"]      = args.w_timing
    if args.w_modality    is not None: fw["modality"]    = args.w_modality
    if args.w_outcome     is not None: fw["outcome"]     = args.w_outcome

    rescore_file(
        args.in_path, args.out_path,
        family_weights=fw or None,
        overwrite_sample_weight=not args.keep_old_weight,
    )


if __name__ == "__main__":
    main()
