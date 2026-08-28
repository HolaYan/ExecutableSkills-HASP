"""Load test data and split into 6 subsets (3 datasets × 2 variants).

Test data lives at `data/code/{humaneval_plus,mbpp_plus,bigcodebench}.jsonl`.
The first 100 rows of each file form the testset (50 problems × 2 variants).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

DATASETS = ["humaneval_plus", "mbpp_plus", "bigcodebench"]
# data/code/_eval_plus_split_stats.json: test_entries=200 per file (100 unique
# problems × 2 variants). Older code_eval.yaml comments saying "100 entries"
# are stale — the project's own configs use num_samples=200.
TESTSET_SIZE = 200


def load_testset(data_dir: str | Path, dataset: str) -> List[dict]:
    path = Path(data_dir) / f"{dataset}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:TESTSET_SIZE]


def split_by_variant(rows: List[dict]) -> Dict[str, List[dict]]:
    by_variant: Dict[str, List[dict]] = {}
    for r in rows:
        v = r.get("variant", "default")
        by_variant.setdefault(v, []).append(r)
    return by_variant


def load_all_subsets(data_dir: str | Path) -> Dict[str, Dict[str, List[dict]]]:
    """Return {dataset: {variant: [row, ...]}} for all 3 datasets."""
    out: Dict[str, Dict[str, List[dict]]] = {}
    for ds in DATASETS:
        rows = load_testset(data_dir, ds)
        out[ds] = split_by_variant(rows)
    return out


def iter_subsets(data_dir: str | Path) -> Iterable[tuple[str, str, List[dict]]]:
    """Yield (dataset, variant, rows) tuples — exactly 6 subsets."""
    for ds, variants in load_all_subsets(data_dir).items():
        for v, rows in sorted(variants.items()):
            yield ds, v, rows
