"""skills_off vs pf_select comparison for an arbitrary model — the canonical
protocol of `Agentic_RL/rl/eval/PF_SELECT_EVAL.md`, reproduced in HASP:

  ReAct prompt, n samples / question at T=0.7 (T=0 if n==1), max_tokens 8192,
  skills_off  = Turn-1 rollouts as-is
  pf_select   = SAME Turn-1 rollouts -> model selects PFs -> A/B/C dispatch
  pass@k by the repo's own extractor (finish[] -> \\boxed -> Answer:) + EM.

Because pf_select reuses the Turn-1s, the two numbers are paired per rollout
and the per-rollout transition table (rescue / broke, split by whether Turn 1
committed an answer) is reported alongside pass@1.

Usage
  python pf_select/eval_models.py --model Qwen/Qwen3-8B --tag q3_8b \\
      --datasets aime24,amc23,olympiadbench --n 64 [--thinking] [--tp 4]
Outputs: data/model_eval/<tag>/<ds>_results.json, summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HASP))
from hasp_paths import setup_compile_caches  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402
_P = _protocol()
setup_compile_caches()   # before torch/vllm are imported

import pandas as pd  # noqa: E402

from pf_select.pf_select_eval import run_inference_pf_select  # noqa: E402
from verifiers.reference_em import (  # noqa: E402
    em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
)

_COMMIT = re.compile(r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:", re.I | re.M)


def pass_at_k(n_correct: int, n: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    from math import comb
    if n - n_correct < k:
        return 1.0
    return 1.0 - comb(n - n_correct, k) / comb(n, k)


def norm_gold(g: str) -> str:
    """Normalize a gold answer so scoring is not defeated by its storage format.

    AMC23's source jsonl stores integer answers as float strings ("27.0").
    `_em_match_multi("27", "27.0")` is False, so an entire dataset scored
    0.0008 pass@1 (2 of 2,560) before this — a data-format artifact that looks
    exactly like a broken model. AIME24 stores clean ints and was unaffected,
    which is what made the bug look dataset-specific rather than systemic.
    Only exact-integer floats are collapsed; "0.5" and LaTeX golds are left
    alone.
    """
    s = str(g).strip()
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    return str(int(f)) if f.is_integer() else s


def score_mode(questions, golds, texts_per_q, ks) -> tuple[dict, list]:
    per_q = []
    agg = {f"pass@{k}": 0.0 for k in ks}
    for q, g, texts in zip(questions, golds, texts_per_q):
        g = norm_gold(g)
        preds = [_extract_answer_math(t) for t in texts]
        oks = [_em_match_multi(p, g) for p in preds]
        nc = sum(oks); n = len(texts)
        for k in ks:
            if k <= n:
                agg[f"pass@{k}"] += pass_at_k(nc, n, k)
        per_q.append(dict(preds=preds, oks=oks, n_correct=nc, n_total=n))
    for k in ks:
        agg[f"pass@{k}"] /= max(1, len(questions))
    return agg, per_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", default=_P.accuracy.datasets)
    ap.add_argument("--n", type=int, default=_P.accuracy.n)
    ap.add_argument("--max-samples", type=int, default=_P.accuracy.max_samples)
    ap.add_argument("--max-tokens", type=int, default=_P.accuracy.max_tokens)
    ap.add_argument("--max-model-len", type=int, default=_P.accuracy.max_model_len)
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel)
    ap.add_argument("--thinking", action="store_true")
    # forge tier-2: point the eval at a probe library containing candidate PFs
    # (default None -> pf_select_eval's own HASP skills/ dir)
    ap.add_argument("--skills-dir", default=os.environ.get("SKILLS_DIR") or None)
    # Name the skills instead of letting the model pick them. This is how a
    # single skill gets a number of its own: the selection turn is skipped, so
    # what the run measures is that skill and nothing else.
    ap.add_argument("--skills", default=os.environ.get("SKILLS") or None,
                    help="comma-separated skill ids; skips the PF selection turn")
    a = ap.parse_args()

    out_dir = _HASP / "data" / "model_eval" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [1, 4, 8, 16, 32, 64]
    summary = {}
    for ds in a.datasets.split(","):
        df = pd.read_parquet(_HASP / "data" / "eval" / f"{ds}.parquet").head(a.max_samples)
        qs, ids = df["question"].tolist(), df["id"].tolist()
        # normalize at load so the saved results carry the same gold that was scored
        golds = [norm_gold(g) for g in df["gold_answer"].astype(str).tolist()]
        temp = 0.0 if a.n == 1 else _P.accuracy.temperature
        finals, t1s, cases = run_inference_pf_select(
            a.model, qs, ["math"] * len(qs), max_tokens=a.max_tokens, temperature=temp,
            n=a.n, max_model_len=a.max_model_len, tp=a.tp, enable_thinking=a.thinking,
            return_t1=True, pf_skill_library_dir=a.skills_dir,
            force_skill_ids=[x.strip() for x in a.skills.split(",")] if a.skills else None,
        )
        if a.n == 1:
            finals, t1s, cases = [[x] for x in finals], [[x] for x in t1s], cases
        off_agg, off_q = score_mode(qs, golds, t1s, ks)
        on_agg, on_q = score_mode(qs, golds, finals, ks)

        # paired transitions
        tr = Counter()
        for oq, sq, tl in zip(off_q, on_q, t1s):
            for o, s, t in zip(oq["oks"], sq["oks"], tl):
                committed = bool(_COMMIT.search(t))
                key = ("rescue" if (not o and s) else "broke" if (o and not s)
                       else "both_ok" if o else "both_bad")
                tr[key] += 1
                if key == "rescue":
                    tr["rescue_committed" if committed else "rescue_uncommitted"] += 1
                tr["committed" if committed else "uncommitted"] += 1
        case_dist = Counter(c for cl in cases for c in cl)
        summary[ds] = dict(total=len(qs), n=a.n, skills_off=off_agg, pf_select=on_agg,
                           delta_pass1=on_agg["pass@1"] - off_agg["pass@1"],
                           transitions=dict(tr), cases=dict(case_dist))
        json.dump(dict(total=len(qs), skills_off=off_agg, pf_select=on_agg,
                       results=[dict(id=i, question=q, gold=g,
                                     off_predictions=oq["preds"], on_predictions=sq["preds"],
                                     off_responses=tl, on_responses=fl, cases=cl)
                                for i, q, g, oq, sq, tl, fl, cl in zip(ids, qs, golds, off_q, on_q, t1s, finals, cases)]),
                  open(out_dir / f"{ds}_results.json", "w"))
        print(f"[eval] {ds}: skills_off pass@1={off_agg['pass@1']:.4f}  pf_select pass@1={on_agg['pass@1']:.4f}  "
              f"Δ={on_agg['pass@1'] - off_agg['pass@1']:+.4f}  transitions={dict(tr)}  cases={dict(case_dist)}", flush=True)
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"[eval] done -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
