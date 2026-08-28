"""Offline evaluation of anchor v2 on mined committed-wrong error cases.

No GPU needed. Measures, per dataset and overall:

  - coverage: fraction of error cases where a deterministic checker produces
    a concrete anchor (evidence-backed step-level failure);
  - anchor position: where in the trajectory the anchor lands (early anchors
    are the interesting ones — an anchor at 95% of the text degenerates to
    end-of-text feedback);
  - gold-recoverable coverage: coverage restricted to the gold_in_reasoning
    subset, plus whether the anchor lands BEFORE the last literal occurrence
    of the gold value (if it does, the regeneration prefix still contains the
    correct value the model once had — the best possible starting point);
  - checker attribution and example verdicts for eyeballing.

Usage:
    python anchor/eval_offline.py [--cases data/error_cases] [--examples 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from anchor.anchor import locate_anchor  # noqa: E402


def _last_gold_pos(text: str, gold: str) -> int:
    g = str(gold).strip()
    return text.rfind(g) if g else -1


def control_set_rate(per_dataset_cap: int = 120) -> tuple[int, int, Counter]:
    """Anchor rate on committed-CORRECT rollouts.

    A correct rollout should (mostly) contain no concrete step errors, so the
    anchor rate here estimates the checkers' false-positive rate. Not exact —
    correct rollouts can contain self-corrected intermediate slips, which are
    genuine anchors — but a small gap vs. the error-set coverage would mean
    the checkers carry no signal.
    """
    import re as _re
    from hasp_paths import rollouts_dir
    # only this stage reads the raw rollouts
    from verifiers.reference_em import em_match_multi as _em_match_multi
    commit_re = _re.compile(
        r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:",
        _re.IGNORECASE | _re.MULTILINE)
    n = hits = 0
    chk = Counter()
    for ds in ["aime24", "amc23", "math500", "gsm8k", "olympiadbench"]:
        off = json.loads((rollouts_dir() / "skills_off" / f"{ds}_results.json").read_text())["results"]
        kept = 0
        for q in off:
            for p, r in zip(q["all_predictions"], q["all_responses"]):
                if kept >= per_dataset_cap:
                    break
                if not commit_re.search(r) or not _em_match_multi(p, str(q["gold"])):
                    continue
                kept += 1; n += 1
                res = locate_anchor(r)
                if res.anchored:
                    hits += 1
                    chk[res.evidence.checker] += 1
            if kept >= per_dataset_cap:
                break
    return hits, n, chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(_HASP / "data" / "error_cases"))
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--control", action="store_true",
                    help="also measure anchor rate on committed-correct rollouts")
    args = ap.parse_args()

    case_dir = Path(args.cases)
    rows = []
    grand = Counter()
    checker_hits = Counter()
    examples = []

    for f in sorted(case_dir.glob("*.jsonl")):
        ds = f.stem
        n = anch = early = gir_n = gir_anch = gir_before_gold = 0
        pos_fracs = []
        for line in f.open():
            r = json.loads(line)
            n += 1
            res = locate_anchor(r["orig_response"])
            gir = bool(r.get("gold_in_reasoning"))
            gir_n += gir
            if not res.anchored:
                continue
            anch += 1
            checker_hits[res.evidence.checker] += 1
            frac = res.truncate_at / max(1, len(r["orig_response"]))
            pos_fracs.append(frac)
            if frac < 0.8:
                early += 1
            if gir:
                gir_anch += 1
                lg = _last_gold_pos(r["orig_response"], r["gold"])
                if lg >= 0 and res.truncate_at > lg:
                    gir_before_gold += 1  # gold survives in the prefix
            if len(examples) < args.examples * 5 and res.evidence:
                examples.append((ds, r["qid"], res.evidence.checker,
                                 res.evidence.verdict[:140], f"{frac:.0%}"))
        med = sorted(pos_fracs)[len(pos_fracs) // 2] if pos_fracs else float("nan")
        rows.append((ds, n, anch, f"{anch / max(1, n):.1%}", early,
                     f"{med:.0%}" if pos_fracs else "-",
                     gir_n, gir_anch, gir_before_gold))
        grand.update(dict(n=n, anch=anch, early=early, gir_n=gir_n,
                          gir_anch=gir_anch, gir_bg=gir_before_gold))

    hdr = ["dataset", "cases", "anchored", "cov", "anchor<80%", "med_pos",
           "gold_in_r", "gir_anch", "gold_in_prefix"]
    print("  ".join(f"{h:>13}" for h in hdr))
    for r in rows:
        print("  ".join(f"{v:>13}" for v in r))
    g = grand
    print(f"\nTOTAL: coverage {g['anch']}/{g['n']} = {g['anch'] / max(1, g['n']):.1%}   "
          f"gold_in_reasoning subset: {g['gir_anch']}/{g['gir_n']} anchored, "
          f"{g['gir_bg']} keep the gold value inside the regeneration prefix")
    print("\nchecker attribution:", dict(checker_hits))
    print("\n=== sample verdicts ===")
    for ds, qid, chk, verdict, pos in examples[: args.examples * 5]:
        print(f"  [{ds}/{qid}] ({chk} @ {pos}) {verdict}")

    if args.control:
        hits, n, chk = control_set_rate()
        print(f"\n=== control set (committed-CORRECT rollouts) ===")
        print(f"anchored {hits}/{n} = {hits / max(1, n):.1%}  "
              f"(upper bound on false-positive rate)  {dict(chk)}")


if __name__ == "__main__":
    main()
