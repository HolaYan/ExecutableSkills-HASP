"""Offline precision/recall of the answer-drift anchor.

Error set (committed-wrong cases):
  drift_found         a drift anchor exists
  drift_earlier=gold  the abandoned earlier claim IS the gold  -> recoverable
Control set (committed-correct rollouts):
  drift_found         the rollout also drifted (but ended correct). These are
                      the risky ones: an anchored regeneration could steer back
                      to the wrong earlier value. Measures the broke-risk floor.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HASP))
from anchor.anchor import check_answer_drift, _norm_claim  # noqa: E402

from hasp_paths import rollouts_dir  # noqa: E402
# Resolved lazily: only the corpus-building stage reads the raw rollouts,
# so scoring an existing corpus needs no upstream checkout at all.
from verifiers.reference_em import em_match_multi as _em_match_multi  # noqa: E402

_COMMIT = re.compile(r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:", re.I | re.M)
DS = ["aime24", "amc23", "math500", "gsm8k", "olympiadbench"]


def main() -> None:
    print("=== error set (committed-wrong) ===")
    tot = Counter(); ex = []
    for f in sorted((_HASP / "data/error_cases").glob("*.jsonl")):
        c = Counter()
        for line in f.open():
            r = json.loads(line); c["n"] += 1
            ev = check_answer_drift(r["orig_response"], r["wrong_pred"])
            if ev is None:
                continue
            c["drift"] += 1
            x = ev.claim.split(" -> ")[0]
            if _norm_claim(x) == _norm_claim(r["gold"]):
                c["earlier_is_gold"] += 1
                if len(ex) < 6:
                    ex.append((r["dataset"], r["qid"], ev.claim, r["gold"], f"{ev.span[0] / len(r['orig_response']):.0%}"))
            if r["gold_in_reasoning"]:
                c["drift_on_gir"] += 1
        tot.update(c)
        print(f"  {f.stem:<14} n={c['n']:>4}  drift={c['drift']:>3} ({c['drift'] / max(1, c['n']):5.1%})  "
              f"earlier==gold={c['earlier_is_gold']:>3}")
    print(f"  TOTAL          n={tot['n']:>4}  drift={tot['drift']:>3} ({tot['drift'] / max(1, tot['n']):5.1%})  "
          f"earlier==gold={tot['earlier_is_gold']:>3} "
          f"(= {tot['earlier_is_gold'] / max(1, tot['n']):.1%} of all committed-wrong cases are "
          f"recoverable by drift anchoring alone)")
    print("\n  examples (abandoned -> final | gold | drift position):")
    for ds, q, claim, g, pos in ex:
        print(f"    [{ds}/{q}] {claim} | gold={g} | @{pos}")

    print("\n=== control set (committed-correct, cap 150/dataset) ===")
    n = drift = 0
    for ds in DS:
        off = json.loads((rollouts_dir() / "skills_off" / f"{ds}_results.json").read_text())["results"]
        k = 0
        for q in off:
            for p, r in zip(q["all_predictions"], q["all_responses"]):
                if k >= 150:
                    break
                if not _COMMIT.search(r) or not _em_match_multi(p, str(q["gold"])):
                    continue
                k += 1; n += 1
                if check_answer_drift(r, p) is not None:
                    drift += 1
            if k >= 150:
                break
    print(f"  drift on correct rollouts: {drift}/{n} = {drift / max(1, n):.1%}  "
          f"(broke-risk floor; fallback only protects against unparseable output, "
          f"not against a confident wrong regeneration)")


if __name__ == "__main__":
    main()
