"""PF-library blind-spot analysis over mined committed-wrong cases.

Two questions:
  1. Which PFs fire on committed-wrong rollouts and change nothing?
     (Case C feedback ids from the pf_select tail; Case B = format rewrite.)
  2. What is the relation between the wrong answer and the gold? A cheap,
     deterministic taxonomy that tells us which *checkable* error families
     exist and therefore which new PFs / anchor checkers would pay off:
        off_by_one        |pred - gold| == 1
        off_by_small      |pred - gold| <= 5  (but != 1)
        factor_k          pred == k*gold or gold == k*pred, k in 2..10
        sign              pred == -gold
        digit_perm        same multiset of digits
        last_step_slip    gold literally appears in the reasoning (had it, lost it)
        partial_answer    gold is a substring of pred or vice versa (e.g. m vs m+n)
        other
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HASP = Path(__file__).resolve().parents[2]


def _num(s: str):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def taxonomy(pred: str, gold: str, gold_in_reasoning: bool) -> str:
    p, g = _num(pred), _num(gold)
    if gold_in_reasoning:
        return "last_step_slip"
    if p is not None and g is not None:
        if p == -g and g != 0:
            return "sign"
        d = abs(p - g)
        if d == 1:
            return "off_by_one"
        if 0 < d <= 5:
            return "off_by_small"
        for k in range(2, 11):
            if p == k * g or g == k * p:
                return f"factor_{k}"
        if p == int(p) and g == int(g) and sorted(str(int(abs(p)))) == sorted(str(int(abs(g)))):
            return "digit_perm"
    ps, gs = str(pred).strip(), str(gold).strip()
    if ps and gs and (ps in gs or gs in ps):
        return "partial_answer"
    return "other"


def main() -> None:
    case_dir = _HASP / "data" / "error_cases"
    tax = Counter(); tax_ds = defaultdict(Counter)
    case_dist = Counter(); fired_on_wrong = Counter()
    n = 0
    for f in sorted(case_dir.glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line); n += 1
            t = taxonomy(r["wrong_pred"], r["gold"], r["gold_in_reasoning"])
            tax[t] += 1; tax_ds[r["dataset"]][t] += 1
            case_dist[r["pf_case"]] += 1
            for sid in r["pf_feedback_ids"]:
                fired_on_wrong[sid] += 1

    print(f"=== {n} committed-wrong cases ===\n")
    print("PF dispatch on these cases:", dict(case_dist))
    print("  (A = PF did not fire / collapsed; B = format rewrite only; C = generic feedback, no fix)\n")
    print("Case-C feedback PFs that fired and fixed nothing:")
    for sid, c in fired_on_wrong.most_common(12):
        print(f"  {c:>4}  {sid}")
    print("\nError taxonomy (wrong_pred vs gold):")
    for t, c in tax.most_common():
        print(f"  {c:>4} ({c / n:5.1%})  {t}")
    print("\nper dataset:")
    for ds, ct in tax_ds.items():
        top = ", ".join(f"{t}={c}" for t, c in ct.most_common(4))
        print(f"  {ds:<14} {top}")

    checkable = sum(c for t, c in tax.items()
                    if t in ("last_step_slip", "off_by_one", "off_by_small", "sign", "digit_perm")
                    or t.startswith("factor_"))
    print(f"\ncheap-checkable families (slip/off-by/factor/sign/perm): {checkable}/{n} = {checkable / n:.1%}")


if __name__ == "__main__":
    main()
