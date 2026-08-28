"""Stage 1 — group wrong cases into candidate PF families.

Two independent signals decide whether a family is worth proposing a PF for:

  population   how many wrong cases fall in the family (the six falsified
               "more skills" PFs all had ~zero population — that alone would
               have killed them before a single GPU-hour was spent);
  claim surface  what CHECKABLE artifact the model itself wrote in those
               responses. Every PF that ever produced rescues anchors on such
               an artifact: a self-written `compute[...]` Observation, a
               `\\boxed{}` commit, a spec `>>>` example, an enumeration. A
               family with population but no claim surface has nothing for a
               deterministic checker to attach to, and belongs to the helper
               path instead.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

_HASP = Path(__file__).resolve().parents[2]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills_construct.mining.analyze_blindspots import taxonomy  # noqa: E402

# What the model wrote that a checker could re-verify, and the PF family that
# already exploits it (so the proposer is told what is taken).
CLAIM_SURFACES = {
    "self_computed_observation": (re.compile(r"compute\s*\[", re.I), "arithmetic_slip"),
    "boxed_commit":              (re.compile(r"\\boxed\s*\{"), "boundary_violation"),
    "explicit_equation":         (re.compile(r"(?m)^[^=\n]{1,60}=[^=\n]{1,60}$"), None),
    "enumeration":               (re.compile(r"\b(?:case|for each|we (?:list|enumerate)|there are)\b.*\b\d+\b", re.I), "counting_small_case_check"),
    "claimed_uniqueness":        (re.compile(r"\b(?:the only|unique|exactly one)\b", re.I), "claimed_unique_solution_search"),
    "interval_sign":             (re.compile(r"[<>]\s*0\b|\bpositive on\b|\bnegative on\b", re.I), "interval_sign_check"),
    "hedged_commit":             (re.compile(r"\b(?:guess|probably|I think|approximately|roughly)\b", re.I), "unsupported_final_answer"),
    "spec_example":              (re.compile(r">>>|\bassert\b"), "spec_example_check"),
    "traceback_prone_api":       (re.compile(r"\.\w+\(|\bimport\b"), "exception_contract_check"),
}


def _surfaces(text: str) -> List[str]:
    return [k for k, (rx, _) in CLAIM_SURFACES.items() if rx.search(text or "")]


def cluster(cases: List[Dict], min_population: int = 8) -> List[Dict]:
    """-> family candidates, most-promising first."""
    by_fam: Dict[str, List[Dict]] = defaultdict(list)
    correct_surf = Counter()

    for c in cases:
        if c["label"] == "correct":
            correct_surf.update(_surfaces(c["response"]))
            continue
        gold_in = bool(c["gold"]) and c["gold"] in (c["response"] or "")
        fam = taxonomy(c["pred"], c["gold"], gold_in) if c["gold"] else "code_failure"
        by_fam[fam].append(c)

    n_correct = sum(1 for c in cases if c["label"] == "correct") or 1
    n_wrong = sum(1 for c in cases if c["label"] == "wrong") or 1

    out = []
    for fam, members in by_fam.items():
        if len(members) < min_population:
            continue
        surf = Counter()
        for m in members:
            surf.update(_surfaces(m["response"]))
        # A surface is only informative if it is more common among wrong
        # responses than among correct ones — the same lift test the screening
        # gate applies later, computed cheaply here on presence alone.
        ranked = []
        for s, k in surf.most_common():
            w_rate, c_rate = k / len(members), correct_surf[s] / n_correct
            ranked.append(dict(surface=s, wrong_rate=round(w_rate, 3),
                               correct_rate=round(c_rate, 3),
                               lift=round(w_rate / max(c_rate, 1e-6), 2),
                               taken_by=CLAIM_SURFACES[s][1]))
        out.append(dict(
            family=fam, n_wrong=len(members),
            share_of_wrong=round(len(members) / n_wrong, 3),
            claim_surfaces=ranked[:6],
            best_lift=max([r["lift"] for r in ranked], default=0.0),
            examples=[m["uid"] for m in members[:12]],
        ))

    out.sort(key=lambda d: (d["best_lift"] >= 1.5, d["n_wrong"]), reverse=True)
    return out


def render(families: List[Dict]) -> str:
    lines = []
    for f in families:
        lines.append(f"{f['family']:<18} n_wrong={f['n_wrong']:<4} "
                     f"({f['share_of_wrong']:.0%} of wrong)  best_lift={f['best_lift']}")
        for r in f["claim_surfaces"][:3]:
            taken = f"  [taken by {r['taken_by']}]" if r["taken_by"] else "  [FREE]"
            lines.append(f"    {r['surface']:<26} wrong {r['wrong_rate']:.1%} / "
                         f"correct {r['correct_rate']:.1%}  lift {r['lift']}{taken}")
    return "\n".join(lines)
