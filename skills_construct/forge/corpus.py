"""The screening corpus: model-written artifacts labelled wrong / correct.

Screening a candidate PF needs BOTH sets. The wrong set says whether the
family has any population at all; the correct set is the false-positive floor,
and it is the set that killed most candidates — a checker that fires on
correct solutions turns a working rollout into a broken one.

Sources (all full-text; never read questions from `*_results.json`, which
truncates them to 100 chars — see the data-bug note in the README):
  math : data/llm_anchor/cases.jsonl        321 wrong / 146 correct
  code : anchor/eval_code_polished.build_cases()  failing / passing solutions
  web  : on hold until SerpAPI quota returns
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

_HASP = Path(__file__).resolve().parents[2]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))


def load_math(cases_path: Path | None = None) -> List[Dict]:
    """-> [{uid, label: wrong|correct, question, response, pred, gold, dataset}]

    `gold` is carried for CLUSTERING and REPORTING only. `screen.py` never
    passes it to a checker; `spec.validate_spec` rejects checkers that mention
    it. Keeping it here is what lets cluster.py name error families.
    """
    from hasp_paths import data_dir
    p = Path(cases_path) if cases_path else data_dir() / "llm_anchor" / "cases.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"math corpus not found: {p}\n"
            "  Build it first:  python anchor/llm_locate.py --stage build\n"
            "  (that stage reads the raw rollouts — see 'Upstream corpora' in README.md)"
        )
    out = []
    for line in p.open():
        r = json.loads(line)
        out.append(dict(
            uid=r["uid"], label=r["label"], dataset=r.get("dataset", ""),
            question=r.get("question", ""), response=r.get("response", ""),
            pred=str(r.get("pred", "")), gold=str(r.get("gold", "")),
        ))
    return out


def load_code(per_arm_cap: int = 400) -> List[Dict]:
    """Failing (label=wrong) / passing (label=correct) code solutions.

    `response` is the model's completion, `question` the spec — both are
    model- or spec-written text, so checkers screened here see exactly what
    they would see at inference.
    """
    from anchor.eval_code_polished import build_cases
    out = []
    for c in build_cases(per_arm_cap):
        out.append(dict(
            uid=c.get("uid") or f"{c.get('ds','code')}_{c.get('qid','')}_{c.get('arm','')}",
            label="wrong" if not c.get("passed") else "correct",
            dataset=c.get("ds", "code"), question=c.get("question", ""),
            response=c.get("final") or c.get("response", ""),
            pred=c.get("final", ""), gold="",
            entry_point=c.get("entry_point", ""), public_test_code=c.get("public_test_code", ""),
        ))
    return out


def load(domain: str, **kw) -> List[Dict]:
    if domain == "math":
        return load_math(kw.get("cases_path"))
    if domain == "code":
        return load_code(kw.get("per_arm_cap", 400))
    raise SystemExit(
        f"no screening corpus for domain {domain!r}. "
        "web is on hold until SerpAPI quota returns — a PF that cannot be "
        "screened against a correct-set control must not be forged."
    )


def summary(cases: List[Dict]) -> str:
    w = sum(1 for c in cases if c["label"] == "wrong")
    return f"{len(cases)} cases: {w} wrong / {len(cases) - w} correct"
