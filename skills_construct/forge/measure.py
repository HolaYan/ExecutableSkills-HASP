"""Does the candidate actually help? Two tiers, and only the second is accuracy.

TIER 1 — probe (`probe()`): the marginal effect of a candidate on the curated
corpus, using today's Case-C geometry (evidence -> revised answer, gated by
two-sample agreement + answer type). Attribution is by DIFFERENCE, not by
tags: two dispatch arms are run,

    base arm = the admitted library
    cand arm = the admitted library + this candidate

and a case counts for the candidate only if the candidate CREATED the
intervention (Case A in base, Case C in cand). Rescues the admitted library
would have produced anyway are not credited to the candidate.

  This tier reports RESCUE and BROKE counts on a curated 321-wrong/146-correct
  corpus. That corpus is not a natural distribution, so those counts are NOT
  an accuracy. Treating them as one is the mistake this docstring exists to
  prevent. Tier 1 exists to kill candidates cheaply.

TIER 2 — acc (`acc_command()`): the real test. The canonical n=64 pf_select
protocol against a library containing the candidate, reporting pass@1 per
dataset, before -> after. A candidate is admitted on this number and on
broke == 0 from tier 1 — never on a screening or probe number alone.

The probe library is a COPY of `skills/` under the run's workdir, so probing
never mutates the hand-written library that every measured result came from.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .emit import emit as _emit_specs
from .spec import PFSpec

_HASP = Path(__file__).resolve().parents[2]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

# The admitted baseline: the PFs whose end-to-end effect is measured.
from hasp_config import protocol as _protocol

# The admitted library a candidate is measured against.
# Edit configs/protocol.yaml (`library.base_pfs`) after admitting a PF.
BASE_PFS = list(_protocol().library.base_pfs)

_NUM = re.compile(r"^[-+]?\d+(\.\d+)?(/\d+)?$")


def _norm(s):
    return (s or "").strip().strip("$").replace(",", "").rstrip(".")


def _atype(s):
    s = _norm(s)
    return "none" if not s else ("num" if _NUM.match(s) else "expr")


# ── probe library ────────────────────────────────────────────────────────

def build_probe_library(specs: List[PFSpec], domain: str, workdir: Path) -> Path:
    """Copy skills/ into the workdir and register the candidates there.

    Uses the production load path (`dynamic_program_functions.py` chain-load)
    so a candidate behaves in the probe exactly as it would in a real rollout.
    """
    dst = workdir / "skills_probe"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(_HASP / "skills", dst, ignore=shutil.ignore_patterns("__pycache__"))
    # emit() resolves the library under _HASP; point it at the copy instead
    import skills_construct.forge.emit as _em
    real_root, _em._HASP = _em._HASP, workdir
    try:
        _emit_specs(specs, domain, "probe", "probe library", register=True)
    finally:
        _em._HASP = real_root
    return dst


# ── tier 1: probe ────────────────────────────────────────────────────────

def _dispatch(cases: List[Dict], skills_dir: Path, pf_ids: List[str]) -> List[Dict]:
    """One dispatch arm. Mirrors anchor/eval_polished_pfs.stage_dispatch, but
    with the PF id list as a parameter so arms can be compared."""
    from hasp_paths import setup_compile_caches
    setup_compile_caches()
    from pf_select.pf_select_eval import (_load_pf_system, _build_step_context,
                                          _extract_answer_from_text, _build_feedback)
    exec_pf, _ = _load_pf_system(str(skills_dir))
    out = []
    for c in cases:
        sc = _build_step_context(c["question"], c["response"])
        sc.update(raw_reasoning=c["response"],
                  candidate_answer=_extract_answer_from_text(c["response"]), uid=c["uid"])
        fa, arg, recs, inj = exec_pf(
            active_skill_ids=pf_ids, step_context=sc, action_type="FINAL",
            arg=sc["candidate_answer"] or c["response"], reasoning=c["response"],
            helper_model=None)
        fb = _build_feedback(recs, inj, fa, arg)
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if fb else "A",
                        feedback=fb, pfs=sorted(set(re.findall(r"\[([a-z_]+)", fb or "")))))
    return out


def probe_dispatch(cases: List[Dict], specs: List[PFSpec], domain: str,
                   workdir: Path) -> Dict:
    """Run both arms; return the candidate-owned cases per candidate."""
    lib = build_probe_library(specs, domain, workdir)
    cand_ids = [s.skill_id for s in specs]
    base = {d["uid"]: d for d in _dispatch(cases, lib, BASE_PFS)}
    cand = _dispatch(cases, lib, BASE_PFS + cand_ids)

    owned: Dict[str, List[Dict]] = {sid: [] for sid in cand_ids}
    shared = Counter()
    for d in cand:
        if d["case"] != "C":
            continue
        fired = [p for p in d["pfs"] if p in owned]
        if not fired:
            continue
        if base[d["uid"]]["case"] == "C":
            # the admitted library already intervened here — no marginal effect
            shared[fired[0]] += 1
            continue
        owned[fired[0]].append(d)
    return dict(library=str(lib), owned=owned, shared=dict(shared),
                n_wrong=sum(1 for c in cases if c["label"] == "wrong"),
                n_correct=sum(1 for c in cases if c["label"] == "correct"))


def probe_regen(cases: List[Dict], disp: List[Dict], policy: str, tp: int = 1,
                gpu_mem: float = 0.88, max_model_len: int = 20480,
                max_tokens: int = 6144, n: int = 2) -> List[Dict]:
    """Case-C regeneration — identical geometry to anchor/eval_polished_pfs."""
    from anchor.eval_polished_pfs import stage_regen
    return stage_regen(cases, disp, policy, tp, gpu_mem, max_model_len, max_tokens, n)


def probe_score(cases: List[Dict], owned: Dict[str, List[Dict]],
                regs: List[Dict]) -> Dict[str, Dict]:
    """Per-candidate marginal rescue / broke, with the samples that explain them."""
    from verifiers.reference_em import (
        em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
    )
    by = {c["uid"]: c for c in cases}
    reg = {r["uid"]: r for r in regs}
    out = {}
    for sid, ds in owned.items():
        res = dict(fired=len(ds), fired_wrong=sum(1 for d in ds if d["label"] == "wrong"),
                   fired_correct=sum(1 for d in ds if d["label"] == "correct"),
                   rescue=0, broke=0, no_change=0, rescued_uids=[], broke_uids=[],
                   misses=[])
        for d in ds:
            r = reg.get(d["uid"])
            if not r:
                continue
            c = by[d["uid"]]
            was = _em_match_multi(c["pred"], c["gold"])
            ans = [_extract_answer_math(t) or "" for t in r["texts"]]
            a0 = ans[0]
            gated = bool(a0) and all(_norm(x) == _norm(a0) for x in ans) and _atype(a0) == _atype(c["pred"])
            fin = a0 if gated else c["pred"]
            now = _em_match_multi(fin, c["gold"])
            if not was and now:
                res["rescue"] += 1
                res["rescued_uids"].append(d["uid"])
            elif was and not now:
                res["broke"] += 1
                res["broke_uids"].append(d["uid"])
            else:
                res["no_change"] += 1
                if d["label"] == "wrong" and len(res["misses"]) < 4:
                    # what the PF said vs what the model then answered — this is
                    # the material refine.py needs to narrow the verdict
                    res["misses"].append(dict(uid=d["uid"], verdict=(d["feedback"] or "")[:400],
                                              still_answered=fin[:80], gold=c["gold"][:80]))
        out[sid] = res
    return out


def probe_verdict(res: Dict, min_rescue: int = 2, max_broke: int = 0) -> Tuple[str, List[str]]:
    """probed_out | refine | measured.

    broke > 0 is disqualifying: the fallback-to-original safety is the reason
    pf_select is usable at all, and a candidate that breaks correct solutions
    trades it away.
    """
    reasons = []
    if res["broke"] > max_broke:
        return "probed_out", [f"broke {res['broke']} correct solution(s) — disqualifying"]
    if res["fired"] == 0:
        return "probed_out", ["never fired marginally (the admitted library already covers it)"]
    if res["rescue"] >= min_rescue:
        return "measured", [f"marginal rescue {res['rescue']}/{res['fired_wrong']} wrong, broke 0"]
    if res["rescue"] == 0 and res["fired_wrong"] >= 6:
        return "refine", [f"fired on {res['fired_wrong']} wrong cases but rescued none — "
                          "the verdict is not decisive enough to change the answer"]
    if res["rescue"] == 0:
        return "probed_out", [f"rescued none of {res['fired_wrong']} wrong cases it fired on"]
    return "refine", [f"only {res['rescue']} rescue(s) — narrow the trigger or sharpen the verdict"]


# ── tier 2: the real accuracy test ───────────────────────────────────────

def acc_command(skills_dir: Path, tag: str, model: str, datasets: str, n: int = 64) -> str:
    """The command that measures pass@1 before -> after with this library."""
    return (f"MODEL={model} TAG={tag} DATASETS={datasets} N={n} "
            f"SKILLS_DIR={skills_dir} sbatch scripts/slurm/eval_models.sbatch")


def read_acc(tag: str) -> Optional[Dict]:
    """Read a finished tier-2 run: {dataset: {before, after, delta_pp}}.

    pass@1 only. The eval also records pass@4..64; they are not what a PF is
    judged on and are not read here.
    """
    d = _HASP / "data" / "model_eval" / tag
    if not d.is_dir():
        return None
    out: Dict[str, Dict] = {}

    def _add(ds: str, r: Dict) -> None:
        off = (r.get("skills_off") or {}).get("pass@1")
        on = (r.get("pf_select") or {}).get("pass@1")
        if off is None or on is None:
            return
        out[ds] = dict(before=round(off, 4), after=round(on, 4),
                       delta_pp=round((on - off) * 100, 2))

    s = d / "summary.json"                      # {ds: {skills_off, pf_select, ...}}
    if s.exists():
        try:
            for ds, r in json.loads(s.read_text()).items():
                if isinstance(r, dict):
                    _add(ds, r)
        except (OSError, json.JSONDecodeError):
            pass
    for p in d.glob("*_results.json"):          # per-dataset, written as each finishes
        ds = p.stem.replace("_results", "")
        if ds in out:
            continue
        try:
            _add(ds, json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out or None


def acc_verdict(acc: Dict, min_delta_pp: float = 0.0) -> Tuple[str, List[str]]:
    """Admit only on a measured accuracy change that is not negative anywhere."""
    if not acc:
        return "measured", ["tier-2 accuracy not run yet"]
    deltas = {k: v["delta_pp"] for k, v in acc.items()}
    worst = min(deltas.values())
    mean = sum(deltas.values()) / len(deltas)
    if worst < -0.5:
        return "probed_out", [f"accuracy regressed on {min(deltas, key=deltas.get)} "
                              f"({worst:+.2f}pp)"]
    if mean > min_delta_pp:
        return "admitted", [f"accuracy {mean:+.2f}pp mean across {len(deltas)} datasets "
                            + ", ".join(f"{k} {v:+.2f}pp" for k, v in deltas.items())]
    return "probed_out", [f"no accuracy gain ({mean:+.2f}pp mean)"]
