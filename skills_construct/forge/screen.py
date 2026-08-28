"""Stage 3 — THE GATE. Everything else in this module exists to feed it.

A candidate PF is run over the wrong set and the correct set and judged on
three numbers:

    fire_wrong    does the family have any population at all?
    fire_correct  the false-positive floor: every point here is a working
                  rollout the PF would interrupt
    lift          fire_wrong / fire_correct — whether the checker separates
                  the two sets at all, or just fires on everything

The thresholds below are constants, not laws. Calibrate them on skills whose
end-to-end effect you have measured — the shape to look for is a candidate that
fires far more often on failures than on solutions that were already correct,
with a low absolute rate on the latter. A candidate that fires on both equally
interrupts working reasoning for nothing.
`lift >= 2.0` is the line that separates every PF that worked from every PF
that did not. It is a necessary condition, not a sufficient one: passing the
gate earns a candidate an end-to-end regeneration run, nothing more.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .spec import PFSpec

_HASP = Path(__file__).resolve().parents[2]
_RUNNER = Path(__file__).resolve().parent / "_screen_runner.py"

# ── calibrated thresholds (see module docstring) ──
# The gate. Values live in configs/protocol.yaml (`screen:`) — they are read off
# the boundary between what worked and what was falsified, not chosen by taste.
from hasp_config import protocol as _protocol

_S = _protocol().screen
MIN_FIRE_WRONG = _S.min_fire_wrong      # below this the family has no population
MAX_FIRE_CORRECT = _S.max_fire_correct  # compute_observation_verify sat at 4.9% and held
MIN_LIFT = _S.min_lift                  # the survived/falsified boundary
MAX_ERROR_RATE = _S.max_error_rate      # a checker that crashes on real inputs is not done


@dataclass
class ScreenResult:
    skill_id: str
    n_wrong: int
    n_correct: int
    fire_wrong: float
    fire_correct: float
    lift: float
    error_rate: float
    repairs: int
    verdict: str               # accept | reject
    reasons: List[str]
    samples: List[Dict]        # a few fired verdicts, for human review

    def to_json(self) -> Dict:
        return self.__dict__ | {"samples": self.samples[:5]}

    def line(self) -> str:
        mark = "ACCEPT" if self.verdict == "accept" else "reject"
        return (f"  [{mark}] {self.skill_id:<34} "
                f"wrong {self.fire_wrong:>6.1%}  correct {self.fire_correct:>6.1%}  "
                f"lift {self.lift:>5.2f}"
                + ("   " + "; ".join(self.reasons) if self.reasons else ""))


def _run_candidate(spec: PFSpec, corpus_path: Path, cpu_s: int, wall_s: int,
                   workdir: Path) -> Optional[List[Dict]]:
    """Run one checker over the corpus in a contained subprocess."""
    ck = workdir / f"{spec.skill_id}_checker.py"
    ck.write_text(spec.checker_src)
    out = workdir / f"{spec.skill_id}_fires.jsonl"
    cmd = [sys.executable, str(_RUNNER), "--checker", str(ck), "--kind", spec.checker_kind,
           "--corpus", str(corpus_path), "--out", str(out), "--cpu-s", str(cpu_s)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=wall_s, cwd=str(_HASP))
    except subprocess.TimeoutExpired:
        return None
    if not out.exists():
        return None
    recs = [json.loads(l) for l in out.open() if l.strip()]
    # A checker killed by RLIMIT_CPU leaves a partial file: treat as unsafe.
    if p.returncode != 0 and len(recs) == 0:
        return None
    return recs


def screen_one(spec: PFSpec, cases: List[Dict], corpus_path: Path, workdir: Path,
               cpu_s: int = 180, wall_s: int = 300) -> ScreenResult:
    n_wrong = sum(1 for c in cases if c["label"] == "wrong")
    n_correct = sum(1 for c in cases if c["label"] == "correct")
    recs = _run_candidate(spec, corpus_path, cpu_s, wall_s, workdir)

    if recs is None:
        return ScreenResult(spec.skill_id, n_wrong, n_correct, 0.0, 0.0, 0.0, 1.0, 0,
                            "reject", ["checker hung or exceeded its CPU/memory limit"], [])
    if len(recs) < 0.9 * len(cases):
        return ScreenResult(spec.skill_id, n_wrong, n_correct, 0.0, 0.0, 0.0, 1.0, 0,
                            "reject", [f"checker died after {len(recs)}/{len(cases)} cases"], [])

    fw = sum(1 for r in recs if r["label"] == "wrong" and r["fired"])
    fc = sum(1 for r in recs if r["label"] == "correct" and r["fired"])
    errs = sum(1 for r in recs if r["err"])
    fire_wrong = fw / max(n_wrong, 1)
    fire_correct = fc / max(n_correct, 1)
    lift = fire_wrong / max(fire_correct, 1e-6)
    err_rate = errs / max(len(recs), 1)
    repairs = sum(1 for r in recs if r.get("has_fix"))

    reasons: List[str] = []
    if fire_wrong < MIN_FIRE_WRONG:
        reasons.append(f"no population (fires on {fire_wrong:.1%} of wrong, need {MIN_FIRE_WRONG:.0%})")
    if fire_correct > MAX_FIRE_CORRECT:
        reasons.append(f"false-positive floor too high ({fire_correct:.1%} > {MAX_FIRE_CORRECT:.0%})")
    if lift < MIN_LIFT:
        reasons.append(f"does not separate wrong from correct (lift {lift:.2f} < {MIN_LIFT})")
    if err_rate > MAX_ERROR_RATE:
        ex = next((r["err"] for r in recs if r["err"]), "")
        reasons.append(f"crashes on {err_rate:.1%} of inputs ({ex})")

    samples = [dict(uid=r["uid"], label=r["label"], verdict=r["verdict"])
               for r in recs if r["fired"]][:8]
    return ScreenResult(spec.skill_id, n_wrong, n_correct, round(fire_wrong, 4),
                        round(fire_correct, 4), round(lift, 2), round(err_rate, 4), repairs,
                        "reject" if reasons else "accept", reasons, samples)


def screen_all(specs: List[PFSpec], cases: List[Dict], workdir: Path,
               cpu_s: int = 180, wall_s: int = 300) -> List[ScreenResult]:
    workdir.mkdir(parents=True, exist_ok=True)
    corpus_path = workdir / "corpus.jsonl"
    with corpus_path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    results = []
    for s in specs:
        r = screen_one(s, cases, corpus_path, workdir, cpu_s, wall_s)
        s.screen = r.to_json()
        results.append(r)
        print(r.line(), flush=True)
    return results
