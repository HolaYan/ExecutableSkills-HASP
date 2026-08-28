"""Subprocess worker for screening one candidate checker. Do not import.

Runs with the repo on sys.path ON PURPOSE: it uses production's own step
splitter, so a checker's screened fire rate is the rate it will have at
inference. Fidelity matters more than isolation here — the code comes from
our own proposer model on our own prompts, and it is screened before it is
ever registered.

Containment is still real: CPU/address-space rlimits, no network, and results
are flushed per case so that a checker which hangs or explodes still leaves a
usable partial record (the parent then rejects it as unsafe/slow).
"""
from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HASP))


def _lock_down(cpu_s: int, mem_gb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (mem_gb << 30, mem_gb << 30))

    def _no(*a, **k):
        raise OSError("network disabled")

    socket.socket = _no


def _load_steps():
    """Production's step splitter (skills/executable/math/evidence_pfs.py::_steps)."""
    import importlib.util
    p = _HASP / "skills" / "math" / "evidence_pfs.py"
    spec = importlib.util.spec_from_file_location("_forge_evidence_pfs", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checker", required=True)   # .py defining def check(...)
    ap.add_argument("--kind", required=True, choices=["step", "answer"])
    ap.add_argument("--corpus", required=True)    # .jsonl
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-s", type=int, default=120)
    ap.add_argument("--mem-gb", type=int, default=2)
    a = ap.parse_args()

    _lock_down(a.cpu_s, a.mem_gb)

    ns: dict = {}
    exec(compile(Path(a.checker).read_text(), "<candidate>", "exec"), ns)
    check = ns["check"]
    steps = _load_steps() if a.kind == "step" else None

    with open(a.out, "w") as fo:
        for line in open(a.corpus):
            c = json.loads(line)
            text = c.get("response") or ""
            rec = dict(uid=c["uid"], label=c["label"], fired=False, verdict="", err="")
            try:
                if a.kind == "step":
                    for st in steps(text):
                        r = check(st.text, text, st.char_start)
                        if r and r.get("verdict"):
                            rec.update(fired=True, verdict=str(r["verdict"])[:300],
                                       step_idx=st.idx, has_fix=bool(r.get("fix")))
                            break
                else:
                    # ctx mirrors the inference-time step_context: model-written
                    # fields only. No gold, ever.
                    ctx = {"question": c.get("question", ""), "uid": c["uid"],
                           "entry_point": c.get("entry_point", ""),
                           "public_test_code": c.get("public_test_code", "")}
                    v = check(text, c.get("pred", ""), ctx)
                    if v:
                        rec.update(fired=True, verdict=str(v)[:300])
            except Exception as e:  # a crashing checker is a rejected checker
                rec["err"] = f"{type(e).__name__}: {e}"[:200]
            fo.write(json.dumps(rec) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
