"""Run one evolve cycle against a checkpoint, outside a training loop.

Useful for two things the callback cannot do: trying the cycle on a checkpoint
before committing a training run to it, and re-running a generation whose
proposals you want to inspect.

    python -m evolving.run --ckpt <path> --domain math --generation 1
    python -m evolving.run --history --library training/outputs/H1/library
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from evolving.callback import run_cycle          # noqa: E402
from evolving.config import EvolveConfig         # noqa: E402
from evolving.library import EvolvingLibrary     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="model to evaluate and to propose with")
    ap.add_argument("--domain", default="math", choices=["math", "code", "web"])
    ap.add_argument("--library", default=None, help="run-scoped library dir")
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--eval-size", type=int, default=48)
    ap.add_argument("--eval-dataset", default="")
    ap.add_argument("--max-admit", type=int, default=2)
    ap.add_argument("--history", action="store_true", help="print the library's history and exit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lib_root = Path(a.library or (_HASP / "data" / "evolving" / f"{a.domain}_library"))

    if a.history:
        lib = EvolvingLibrary(lib_root, a.domain)
        print(lib.summary())
        for r in lib.history():
            if r.get("event") != "generation":
                continue
            ids = ", ".join(x["skill_id"] for x in r.get("admitted", [])) or "(none)"
            p1 = f"{r['pass1']:.4f}" if r.get("pass1") is not None else "  —   "
            print(f"  gen {r['generation']:<3} step {r['step']:<7} pass@1 {p1}  "
                  f"failures {r.get('n_failures','?'):<4} admitted: {ids}")
        return

    if not a.ckpt:
        raise SystemExit("--ckpt is required unless --history is given")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    cfg = EvolveConfig(enabled=True, domain=a.domain, eval_size=a.eval_size,
                       eval_dataset=a.eval_dataset, max_admit_per_cycle=a.max_admit,
                       library_dir=str(lib_root))
    lib = EvolvingLibrary.create(lib_root, a.domain)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16,
                                                 device_map="auto")
    rec = run_cycle(model, tok, cfg, lib, a.generation, a.step,
                    Path(a.library or lib_root).parent / "evolve_work")
    print(json.dumps(rec, indent=2))
    print(lib.summary())


if __name__ == "__main__":
    main()
