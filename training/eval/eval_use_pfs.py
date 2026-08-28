"""Evaluate post-trained student on Objective A.

Wraps `scripts/run_skill_eval.py` so that we can reuse the existing
inference harness (which already knows how to load a model via the
agent config). For each trained checkpoint we run with PFs-OFF and
PFs-ON and report EM/F1 delta.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def run(ckpt: str, agent_eval_yaml: str, output_dir: str, ablation: str):
    """Invoke scripts/run_skill_eval.py with model overridden to `ckpt`."""
    # Patch agent_eval yaml on the fly
    with open(agent_eval_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["path"] = ckpt
    cfg["experiment"]["output_dir"] = output_dir

    tmp_yaml = Path(output_dir) / f"eval_{ablation}.yaml"
    tmp_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_yaml, "w") as f:
        yaml.dump(cfg, f)

    cmd = [
        "python", "scripts/run_skill_eval.py",
        "--config", str(tmp_yaml),
        "--ablations", ablation,
        "--resume",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained model checkpoint")
    ap.add_argument("--agent-eval-yaml", default="configs/agent_eval.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ablations", nargs="+", default=["baseline", "skills_top10"])
    args = ap.parse_args()

    for ab in args.ablations:
        run(args.ckpt, args.agent_eval_yaml, args.output_dir, ab)


if __name__ == "__main__":
    main()
