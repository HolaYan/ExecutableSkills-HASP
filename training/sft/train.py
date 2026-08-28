"""CLI entry point: `python -m training.sft.train --config <yaml>`.

Workflow:
  1. Load experiment yaml
  2. Build training data (invokes data builders if not pre-materialised)
  3. Kick off SFT / DPO via TRL
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from ..signals import SignalAggregator
from ..signals.aggregator import AggregatorConfig
from ..data import UsePFsBuilder, EvolveBuilder
from ..data.use_pfs_builder import UsePFsBuilderConfig
from ..data.evolve_builder import EvolveBuilderConfig
from ..data.signal_filter import resolve_enabled_signals
from .trainer import SFTRunner, DPORunner, TrainerHyperparams


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


# ------------------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------------------

def _load_trajectories(path: str):
    """Load EpisodeTrajectory list from jsonl file(s)."""
    import json
    from training.signals.trajectory import EpisodeTrajectory

    p = Path(path)
    files = sorted(p.rglob("trajectories*.jsonl")) if p.is_dir() else [p]
    trajs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    trajs.append(EpisodeTrajectory.from_dict(json.loads(line)))
    logger.info("Loaded %d trajectories from %s", len(trajs), path)
    return trajs


def _load_candidates_and_reviews(path: str):
    """Load CandidateSkill/ReviewResult lists from a self_improving run.

    File layout (from `self_improving/pipeline.py`):
        <root>/epoch_<N>/proposals/candidate_skills.json   (list[dict])
        <root>/epoch_<N>/reviews/skill_reviews.json        (list[dict])
    """
    import json
    from dataclasses import fields as _fields

    from skills_construct.candidate import CandidateSkill
    from skills_construct.candidate import ReviewResult

    p = Path(path)
    cand_files = list(p.rglob("proposals/candidate_skills.json")) + \
                 list(p.rglob("candidate_skills*.json"))
    rev_files = list(p.rglob("reviews/skill_reviews.json")) + \
                list(p.rglob("skill_reviews*.json"))

    def _load_list(f):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("candidates", "skills", "reviews", "items"):
                if k in data and isinstance(data[k], list):
                    return data[k]
        return []

    def _filter_fields(cls, d):
        allowed = {f.name for f in _fields(cls)}
        return {k: v for k, v in d.items() if k in allowed}

    cands_raw = []
    for cf in sorted(cand_files):
        cands_raw.extend(_load_list(cf))
    revs_raw = []
    for rf in sorted(rev_files):
        revs_raw.extend(_load_list(rf))

    cands = [CandidateSkill(**_filter_fields(CandidateSkill, c)) if isinstance(c, dict) else c
             for c in cands_raw]
    revs = [ReviewResult(**_filter_fields(ReviewResult, r)) if isinstance(r, dict) else r
            for r in revs_raw]
    logger.info("Loaded %d candidates, %d reviews from %s", len(cands), len(revs), path)
    return cands, revs


def _materialize_data(cfg: dict, out_dir: Path) -> Path:
    """Build the JSONL training file if not already present."""
    objective = cfg["objective"]                       # "A" or "B"
    enabled = resolve_enabled_signals(cfg["signals"]["enabled"])
    weights = cfg["signals"].get("weights") or {}
    threshold = cfg["signals"].get("threshold", 0.3)
    fmt = cfg.get("data_format", "sft")                # "sft" | "dpo"

    agg = SignalAggregator(AggregatorConfig(
        enabled=enabled, weights=weights,
        normalize=cfg["signals"].get("normalize", True),
        mode=cfg["signals"].get("mode", "coarse"),
    ))

    if objective == "A":
        b = UsePFsBuilder(UsePFsBuilderConfig(
            output_dir=str(out_dir),
            enabled_signals=enabled,
            signal_weights=weights,
            threshold=threshold,
            formats=[fmt],
            top_k_per_episode=cfg["signals"].get("top_k_per_episode"),
        ), agg)
        trajs = _load_trajectories(cfg["data"]["trajectories_path"])
        outputs = b.build(trajs)
        return outputs[fmt]

    elif objective == "B":
        b = EvolveBuilder(EvolveBuilderConfig(
            output_dir=str(out_dir),
            q_skill_threshold=cfg.get("evolve", {}).get("q_skill_threshold", 0.5),
            lam_val_gain=cfg.get("evolve", {}).get("lam_val_gain", 0.5),
            formats=[fmt],
        ), agg)
        cands, revs = _load_candidates_and_reviews(cfg["data"]["self_improving_dir"])
        outputs = b.build(cands, revs)
        return outputs[fmt]

    raise ValueError(f"Unknown objective: {objective}")


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment"]["id"]
    base_out = Path(cfg["experiment"]["output_dir"]) / exp_id
    data_out = base_out / "data"
    ckpt_out = base_out / "ckpt"
    data_out.mkdir(parents=True, exist_ok=True)
    ckpt_out.mkdir(parents=True, exist_ok=True)

    logger.info("Experiment %s — objective %s — signals %s",
                exp_id, cfg["objective"], cfg["signals"]["enabled"])

    data_file = _materialize_data(cfg, data_out)

    t = cfg["trainer"]
    hp = TrainerHyperparams(
        model_path=cfg["model"]["path"],
        output_dir=str(ckpt_out),
        data_path=str(data_file),
        num_train_epochs=t.get("num_train_epochs", 3.0),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 1e-5),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        max_seq_length=t.get("max_seq_length", 4096),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        save_every_n_epochs=t.get("save_every_n_epochs"),
        deepspeed=t.get("deepspeed"),
        use_lora=t.get("use_lora", False),
        lora_r=t.get("lora_r", 16),
        lora_alpha=t.get("lora_alpha", 32),
        report_to=t.get("report_to", "wandb"),
        run_name=exp_id,
        seed=cfg["experiment"].get("seed", 42),
    )

    method = cfg.get("method", "sft")   # "sft" or "dpo"
    if method == "dpo":
        runner = DPORunner(hp, beta=cfg["trainer"].get("dpo_beta", 0.1))
    else:
        runner = SFTRunner(hp)
    runner.run()


if __name__ == "__main__":
    main()
