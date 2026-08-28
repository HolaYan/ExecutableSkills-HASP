"""Knobs for in-training evolution.

The defaults are deliberately conservative. Evolution runs *inside* a training
job: every cycle costs GPU time the trainer is not using for training, and
every PF it admits changes the rollout distribution of everything that comes
after. A cycle that is too frequent, too large, or too permissive does not
produce a better library — it produces a training run whose results cannot be
attributed to anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvolveConfig:
    enabled: bool = False

    # ── when to pause ──
    every_steps: int = 200          # optimizer steps between evolve cycles
    skip_first: int = 200           # no cycle before this step (early ckpts fail for
                                    # reasons that are about training, not about skills)
    max_generations: int = 8        # hard cap on cycles per run

    # ── the eval at the pause ──
    eval_size: int = 48             # held-out questions per cycle
    eval_max_new_tokens: int = 1024
    eval_batch_size: int = 8
    eval_dataset: str = ""          # parquet under data/eval/; default = the domain's

    # ── distilling failures into PFs ──
    proposer: str = "self"          # "self" = the model being trained (the only one
                                    # on the GPU mid-run); or a model path if you have
                                    # the memory for a second engine
    families_per_cycle: int = 3
    candidates_per_family: int = 2
    max_admit_per_cycle: int = 2    # cap on library growth per generation
    propose_max_tokens: int = 3072

    # ── gates (mid-training uses the CHEAP ones only) ──
    # The probe and the n=64 accuracy test are far too expensive to run inside a
    # training loop; what runs here is the structural gate plus the offline
    # precision screen against the correct-set control from this same eval.
    # Anything admitted here is provisional and must still face the full
    # forge gates before it is treated as measured.
    min_fire_wrong: float = 0.05
    max_fire_correct: float = 0.05
    min_lift: float = 2.0

    # ── reviewing what is already in the library ──
    # Each cycle scores the skills that fired during its own evaluation with the
    # four credit signals (timing / modality / correctness / outcome). Costs
    # nothing extra — the rollouts already exist — and it is the only part of
    # the loop that looks backwards at what the library already contains.
    review_enabled: bool = True
    review_min_fires: int = 3      # below this a skill fired too rarely to judge

    # ── bookkeeping ──
    library_dir: str = ""           # run-scoped library (auto: {output_dir}/library)
    domain: str = "math"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "EvolveConfig":
        """Read the `evolve:` block of a training config."""
        blk = dict(cfg.get("evolve") or {})
        blk.setdefault("domain", cfg.get("domain", "math"))
        known = EvolveConfig.__dataclass_fields__
        return EvolveConfig(**{k: v for k, v in blk.items() if k in known})
