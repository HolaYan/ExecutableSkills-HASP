"""One library-evolution step, driven by the current student checkpoint.

Wraps `self_improving.pipeline.SelfImprovingPipeline` so we can run it for a
single epoch against a student model at a user-supplied checkpoint path.

Notes on wiring:
  * `SelfImprovingPipeline._init_models` treats `student_model` as a HF model
    path whenever it doesn't match a key in `api_models` — so passing the
    local ckpt path directly is the canonical way to get a vLLM student.
  * Library root is scoped via top-level `seed_skill_dir` /
    `generated_skill_dir` (not a single `library.library_dir`), so the
    experiment-local library persists across iterations.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def run_evolve_step(
    base_config_path: str,
    student_ckpt_path: str,
    iter_output_dir: str,
    library_dir: str,
    num_epochs: int = 1,
    seed_samples: Optional[int] = None,
    prefilter_baseline_failures: bool = False,
    prefilter_cap_k: int = 0,
) -> dict:
    """Run one self-improving epoch against the given student checkpoint."""
    from self_improving.configs import load_config_from_yaml
    from self_improving.pipeline import SelfImprovingPipeline

    cfg = copy.deepcopy(load_config_from_yaml(base_config_path))

    # Student = local checkpoint. `student_model` is treated as a HF path
    # because it does not match any `api_models` key.
    cfg.student_model = student_ckpt_path

    # Scope outputs & library to this iteration
    cfg.output_dir = iter_output_dir
    lib_root = Path(library_dir)
    cfg.seed_skill_dir = str(lib_root / "seed") + "/"
    cfg.generated_skill_dir = str(lib_root / "generated") + "/"
    cfg.skill_snapshots_dir = str(lib_root / "snapshots") + "/"

    cfg.num_epochs = num_epochs
    if seed_samples is not None:
        cfg.validation.seed_samples_per_dataset = seed_samples

    # Two-stage Phase A prefilter
    cfg.prefilter_baseline_failures = bool(prefilter_baseline_failures)
    cfg.prefilter_cap_k = int(prefilter_cap_k)

    logger.info("Evolve step: student=%s, lib=%s, out=%s",
                student_ckpt_path, library_dir, iter_output_dir)
    pipeline = SelfImprovingPipeline(cfg)
    pipeline.setup()
    return pipeline.run(start_epoch=0)
