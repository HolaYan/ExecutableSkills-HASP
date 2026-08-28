"""Rebuild Obj-A training data from the latest self-improving outputs.

After each library-evolve step, the skill library has changed, so trajectories
collected under the new library produce different PF signals. This module
re-runs the data builders (UsePFsBuilder / EvolveBuilder) against the latest
trajectories and writes fresh jsonls for the next train iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from ..signals import SignalAggregator
from ..signals.aggregator import AggregatorConfig
from ..data import UsePFsBuilder, EvolveBuilder
from ..data.use_pfs_builder import UsePFsBuilderConfig
from ..data.evolve_builder import EvolveBuilderConfig
from ..data.signal_filter import resolve_enabled_signals
from ..sft.train import _load_trajectories, _load_candidates_and_reviews

logger = logging.getLogger(__name__)


def refresh_objA(
    self_improving_dir: str,
    out_dir: str,
    enabled_signals: str = "all",
    threshold: float = 0.25,
    formats: List[str] = None,
    signal_mode: str = "coarse",
) -> Dict[str, Path]:
    """Rebuild Obj-A jsonls from the trajectories under `self_improving_dir`.

    `signal_mode`:
        "coarse" (default) — collapse 15 sub-signals into 4 family scalars
                             {correctness, timing, modality, outcome} and
                             combine via DEFAULT_FAMILY_WEIGHTS.
        "fine"            — legacy 15-dim weighted sum.
    """
    formats = formats or ["sft", "prompt"]
    enabled = resolve_enabled_signals(enabled_signals)
    agg = SignalAggregator(AggregatorConfig(
        enabled=enabled, normalize=True, mode=signal_mode,
    ))

    trajs = _load_trajectories(self_improving_dir)
    if not trajs:
        raise RuntimeError(f"No trajectories under {self_improving_dir}")

    builder = UsePFsBuilder(
        UsePFsBuilderConfig(
            output_dir=str(out_dir),
            enabled_signals=enabled,
            threshold=threshold,
            formats=formats,
        ),
        agg,
    )
    outputs = builder.build(trajs)
    logger.info("Refreshed Obj-A data at %s", out_dir)
    return outputs


def refresh_objB(
    self_improving_dir: str,
    out_dir: str,
    q_skill_threshold: float = 0.5,
    lam_val_gain: float = 0.5,
    formats: List[str] = None,
) -> Dict[str, Path]:
    """Rebuild Obj-B jsonls from candidates/reviews under `self_improving_dir`."""
    formats = formats or ["sft"]
    agg = SignalAggregator(AggregatorConfig(enabled=resolve_enabled_signals("all"), normalize=True))

    cands, revs = _load_candidates_and_reviews(self_improving_dir)
    if not cands or not revs:
        logger.warning("No candidates/reviews under %s — skipping Obj-B refresh", self_improving_dir)
        return {}

    builder = EvolveBuilder(
        EvolveBuilderConfig(
            output_dir=str(out_dir),
            q_skill_threshold=q_skill_threshold,
            lam_val_gain=lam_val_gain,
            formats=formats,
        ),
        agg,
    )
    outputs = builder.build(cands, revs)
    logger.info("Refreshed Obj-B data at %s", out_dir)
    return outputs
