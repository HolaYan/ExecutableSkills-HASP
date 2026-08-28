"""The training-loop hook: every N steps, pause → eval → distill → grow.

    on_step_end(step % every_steps == 0)
        │
        ├─ eval the LIVE weights on a held-out set        (pass@1 for the curve)
        ├─ split those rollouts into wrong / correct      (the screening corpus)
        ├─ cluster the failures, propose PFs, gate them   (cheap gates only)
        └─ append survivors to the run-scoped library     (with provenance)

Two properties this callback must have, because it runs unattended inside a
job that may have been queued for a day:

  * it must never kill the run. Every cycle is wrapped; a failure is logged
    and training continues with the library it already had.
  * it must leave the model exactly as it found it. Generation needs eval mode
    and a KV cache, training needs neither; `_inference_mode` restores both.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional

from .config import EvolveConfig
from .library import EvolvingLibrary

logger = logging.getLogger(__name__)


def run_cycle(model, tok, cfg: EvolveConfig, lib: EvolvingLibrary,
              generation: int, step: int, workdir: Path) -> dict:
    """One evolve cycle. Returns a record; raises nothing the caller must catch
    beyond what `EvolveCallback` already handles."""
    from .distill import distill
    from .eval_probe import load_eval_set, run_eval

    qs, golds, ids = load_eval_set(cfg.domain, cfg.eval_dataset, cfg.eval_size,
                                   seed=generation)
    pass1, cases = run_eval(model, tok, qs, golds, ids,
                            max_new_tokens=cfg.eval_max_new_tokens,
                            batch_size=cfg.eval_batch_size, step=step)
    n_wrong = sum(1 for c in cases if c["label"] == "wrong")
    logger.info("[evolve] gen %d @ step %d: pass@1 %.4f on %d held-out (%d failures)",
                generation, step, pass1, len(cases), n_wrong)

    review = {}
    if cfg.review_enabled:
        review = _review(cases, cfg)
        if review:
            from .review import render
            logger.info("[evolve] library review, worst outcome first:\n%s", render(review))

    specs = distill(cases, cfg, model, tok, workdir / f"gen{generation}",
                    falsified_note=lib.falsified_note(),
                    known_ids=set(lib.admitted_ids()))
    n = lib.admit(specs, generation, step, pass1=pass1, review=review)
    if n:
        logger.info("[evolve] gen %d admitted %d PF(s): %s", generation, n,
                    ", ".join(s.skill_id for s in specs))
    else:
        logger.info("[evolve] gen %d admitted nothing — the library is unchanged", generation)
    return dict(generation=generation, step=step, pass1=pass1,
                n_failures=n_wrong, admitted=[s.skill_id for s in specs],
                review=review)


def _review(cases, cfg: EvolveConfig):
    """Score the skills that fired on this cycle's own rollouts.

    Dispatch is re-run over the evaluated rollouts to recover PF activation
    records — generation is the expensive half and it has already happened, so
    this is CPU only. A failure here must not cost the cycle its new skills, so
    it degrades to an empty review.
    """
    try:
        from pf_select.pf_select_eval import _load_pf_system, _build_step_context
        from .review import as_trajectory, flag_for_retirement, review_library
        exec_pf, library = _load_pf_system("skills")
        ids = sorted(library)
        trajs = []
        for c in cases:
            sc = _build_step_context(c["question"], c["response"])
            sc.update(raw_reasoning=c["response"], uid=c["uid"], domain=cfg.domain)
            _, _, recs, _ = exec_pf(active_skill_ids=ids, step_context=sc,
                                    action_type="FINAL", arg=c.get("pred", ""),
                                    reasoning=c["response"], teacher_model=None)
            trajs.append(as_trajectory(c["uid"], c["question"], c["response"],
                                       c["label"] == "correct", recs, ids))
        rev = review_library(trajs)
        for f in flag_for_retirement(rev, min_fires=cfg.review_min_fires):
            logger.info("[evolve] %s is not paying off: %s",
                        f["skill_id"], "; ".join(f["reasons"]))
        return rev
    except Exception as e:
        logger.warning("[evolve] library review unavailable (%s); continuing", e)
        return {}


def build_callback(cfg: EvolveConfig, output_dir: str):
    """Construct the HF `TrainerCallback`, or None when evolution is off."""
    if not cfg.enabled:
        return None
    from transformers import TrainerCallback

    lib_root = Path(cfg.library_dir or (Path(output_dir) / "library"))
    lib = EvolvingLibrary.create(lib_root, cfg.domain)
    work = Path(output_dir) / "evolve"
    work.mkdir(parents=True, exist_ok=True)

    class EvolveCallback(TrainerCallback):
        def __init__(self):
            self.generation = 0
            self.records: List[dict] = []

        def on_step_end(self, args, state, control, model=None, **kw):
            step = int(state.global_step)
            if step < cfg.skip_first or step % cfg.every_steps != 0:
                return control
            if self.generation >= cfg.max_generations:
                return control
            if model is None:
                logger.warning("[evolve] no model handed to the callback; skipping")
                return control
            tok = kw.get("processing_class") or kw.get("tokenizer")
            if tok is None:
                logger.warning("[evolve] no tokenizer handed to the callback; skipping")
                return control

            self.generation += 1
            try:
                rec = run_cycle(model, tok, cfg, lib, self.generation, step, work)
                self.records.append(rec)
            except Exception:
                # never take the training run down with us
                logger.warning("[evolve] generation %d failed; training continues\n%s",
                               self.generation, traceback.format_exc())
            return control

        def on_train_end(self, args, state, control, **kw):
            logger.info("[evolve] %s", lib.summary())
            logger.info("[evolve] library: %s", lib.root)
            logger.info("[evolve] admitted PFs are PROVISIONAL — screened only. "
                        "Measure them with forge before reporting any of this "
                        "as an accuracy result.")
            return control

    return EvolveCallback()
