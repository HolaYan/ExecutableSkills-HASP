"""The eval at the pause — and the corpus it produces.

Two jobs at once, which is the reason to do this inside training at all:

  1. it measures the checkpoint (pass@1 on a held-out set), so the run has a
     progress curve that is not just training loss;
  2. the same rollouts, split by correctness, ARE the screening corpus the PF
     distiller needs — wrong cases to mine, correct cases as the
     false-positive floor. No separate mining pass, and the failures are the
     CURRENT model's failures rather than the base model's.

Generation runs on the live training model with HF `generate`: mid-run there
is no spare GPU for a vLLM engine, and the trainer's weights are the ones we
want to evaluate anyway. That makes this slow per-sample, so keep `eval_size`
small — it is a probe, not the n=64 protocol.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))
# Resolved lazily: this module is imported by the training callback, which
# must not require the upstream evaluation repo unless an eval actually runs.


@contextmanager
def _inference_mode(model):
    """Generate from a model that is mid-training, then put it back exactly.

    Gradient checkpointing forces `use_cache=False`; leaving it off makes
    generation quadratic, and leaving it ON afterwards silently breaks the
    next training step. Both are restored.
    """
    import torch
    was_training = model.training
    cfg = getattr(model, "config", None)
    prev_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
    gc_enabled = getattr(model, "is_gradient_checkpointing", False)
    try:
        if gc_enabled and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if cfg is not None:
            cfg.use_cache = True
        model.eval()
        with torch.no_grad():
            yield
    finally:
        if cfg is not None and prev_cache is not None:
            cfg.use_cache = prev_cache
        if gc_enabled and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if was_training:
            model.train()


def load_eval_set(domain: str, dataset: str = "", size: int = 48,
                  seed: int = 0) -> Tuple[List[str], List[str], List[str]]:
    """(questions, golds, ids) from a held-out parquet.

    Gold answers go through `norm_gold`: AMC23 stores integer answers as float
    strings ("27.0"), which scores as 0 against a model that answers "27".
    """
    import pandas as pd
    from pf_select.eval_models import norm_gold

    ds = dataset or {"math": "math500", "code": "humaneval_plus", "web": "hotpotqa"}.get(domain, "math500")
    p = _HASP / "data" / "eval" / f"{ds}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no held-out set at {p} — set evolve.eval_dataset")
    df = pd.read_parquet(p)
    if len(df) > size:
        df = df.sample(n=size, random_state=seed)
    return (df["question"].tolist(),
            [norm_gold(g) for g in df["gold_answer"].astype(str).tolist()],
            df["id"].astype(str).tolist())


def run_eval(model, tok, questions: List[str], golds: List[str], ids: List[str],
             *, max_new_tokens: int = 1024, batch_size: int = 8,
             step: int = 0) -> Tuple[float, List[Dict]]:
    """-> (pass@1, corpus rows in forge's screening format).

    Greedy, one sample per question: this is a progress probe and a corpus
    builder, not a pass@k measurement.
    """
    from pf_select.react_prompts import build_react_user_prompt
    from verifiers.reference_em import (
        em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
    )

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": build_react_user_prompt(q)}],
        tokenize=False, add_generation_prompt=True) for q in questions]

    texts: List[str] = []
    prev_side = tok.padding_side
    tok.padding_side = "left"          # required for correct batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        with _inference_mode(model):
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(model.device)
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                for j in range(len(chunk)):
                    gen = out[j][enc["input_ids"].shape[1]:]
                    texts.append(tok.decode(gen, skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side

    rows, n_ok = [], 0
    for qid, q, g, t in zip(ids, questions, golds, texts):
        pred = _extract_answer_math(t) or ""
        ok = bool(_em_match_multi(pred, g))
        n_ok += ok
        rows.append(dict(uid=f"gen{step}_{qid}", label="correct" if ok else "wrong",
                         dataset="evolve_probe", question=q, response=t,
                         pred=pred, gold=g))
    return (n_ok / max(1, len(texts))), rows
