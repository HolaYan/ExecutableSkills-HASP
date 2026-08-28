"""Failures → PF specs, mid-training.

This is forge's `cluster → propose → screen` chain run under two
constraints that do not apply offline:

  * the proposer is the model being trained (it is the only one on the GPU),
    so proposals are weaker than an offline Qwen3-8B's — the gates matter more
    here, not less;
  * the expensive gates are unavailable. The end-to-end probe needs a separate
    regeneration pass and the accuracy test needs n=64; neither fits inside a
    training step. What runs is the structural gate plus the offline precision
    screen against the correct-set control from this same eval.

So anything this module admits is PROVISIONAL. It has been shown to fire on
the current model's failures and to stay quiet on its successes — it has NOT
been shown to change an answer, let alone accuracy. `evolving/README.md` says
where that debt gets paid.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills_construct.forge import cluster as _cluster           # noqa: E402
from skills_construct.forge import screen as _screen             # noqa: E402
from skills_construct.forge.propose import build_prompts, _parse  # noqa: E402
from skills_construct.forge.spec import PFSpec, validate_spec, registered_pf_ids  # noqa: E402

from .config import EvolveConfig                   # noqa: E402


def _generate(model, tok, prompts: List[str], max_new_tokens: int,
              batch_size: int = 2) -> List[str]:
    """Sample proposals from the live training model."""
    from .eval_probe import _inference_mode
    outs: List[str] = []
    prev_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        with _inference_mode(model):
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=8192).to(model.device)
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.8, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
                for j in range(len(chunk)):
                    outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
    return outs


def distill(cases: List[Dict], cfg: EvolveConfig, model, tok, workdir: Path,
            falsified_note: str = "", known_ids: Optional[set] = None) -> List[PFSpec]:
    """One cycle: cluster the failures, propose, gate. Returns admitted specs."""
    workdir.mkdir(parents=True, exist_ok=True)
    n_wrong = sum(1 for c in cases if c["label"] == "wrong")
    if n_wrong < 8:
        print(f"[evolve] only {n_wrong} failures this cycle — too few to cluster; skipping")
        return []

    fams = _cluster.cluster(cases, min_population=4)[: cfg.families_per_cycle]
    if not fams:
        print("[evolve] no family reached the population floor; skipping")
        return []
    print(f"[evolve] families: " + ", ".join(f"{f['family']}({f['n_wrong']})" for f in fams))

    prompts = build_prompts(fams, cases, cfg.candidates_per_family,
                            falsified_note=falsified_note)
    chat = [tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts]
    raw = _generate(model, tok, chat, cfg.propose_max_tokens)

    known = (known_ids or set()) | registered_pf_ids(_HASP)
    specs, seen, dropped = [], set(), 0
    for p, txt in zip(prompts, raw):
        for s in _parse(txt, cfg.domain, p["family"]):
            if s.skill_id in seen:
                continue
            if validate_spec(s, known):
                dropped += 1
                continue
            fam = next((f for f in fams if f["family"] == p["family"]), None)
            s.source_uids = (fam or {}).get("examples", [])[:8]
            seen.add(s.skill_id)
            specs.append(s)
    print(f"[evolve] {len(specs)} proposals passed the structural gate ({dropped} dropped)")
    if not specs:
        return []

    # precision screen, with this cycle's own correct set as the control
    prev = (_screen.MIN_FIRE_WRONG, _screen.MAX_FIRE_CORRECT, _screen.MIN_LIFT)
    _screen.MIN_FIRE_WRONG, _screen.MAX_FIRE_CORRECT, _screen.MIN_LIFT = (
        cfg.min_fire_wrong, cfg.max_fire_correct, cfg.min_lift)
    try:
        _screen.screen_all(specs, cases, workdir / "screen", cpu_s=60, wall_s=120)
    finally:
        _screen.MIN_FIRE_WRONG, _screen.MAX_FIRE_CORRECT, _screen.MIN_LIFT = prev

    accepted = [s for s in specs if (s.screen or {}).get("verdict") == "accept"]
    accepted.sort(key=lambda s: -(s.screen or {}).get("lift", 0.0))
    if len(accepted) > cfg.max_admit_per_cycle:
        print(f"[evolve] {len(accepted)} passed; capping to the {cfg.max_admit_per_cycle} "
              "with the highest lift")
        accepted = accepted[: cfg.max_admit_per_cycle]
    return accepted
