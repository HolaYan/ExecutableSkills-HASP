"""One round of the loop: propose -> screen -> probe -> refine -> re-probe -> emit.

    round N (this file, 1 GPU)
        cluster            families still worth working, minus the ones the
                           ledger says are exhausted
        propose            candidates, with the ledger's falsified list in the
                           prompt so dead ideas are not re-derived
        screen             offline precision gate (cheap, kills most)
        probe              marginal rescue / broke end to end (tier 1)
        refine             one pass over the near-misses, then re-screen and
                           re-probe those revisions
        emit               survivors written to a probe library, NOT admitted

    then, separately (1 GPU, expensive):
        tier 2             the real accuracy test, n=64 pf_select
        admit              `forge --stage admit --acc-tag <tag>` — the only
                           path into the hand-written library

    round N+1 reads the ledger and starts from what is left.

A round is deliberately allowed to end with zero admissions. Rounds that
falsify a family are the ones that make later rounds cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from . import cluster as _cluster
from . import measure as _measure
from . import screen as _screen
from .ledger import Ledger
from .spec import PFSpec

_HASP = Path(__file__).resolve().parents[2]


def _record(led: Ledger, specs: List[PFSpec], rnd: int, family_of: Dict[str, str],
            status: str, reasons_of: Dict[str, List[str]] | None = None) -> None:
    for s in specs:
        led.upsert(s.skill_id, round=rnd, domain=s.domain,
                   family=family_of.get(s.skill_id, ""), status=status,
                   screen=s.screen or {}, reasons=(reasons_of or {}).get(s.skill_id, []),
                   anchor=s.anchor, scope=s.family_scope)


def run_round(domain: str, tag: str, model: str, cases: List[Dict], workdir: Path,
              *, k: int = 3, n_families: int = 6, n_samples: int = 2, tp: int = 1,
              gpu_mem: float = 0.88, max_model_len: int = 32768, max_tokens: int = 4096,
              policy: str = "Qwen/Qwen3-4B-Instruct-2507", min_population: int = 8,
              cpu_s: int = 180, do_refine: bool = True) -> Dict:
    from .propose import load_engine, propose_with_engine
    from .refine import refine as _refine

    led = Ledger(domain=domain)
    rnd = led.next_round()
    workdir = workdir / f"round{rnd}"
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n[round {rnd}] domain={domain}  {led.summary()}\n{'=' * 72}", flush=True)

    # ── cluster, minus exhausted families ──
    fams = _cluster.cluster(cases, min_population=min_population)
    exhausted = set(led.dead_families())
    if exhausted:
        print(f"[round {rnd}] skipping exhausted families: {sorted(exhausted)}", flush=True)
    fams = [f for f in fams if f["family"] not in exhausted][:n_families]
    if not fams:
        print(f"[round {rnd}] no families left to work — the loop is done for this corpus")
        return dict(round=rnd, admitted=[], note="no families left")
    family_of: Dict[str, str] = {}

    # ── propose (engine stays loaded for refine) ──
    llm, tok = load_engine(model, tp, gpu_mem, max_model_len)
    specs = propose_with_engine(fams, cases, domain, llm, tok, k=k, n=n_samples,
                                max_tokens=max_tokens, falsified_note=led.falsified_note())
    for f in fams:
        for s in specs:
            family_of.setdefault(s.skill_id, f["family"])
    _record(led, specs, rnd, family_of, "proposed")
    led.save()
    if not specs:
        print(f"[round {rnd}] nothing proposed")
        return dict(round=rnd, admitted=[], note="nothing proposed")

    # ── screen ──
    print(f"\n[round {rnd}] screen {len(specs)} candidates", flush=True)
    res = _screen.screen_all(specs, cases, workdir / "screen", cpu_s=cpu_s)
    reasons = {r.skill_id: r.reasons for r in res}
    passed = [s for s in specs if (s.screen or {}).get("verdict") == "accept"]
    failed = [s for s in specs if s not in passed]
    _record(led, passed, rnd, family_of, "probed")
    _record(led, failed, rnd, family_of, "screened_out", reasons)
    led.save()

    # ── probe (tier 1) + one refinement pass ──
    survivors: List[PFSpec] = []
    pool, pass_no = passed, 1
    while pool:
        print(f"\n[round {rnd}] probe pass {pass_no}: {len(pool)} candidates", flush=True)
        probe = probe_candidates(cases, pool, domain, workdir / f"probe{pass_no}",
                                 policy, tp, gpu_mem, max_model_len)
        nxt: List[PFSpec] = []
        for s in pool:
            pr = probe.get(s.skill_id, dict(fired=0, fired_wrong=0, fired_correct=0,
                                            rescue=0, broke=0, no_change=0, misses=[]))
            status, why = _measure.probe_verdict(pr)
            led.upsert(s.skill_id, round=rnd, domain=domain,
                       family=family_of.get(s.skill_id, ""), status=status,
                       screen=s.screen or {}, probe=pr, reasons=why,
                       anchor=s.anchor, scope=s.family_scope)
            print(f"  [{status:<10}] {s.skill_id:<30} fired {pr['fired']:>3} "
                  f"(w{pr['fired_wrong']}/c{pr['fired_correct']})  "
                  f"rescue {pr['rescue']}  broke {pr['broke']}   {why[0] if why else ''}",
                  flush=True)
            if status == "measured":
                survivors.append(s)
            elif status == "refine":
                nxt.append(s)
        led.save()

        # candidates still marked `refine` keep that status in the ledger and
        # are picked up by the next round rather than refined twice here
        if not (do_refine and nxt and pass_no == 1):
            break
        print(f"\n[round {rnd}] refine {len(nxt)} near-misses", flush=True)
        v2 = _refine(nxt, probe, domain, llm, tok, max_tokens=max_tokens, n=n_samples)
        if not v2:
            break
        for s in v2:
            family_of[s.skill_id] = family_of.get(_strip_v(s.skill_id), "")
        r2 = _screen.screen_all(v2, cases, workdir / "screen_v2", cpu_s=cpu_s)
        rs2 = {r.skill_id: r.reasons for r in r2}
        ok2 = [s for s in v2 if (s.screen or {}).get("verdict") == "accept"]
        _record(led, ok2, rnd, family_of, "probed")
        _record(led, [s for s in v2 if s not in ok2], rnd, family_of, "screened_out", rs2)
        led.save()
        pool, pass_no = ok2, pass_no + 1

    # ── emit survivors to a probe library for the tier-2 accuracy run ──
    out = dict(round=rnd, survivors=[s.skill_id for s in survivors], admitted=[])
    if survivors:
        lib = _measure.build_probe_library(survivors, domain, workdir / "tier2")
        acc_tag = f"forge_{domain}_r{rnd}"
        out.update(library=str(lib), acc_tag=acc_tag,
                   acc_command=_measure.acc_command(lib, acc_tag, policy,
                                                    "aime24,amc23,olympiadbench"))
        print(f"\n[round {rnd}] {len(survivors)} candidates reached tier 2: "
              f"{', '.join(out['survivors'])}")
        print(f"[round {rnd}] measure real accuracy before -> after with:\n  {out['acc_command']}")
        print(f"[round {rnd}] then admit with:\n  python -m skills_construct.forge --stage admit "
              f"--domain {domain} --acc-tag {acc_tag} --candidates {' '.join(out['survivors'])}")
    else:
        print(f"\n[round {rnd}] no candidate reached tier 2 — the ledger now "
              f"carries why, so round {rnd + 1} starts elsewhere")

    (workdir / "round.json").write_text(json.dumps(out, indent=2))
    (workdir / "candidates.jsonl").write_text(
        "".join(json.dumps(s.to_json()) + "\n" for s in specs))
    led.save()
    print(f"\n[round {rnd}] {led.summary()}")
    return out


def _strip_v(sid: str) -> str:
    import re
    return re.sub(r"_v\d+$", "", sid)


def probe_candidates(cases: List[Dict], specs: List[PFSpec], domain: str, workdir: Path,
                     policy: str, tp: int, gpu_mem: float, max_model_len: int) -> Dict[str, Dict]:
    """Tier 1 end to end: two-arm dispatch, Case-C regeneration, marginal scoring."""
    workdir.mkdir(parents=True, exist_ok=True)
    d = _measure.probe_dispatch(cases, specs, domain, workdir)
    owned = d["owned"]
    flat = [x for v in owned.values() for x in v]
    if not flat:
        return {sid: dict(fired=0, fired_wrong=0, fired_correct=0, rescue=0, broke=0,
                          no_change=0, misses=[]) for sid in owned}
    regs = _measure.probe_regen(cases, flat, policy, tp, gpu_mem, max_model_len)
    (workdir / "regen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regs))
    scored = _measure.probe_score(cases, owned, regs)
    (workdir / "probe.json").write_text(json.dumps(scored, indent=2))
    return scored


def admit(domain: str, acc_tag: str, candidate_ids: List[str], min_delta_pp: float = 0.0) -> Dict:
    """Close the loop: read the tier-2 accuracy run and promote or retire.

    Admission requires a MEASURED accuracy change. A screening number or a
    probe rescue count is not sufficient and never has been.
    """
    led = Ledger(domain=domain)
    acc = _measure.read_acc(acc_tag)
    if acc is None:
        raise SystemExit(f"no tier-2 results under data/model_eval/{acc_tag} — "
                         "run the acc_command from the round first")
    status, why = _measure.acc_verdict(acc, min_delta_pp=min_delta_pp)
    print(f"[admit] tier-2 accuracy for {acc_tag}:")
    for ds, v in acc.items():
        print(f"  {ds:<16} {v['before']:.4f} -> {v['after']:.4f}   ({v['delta_pp']:+.2f}pp)")
    print(f"[admit] verdict: {status} — {why[0]}")

    for sid in candidate_ids:
        led.upsert(sid, status=status, acc=acc, reasons=why,
                   round=(led.get(sid) or {}).get("round", led.next_round() - 1),
                   domain=domain)
    led.save()

    if status == "admitted":
        specs = _collect_specs(domain, candidate_ids)
        if specs:
            from .emit import emit as _emit
            out = _emit(specs, domain, f"admitted_{acc_tag}", f"tier-2 {acc_tag}", register=True)
            print(f"[admit] {len(specs)} PFs written and wired into {out}")
            print("[admit] add them to measure.BASE_PFS so the next round measures "
                  "marginal effect against the new baseline")
    else:
        print("[admit] nothing promoted; the ledger now carries this result into the next round")
    return dict(status=status, acc=acc, reasons=why)


def _collect_specs(domain: str, ids: List[str]) -> List[PFSpec]:
    """Find the full specs for these ids in the round outputs."""
    found: Dict[str, PFSpec] = {}
    base = _HASP / "data" / "forge"
    for p in sorted(base.rglob("candidates.jsonl")):
        for line in p.open():
            if not line.strip():
                continue
            s = PFSpec.from_json(json.loads(line))
            if s.skill_id in ids:
                found[s.skill_id] = s
    return [found[i] for i in ids if i in found]
