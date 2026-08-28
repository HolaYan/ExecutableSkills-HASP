"""forge CLI — staged, like the rest of the workspace's harnesses.

    python -m skills_construct.forge --stage cluster --domain math
    python -m skills_construct.forge --stage propose --domain math --model Qwen/Qwen3-8B   # GPU
    python -m skills_construct.forge --stage screen  --domain math                          # CPU
    python -m skills_construct.forge --stage emit    --domain math [--register]

Stages are separate because they need different machines: propose wants a GPU,
screen is CPU-only and belongs in a `cs` job, cluster and emit are seconds on
the login node. Each stage reads the previous stage's file from
`data/skills_construct/forge/<tag>/`, so any stage can be re-run alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[2]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills_construct.forge import cluster as _cluster          # noqa: E402
from skills_construct.forge import corpus as _corpus            # noqa: E402
from skills_construct.forge import emit as _emit                # noqa: E402
from skills_construct.forge import screen as _screen            # noqa: E402
from skills_construct.forge.spec import PFSpec                  # noqa: E402


def _dir(tag: str) -> Path:
    d = _HASP / "data" / "forge" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cluster", "propose", "screen", "emit",
                             "probe", "round", "admit", "ledger"])
    ap.add_argument("--domain", default="math", choices=["math", "code", "web"])
    ap.add_argument("--tag", default=None, help="run tag (default: <domain>1)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="proposer model (propose stage)")
    ap.add_argument("--k", type=int, default=3, help="PFs to propose per family")
    ap.add_argument("--n", type=int, default=2, help="samples per family prompt")
    ap.add_argument("--families", type=int, default=6, help="top-N families to propose for")
    ap.add_argument("--min-population", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cpu-s", type=int, default=180, help="CPU seconds per candidate (screen)")
    ap.add_argument("--register", action="store_true",
                    help="emit stage: also chain-load generated_pfs.py from the domain library")
    ap.add_argument("--policy", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="policy model regenerated during the probe (tier 1)")
    ap.add_argument("--no-refine", action="store_true", help="round stage: skip the refine pass")
    ap.add_argument("--acc-tag", default=None, help="admit stage: tier-2 run to read")
    ap.add_argument("--candidates", nargs="*", default=[], help="admit stage: skill ids to promote")
    ap.add_argument("--min-delta-pp", type=float, default=0.0,
                    help="admit stage: mean accuracy gain required, in points")
    a = ap.parse_args()
    tag = a.tag or f"{a.domain}1"
    D = _dir(tag)

    if a.stage == "ledger":
        from skills_construct.forge.ledger import Ledger
        led = Ledger(domain=a.domain)
        print(led.summary())
        for r in led.records:
            why = (r.get("reasons") or [""])[0]
            print(f"  r{r.get('round','?')} [{r.get('status','?'):<12}] {r['skill_id']:<32} "
                  f"{r.get('family',''):<16} {why[:70]}")
        dead = led.dead_families()
        if dead:
            print(f"\n  exhausted families (skipped by future rounds): {sorted(dead)}")
        return

    if a.stage == "admit":
        from skills_construct.forge.loop import admit
        if not a.acc_tag:
            raise SystemExit("--acc-tag is required for the admit stage")
        admit(a.domain, a.acc_tag, a.candidates, min_delta_pp=a.min_delta_pp)
        return

    cases = _corpus.load(a.domain)
    print(f"[corpus] {a.domain}: {_corpus.summary(cases)}", flush=True)

    if a.stage == "round":
        from skills_construct.forge.loop import run_round
        run_round(a.domain, tag, a.model, cases, D, k=a.k, n_families=a.families,
                  n_samples=a.n, tp=a.tp, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len,
                  max_tokens=a.max_tokens, policy=a.policy, min_population=a.min_population,
                  cpu_s=a.cpu_s, do_refine=not a.no_refine)
        return

    if a.stage == "cluster":
        fams = _cluster.cluster(cases, min_population=a.min_population)
        (D / "families.json").write_text(json.dumps(fams, indent=2))
        print(_cluster.render(fams))
        print(f"\n[cluster] {len(fams)} families -> {D / 'families.json'}")
        return

    if a.stage == "propose":
        from skills_construct.forge.propose import propose
        fams = json.loads((D / "families.json").read_text())[: a.families]
        specs = propose(fams, cases, a.domain, a.model, k=a.k, tp=a.tp, gpu_mem=a.gpu_mem,
                        max_model_len=a.max_model_len, max_tokens=a.max_tokens, n=a.n)
        (D / "candidates.jsonl").write_text(
            "".join(json.dumps(s.to_json()) + "\n" for s in specs))
        print(f"[propose] {len(specs)} candidates -> {D / 'candidates.jsonl'}")
        return

    specs = [PFSpec.from_json(json.loads(l)) for l in (D / "candidates.jsonl").open() if l.strip()]

    if a.stage == "probe":
        from skills_construct.forge.loop import probe_candidates
        from skills_construct.forge.measure import probe_verdict
        p = D / "screened.jsonl"
        if p.exists():
            specs = [PFSpec.from_json(json.loads(l)) for l in p.open() if l.strip()]
        pool = [s for s in specs if (s.screen or {}).get("verdict") == "accept"]
        print(f"[probe] {len(pool)} screened-accepted candidates, tier 1 (marginal, NOT accuracy)")
        scored = probe_candidates(cases, pool, a.domain, D / "probe", a.policy,
                                  a.tp, a.gpu_mem, a.max_model_len)
        for sid, pr in scored.items():
            st, why = probe_verdict(pr)
            print(f"  [{st:<10}] {sid:<32} fired {pr['fired']:>3} "
                  f"(w{pr['fired_wrong']}/c{pr['fired_correct']})  rescue {pr['rescue']}  "
                  f"broke {pr['broke']}   {why[0] if why else ''}")
        (D / "probe.json").write_text(json.dumps(scored, indent=2))
        return

    if a.stage == "screen":
        print(f"[screen] {len(specs)} candidates against {_corpus.summary(cases)}\n"
              f"         gate: fire_wrong>={_screen.MIN_FIRE_WRONG:.0%}, "
              f"fire_correct<={_screen.MAX_FIRE_CORRECT:.0%}, lift>={_screen.MIN_LIFT}", flush=True)
        res = _screen.screen_all(specs, cases, D / "work", cpu_s=a.cpu_s)
        (D / "screened.jsonl").write_text(
            "".join(json.dumps(s.to_json()) + "\n" for s in specs))
        acc = [r for r in res if r.verdict == "accept"]
        print(f"\n[screen] {len(acc)}/{len(res)} accepted -> {D / 'screened.jsonl'}")
        for r in sorted(acc, key=lambda x: -x.lift):
            print(f"  {r.skill_id:<34} lift {r.lift:>5.2f}  "
                  f"(wrong {r.fire_wrong:.1%} / correct {r.fire_correct:.1%})")
        return

    # emit
    p = D / "screened.jsonl"
    if p.exists():
        specs = [PFSpec.from_json(json.loads(l)) for l in p.open() if l.strip()]
    out = _emit.emit(specs, a.domain, tag, _corpus.summary(cases), register=a.register)
    n = sum(1 for s in specs if (s.screen or {}).get("verdict") == "accept")
    print(f"[emit] {n} PFs -> {out}")
    print("[emit] " + ("chain-loaded from dynamic_program_functions.py"
                       if a.register else
                       "NOT loaded — re-run with --register to wire them in"))
    if n:
        print("[emit] next: measure accuracy before -> after with "
              "anchor/eval_polished_pfs.py before any training use")


if __name__ == "__main__":
    main()
