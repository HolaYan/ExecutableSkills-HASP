"""Evaluate the polished PFs on today's pf_select FINAL path (superset check).

dispatch (CPU): exec_pf with the polished evidence PFs on every case; only
               cases that receive evidence become Case C.
regen (4B):    prompt = ReAct prompt + full rollout + [System Feedback] <evidence>
               + "Please provide a revised answer" — exactly today's Case-C
               geometry — 2 samples at T=0.7.
score (CPU):   gate = both samples agree ∧ same answer type; else original.
Reports rescue on wrong, broke on correct, and the Case distribution.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
from pathlib import Path
_HASP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_HASP))
from hasp_paths import setup_compile_caches  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402
_P = _protocol()
setup_compile_caches()     # before torch/vllm are imported
from pf_select.pf_select_eval import _load_pf_system, _build_step_context, _extract_answer_from_text, _build_feedback  # noqa
from pf_select.react_prompts import build_react_user_prompt  # noqa
from verifiers.reference_em import (  # noqa
    em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
)

PFS = list(_P.library.base_pfs)
OUT = _HASP / "data" / "polished"
_NUM = re.compile(r"^[-+]?\d+(\.\d+)?(/\d+)?$")
def _norm(s): return (s or "").strip().strip("$").replace(",", "").rstrip(".")
def _atype(s): s = _norm(s); return "none" if not s else ("num" if _NUM.match(s) else "expr")


def stage_dispatch(cases):
    exec_pf, lib = _load_pf_system(str(_HASP / "skills"))
    out = []
    for c in cases:
        sc = _build_step_context(c["question"], c["response"])
        sc.update(raw_reasoning=c["response"], candidate_answer=_extract_answer_from_text(c["response"]), uid=c["uid"])
        fa, arg, recs, inj = exec_pf(active_skill_ids=PFS, step_context=sc, action_type="FINAL",
                                     arg=sc["candidate_answer"] or c["response"], reasoning=c["response"], helper_model=None)
        fb = _build_feedback(recs, inj, fa, arg)
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if fb else "A", feedback=fb,
                        pfs=[r.skill_id for r in recs if getattr(r, "activated", False)]))
    return out


def stage_regen(cases, disp, policy, tp, gpu_mem, max_model_len, max_tokens, n):
    from vllm import LLM, SamplingParams
    by = {c["uid"]: c for c in cases}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
              trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); prompts, meta = [], []
    for d in disp:
        if d["case"] != "C":
            continue
        c = by[d["uid"]]
        base = tok.apply_chat_template([{"role": "user", "content": build_react_user_prompt(c["question"])}], tokenize=False, add_generation_prompt=True)
        p = base + c["response"].rstrip() + f"\n\n[System Feedback]\n{d['feedback']}\n\nPlease provide a revised answer. You MUST end with `Action: finish[<answer>]`.\n\nRevised answer:\n"
        if len(tok(p).input_ids) > max_model_len - 512:
            continue
        prompts.append(p); meta.append(d["uid"])
    print(f"[regen] {len(prompts)} Case-C rollouts x {n}", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.math.temperature, max_tokens=max_tokens, n=n), use_tqdm=True)
    return [dict(uid=u, texts=[x.text for x in o.outputs]) for u, o in zip(meta, outs)]


def stage_score(cases, disp, regs):
    by = {c["uid"]: c for c in cases}; n = Counter(c["label"] for c in cases)
    dist = Counter((d["label"], d["case"]) for d in disp)
    print("=== dispatch (polished PFs, deterministic evidence, FINAL path) ===")
    for lab in ("wrong", "correct"):
        print(f"  {lab:<8} n={n[lab]:>4}  Case C (evidence) {dist[(lab,'C')]:>4} ({dist[(lab,'C')]/max(1,n[lab]):.0%})  Case A (silent) {dist[(lab,'A')]:>4}")
    pfc = Counter(p for d in disp if d["case"] == "C" for p in set(re.findall(r"\[([a-z_]+)\]", d["feedback"])))
    print("  evidence by PF:", dict(pfc.most_common()))
    out = Counter()
    for r in regs:
        c = by[r["uid"]]; was = _em_match_multi(c["pred"], c["gold"])
        ans = [_extract_answer_math(t) or "" for t in r["texts"]]; a0 = ans[0]
        for g, acc in (("nogate", bool(a0)), ("gate", bool(a0) and all(_norm(x) == _norm(a0) for x in ans) and _atype(a0) == _atype(c["pred"]))):
            fin = a0 if acc else c["pred"]; now = _em_match_multi(fin, c["gold"])
            out[f"{c['label']}:{g}:rescue"] += (not was and now); out[f"{c['label']}:{g}:broke"] += (was and not now)
    print("\n=== regeneration of Case C (today's geometry) ===")
    for g in ("nogate", "gate"):
        print(f"  {g:<7} wrong: rescue {out[f'wrong:{g}:rescue']:>3}/{n['wrong']}   correct: broke {out[f'correct:{g}:broke']:>3}/{n['correct']}")
    print("\n  (reference: old generic-reminder PFs on this population: rescue 3/2300, broke 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dispatch", "regen", "score"])
    ap.add_argument("--cases", default=str(_HASP / "data/llm_anchor/cases.jsonl"))
    ap.add_argument("--tag", default="pol1")
    ap.add_argument("--policy", default=_P.models.policy_math)
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel); ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization)
    ap.add_argument("--max-model-len", type=int, default=_P.harness.math.max_model_len); ap.add_argument("--max-tokens", type=int, default=_P.harness.math.max_tokens)
    ap.add_argument("--n", type=int, default=_P.harness.math.n)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(a.cases)]
    P = lambda nm: OUT / f"{nm}_{a.tag}.jsonl"
    if a.stage == "dispatch":
        d = stage_dispatch(cases); P("dispatch").write_text("".join(json.dumps(x) + "\n" for x in d)); print("[dispatch] ->", P("dispatch")); return
    disp = [json.loads(l) for l in P("dispatch").open()]
    if a.stage == "regen":
        r = stage_regen(cases, disp, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens, a.n)
        P("regen").write_text("".join(json.dumps(x) + "\n" for x in r)); return
    stage_score(cases, disp, [json.loads(l) for l in P("regen").open()] if P("regen").exists() else [])


if __name__ == "__main__":
    main()
