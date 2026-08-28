"""Offline Case-C evaluation of the code evidence PFs (spec_example_check, …).

Population: Skills_Agent factorial-ablation code episodes (humaneval+ / mbpp+,
arms baseline / pf_no_teacher / pf_with_teacher) with their codejudge verdicts
→ failing (wrong) and passing (correct) solutions. Hidden tests come from
data/code/{humaneval_plus,mbpp_plus}.jsonl (`eval_test_code`, `entry_point`).

dispatch (CPU)  : run the code evidence PFs on every episode; evidence → Case C
regen (7B)      : chat = [user: spec prompt] [assistant: original solution]
                  [user: [System Feedback] <evidence> → corrected full function]
                  2 samples at T=0.7  (policy: Qwen2.5-7B-Instruct)
score (CPU)     : CodeSandbox.evaluate_with_test_script on the hidden driver.
                  gate = every regenerated sample passes the spec's own examples
                  (inference-time checkable) else keep the original.
Reports rescue on failing, broke on passing, pass@1 before/after.
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys
from collections import Counter
from pathlib import Path
_HASP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_HASP))
from hasp_paths import setup_compile_caches  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402
_P = _protocol()
setup_compile_caches()   # before torch/vllm are imported
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("code_evidence", str(_HASP / "skills/executable/code/checkers.py"))
CPF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CPF)
from src.skills_agent.eval.code_sandbox import CodeSandbox  # noqa: E402

from hasp_paths import code_data_dir, code_episodes_dir  # noqa: E402
BENCHMARKS = tuple(os.environ.get("CODE_BENCHMARKS", "humaneval_plus,mbpp_plus").split(","))
EP_ROOT = code_episodes_dir()
RAW = code_data_dir()
OUT = _HASP / "data" / "code_polished"
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _parse_final(f):
    s = str(f or "")
    if s.strip().startswith("{"):
        try: return str(ast.literal_eval(s).get("answer", s))
        except Exception: return s
    return s


def _extract(text: str) -> str:
    m = _FENCE.findall(text)
    return (m[-1] if m else text).strip()


def build_cases(per_arm_cap: int) -> list[dict]:
    lookup = {}
    for bm in BENCHMARKS:
        for row in map(json.loads, (RAW / f"{bm}.jsonl").open()):
            lookup[str(row["sample_id"])] = dict(eval_test_code=row.get("eval_test_code", ""), entry_point=row.get("entry_point"), benchmark=bm)
    cases = []
    for bm in BENCHMARKS:
        for arm in ("baseline", "pf_no_teacher", "pf_with_teacher"):
            try:
                jj = json.load(open(EP_ROOT / bm / arm / "base_clean_episodes.codejudge.json"))["results"]
                eps = {str(e["sample_id"]): e for e in map(json.loads, (EP_ROOT / bm / arm / "base_clean_episodes.jsonl").open())}
            except Exception:
                continue
            kept = Counter()
            for it in jj:
                sid = str(it["sample_id"]); e = eps.get(sid); info = lookup.get(sid)
                if e is None or not info or not info["eval_test_code"]:
                    continue
                lab = "correct" if it.get("passed") else "wrong"
                if kept[lab] >= per_arm_cap:
                    continue
                kept[lab] += 1
                cases.append(dict(uid=f"{lab[0].upper()}:{bm}:{arm}:{sid}", label=lab, benchmark=bm, arm=arm, sample_id=sid,
                                  question=e["question"], answer=_parse_final(e.get("final")),
                                  eval_test_code=info["eval_test_code"], entry_point=info["entry_point"]))
    return cases


def stage_dispatch(cases):
    out = []
    for c in cases:
        ctx = dict(question=c["question"], entry_point=c["entry_point"])
        parts = []
        for sid, fn in (("spec_example_check", CPF.spec_example_evidence), ("signature_conformance_check", CPF.signature_conformance),
                        ("exception_contract_check", getattr(CPF, "exception_contract", None)), ("api_attribute_probe", getattr(CPF, "api_attribute_probe", None))):
            if fn is None:
                continue
            try:
                ev = fn(ctx, c["answer"])
            except Exception:
                ev = None
            if ev:
                parts.append(f"[{sid} @{'example' if sid == 'spec_example_check' else 'final'}] {ev}")
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if parts else "A", feedback="\n\n".join(parts), pfs=[p.split()[0].strip('[') for p in parts]))
    return out


def stage_regen(cases, disp, policy, tp, gpu_mem, max_model_len, max_tokens, n):
    from vllm import LLM, SamplingParams
    by = {c["uid"]: c for c in cases}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); prompts, meta = [], []
    for d in disp:
        if d["case"] != "C":
            continue
        c = by[d["uid"]]
        msgs = [{"role": "user", "content": c["question"]},
                {"role": "assistant", "content": "```python\n" + c["answer"].strip() + "\n```"},
                {"role": "user", "content": f"[System Feedback]\n{d['feedback']}\n\nFix the function accordingly. Output the complete corrected function (with any imports) in ONE ```python code block and nothing else."}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tok(p).input_ids) > max_model_len - 512:
            continue
        prompts.append(p); meta.append(d["uid"])
    print(f"[regen] {len(prompts)} Case-C solutions x {n}", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.code.temperature, max_tokens=max_tokens, n=n), use_tqdm=True)
    return [dict(uid=u, texts=[x.text for x in o.outputs]) for u, o in zip(meta, outs)]


def stage_score(cases, disp, regs):
    by = {c["uid"]: c for c in cases}; n = Counter(c["label"] for c in cases)
    dist = Counter((d["label"], d["case"]) for d in disp)
    print("=== dispatch (code evidence PFs, FINAL path) ===")
    for lab in ("wrong", "correct"):
        print(f"  {lab:<8} n={n[lab]:>4}  Case C {dist[(lab,'C')]:>4} ({dist[(lab,'C')]/max(1,n[lab]):.0%})  silent {dist[(lab,'A')]:>4}")
    sb = CodeSandbox(wall_timeout_s=20.0)
    def hidden_pass(c, code):
        try:
            return bool(sb.evaluate_with_test_script(code, c["eval_test_code"], entry_point=c["entry_point"]).pass_at_1)
        except Exception:
            return False
    out = Counter()
    for r in regs:
        c = by[r["uid"]]; was = (c["label"] == "correct")
        codes = [_extract(t) for t in r["texts"]]
        ctx = dict(question=c["question"], entry_point=c["entry_point"])
        # gate: every regenerated sample passes the spec's own examples (checkable at inference)
        gate_ok = all(CPF.spec_example_evidence(ctx, cd) is None for cd in codes) and bool(codes[0].strip())
        for g, acc in (("nogate", bool(codes[0].strip())), ("gate", gate_ok)):
            now = hidden_pass(c, codes[0]) if acc else was
            out[f"{c['label']}:{g}:rescue"] += (not was and now); out[f"{c['label']}:{g}:broke"] += (was and not now)
    print("\n=== regeneration of Case C (Qwen2.5-7B-Instruct), hidden tests ===")
    for g in ("nogate", "gate"):
        print(f"  {g:<7} failing: rescue {out[f'wrong:{g}:rescue']:>3}/{n['wrong']}   passing: broke {out[f'correct:{g}:broke']:>3}/{n['correct']}")
    tot = n["wrong"] + n["correct"]
    for g in ("nogate", "gate"):
        d = out[f"wrong:{g}:rescue"] - out[f"correct:{g}:broke"]
        print(f"  pass@1 on this population: {n['correct']/tot:.1%} → {(n['correct']+d)/tot:.1%}  ({g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "dispatch", "regen", "score"])
    ap.add_argument("--tag", default="code1"); ap.add_argument("--per-arm-cap", type=int, default=_P.harness.code.per_arm_cap)
    ap.add_argument("--policy", default=_P.models.policy_code)
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel); ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization)
    ap.add_argument("--max-model-len", type=int, default=_P.harness.code.max_model_len); ap.add_argument("--max-tokens", type=int, default=_P.harness.code.max_tokens)
    ap.add_argument("--n", type=int, default=_P.harness.code.n)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    P = lambda nm: OUT / f"{nm}_{a.tag}.jsonl"
    if a.stage == "build":
        cs = build_cases(a.per_arm_cap); P("cases").write_text("".join(json.dumps(x) + "\n" for x in cs))
        print(f"[build] {len(cs)} cases {Counter(c['label'] for c in cs)} -> {P('cases')}"); return
    cases = [json.loads(l) for l in P("cases").open()]
    if a.stage == "dispatch":
        d = stage_dispatch(cases); P("dispatch").write_text("".join(json.dumps(x) + "\n" for x in d)); print("[dispatch] ->", P("dispatch")); return
    disp = [json.loads(l) for l in P("dispatch").open()]
    if a.stage == "regen":
        r = stage_regen(cases, disp, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens, a.n)
        P("regen").write_text("".join(json.dumps(x) + "\n" for x in r)); return
    stage_score(cases, disp, [json.loads(l) for l in P("regen").open()] if P("regen").exists() else [])


if __name__ == "__main__":
    main()
