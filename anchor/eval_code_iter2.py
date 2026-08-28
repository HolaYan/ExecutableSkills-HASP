"""Code PF iteration 2 — two levers measured on top of eval_code_polished (code1):

  (1) best-of-n repair: regenerate n samples; ACCEPT the first sample that passes
      every spec example / public test (inference-time checkable). code1's gate
      demanded that ALL samples pass and picked sample 0 — rescue 12/323.
  (2) differential-testing evidence for the 196 failing solutions that were
      SILENT (they pass the spec's own examples but fail hidden tests): an
      independent second solution is generated from the spec alone; both are
      executed on random inputs typed from the signature; an input on which
      they DISAGREE is concrete evidence ("on input X your code returns A, an
      independent solution returns B — decide from the spec which is right").
      No expected outputs are needed. Accepting the repair is gated by spec
      examples AND by agreement with the majority of {original, alt, repair}
      on the random inputs (cluster vote), so a repair that merely copies a
      wrong alt is rejected.

Stages: alt (7B) → diff (CPU sandbox) → regen (7B, n samples, with evidence from
code1 dispatch OR from diff) → score (hidden tests; best-of-n + cluster gate).
Population: the code1 cases (humaneval+ / mbpp+, 877 passing + 323 failing).
"""
from __future__ import annotations
import argparse, ast, json, os, random, re, sys
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
from anchor.sandbox import run_python  # noqa: E402
from src.skills_agent.eval.code_sandbox import CodeSandbox  # noqa: E402

OUT = _HASP / "data" / "code_polished"
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
_SIG = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M)


def _extract(t): m = _FENCE.findall(t); return (m[-1] if m else t).strip()


def _chat(tok, msgs):
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── typed random inputs from the signature ───────────────────────────────
def _gen_value(ann: str, rng: random.Random, depth=0):
    a = ann.replace(" ", "")
    base = re.sub(r"\[.*", "", a); inner = a[len(base) + 1:-1] if "[" in a else ""
    if base in ("int",): return rng.choice([rng.randint(-5, 5), rng.randint(-100, 100), 0, 1])
    if base in ("float",): return round(rng.uniform(-10, 10), 2)
    if base in ("bool",): return rng.choice([True, False])
    if base in ("str",): return "".join(rng.choice("abc xyz,.!?AB12") for _ in range(rng.randint(0, 12)))
    if base in ("List", "list", "Sequence", "Iterable"):
        el = inner or "int"; return [_gen_value(el, rng, depth + 1) for _ in range(rng.randint(0, 6))]
    if base in ("Tuple", "tuple"):
        return tuple(_gen_value(x, rng, depth + 1) for x in (inner.split(",") if inner else ["int", "int"]))
    if base in ("Dict", "dict"):
        k, v = (inner.split(",", 1) + ["int"])[:2] if inner else ("str", "int")
        return {_gen_value(k, rng, depth + 1): _gen_value(v, rng, depth + 1) for _ in range(rng.randint(0, 4))}
    if base in ("Optional",): return None if rng.random() < 0.3 else _gen_value(inner, rng, depth + 1)
    if base in ("Any", ""): return rng.choice([rng.randint(-5, 5), "ab", [1, 2]])
    return None


def random_inputs(spec: str, entry: str, k: int = 12, seed: int = 0):
    m = next((m for m in _SIG.finditer(spec) if not entry or m.group(1) == entry), None)
    if m is None: return None, []
    params = []
    for p in [p.strip() for p in m.group(2).split(",") if p.strip()]:
        if p.startswith("*"): break
        name, ann = (p.split(":", 1) + [""])[:2]; ann = ann.split("=")[0].strip()
        if not ann: return m.group(1), []          # untyped: no safe generator
        params.append(ann)
    rng = random.Random(seed)
    return m.group(1), [[_gen_value(a, rng) for a in params] for _ in range(k)]


def run_on_inputs(code: str, entry: str, inputs: list) -> list:
    prog = code + "\n\nimport json as _j\n_out=[]\n"
    for i, args in enumerate(inputs):
        prog += f"try:\n    _out.append(repr({entry}(*{args!r})))\nexcept Exception as e:\n    _out.append('EXC:'+type(e).__name__)\n"
    prog += "print('OUTS', _j.dumps(_out))"
    ok, out = run_python(prog, timeout_s=10)
    if not ok: return None
    m = re.search(r"OUTS (\[.*\])", out)
    return json.loads(m.group(1)) if m else None


# ── stages ───────────────────────────────────────────────────────────────
def stage_alt(cases, policy, tp, gpu_mem, max_model_len, max_tokens):
    """Independent second solution from the spec alone (no sight of the original)."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [_chat(tok, [{"role": "user", "content": c["question"] + "\n\nOutput the complete function (with imports) in ONE ```python block and nothing else."}]) for c in cases]
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.code.temperature, max_tokens=max_tokens, n=1), use_tqdm=True)
    return [dict(uid=c["uid"], alt=_extract(o.outputs[0].text if o.outputs else "")) for c, o in zip(cases, outs)]


def stage_diff(cases, alts, disp):
    by_alt = {a["uid"]: a["alt"] for a in alts}; by_d = {d["uid"]: d for d in disp}
    out = []
    for c in cases:
        d = by_d.get(c["uid"], {}); fb = d.get("feedback", ""); src = "spec" if fb else ""
        entry, inputs = random_inputs(c["question"], c["entry_point"])
        diff_ev = ""; outs_o = outs_a = None
        alt = by_alt.get(c["uid"], "")
        if entry and inputs and alt:
            co = CPF.compose_program(c["question"], c["answer"], entry); ca = CPF.compose_program(c["question"], alt, entry)
            outs_o = run_on_inputs(co, entry, inputs); outs_a = run_on_inputs(ca, entry, inputs)
            if outs_o and outs_a:
                for args, oo, oa in zip(inputs, outs_o, outs_a):
                    if oo != oa and not (oo.startswith("EXC") and oa.startswith("EXC")):
                        diff_ev = (f"[differential_test @input] on the input {entry}(*{args!r}) your code returns {oo[:80]} but an "
                                   f"independently written solution returns {oa[:80]}; re-read the specification and decide which is right, then fix the function")
                        break
        if diff_ev and not fb:
            fb, src = diff_ev, "diff"
        elif diff_ev:
            fb = fb + "\n\n" + diff_ev; src = "spec+diff"
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if fb else "A", feedback=fb, source=src,
                        entry=entry, inputs=inputs, outs_orig=outs_o, outs_alt=outs_a))
    return out


def stage_regen(cases, diff, policy, tp, gpu_mem, max_model_len, max_tokens, n):
    from vllm import LLM, SamplingParams
    by = {c["uid"]: c for c in cases}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); prompts, meta = [], []
    for d in diff:
        if d["case"] != "C": continue
        c = by[d["uid"]]
        p = _chat(tok, [{"role": "user", "content": c["question"]},
                        {"role": "assistant", "content": "```python\n" + c["answer"].strip() + "\n```"},
                        {"role": "user", "content": f"[System Feedback]\n{d['feedback']}\n\nFirst state in one sentence what is wrong, then output the complete corrected function (with imports) in ONE ```python block."}])
        if len(tok(p).input_ids) > max_model_len - 512: continue
        prompts.append(p); meta.append(d["uid"])
    print(f"[regen] {len(prompts)} Case-C x {n}", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.code.temperature, max_tokens=max_tokens, n=n), use_tqdm=True)
    return [dict(uid=u, texts=[x.text for x in o.outputs]) for u, o in zip(meta, outs)]


def stage_score(cases, diff, regs):
    by = {c["uid"]: c for c in cases}; bd = {d["uid"]: d for d in diff}; n = Counter(c["label"] for c in cases)
    dist = Counter((d["label"], d["source"] or "silent") for d in diff)
    print("=== evidence sources ===")
    for lab in ("wrong", "correct"):
        print(f"  {lab:<8} n={n[lab]:>4}  spec-only {dist[(lab,'spec')]:>4}  diff-only {dist[(lab,'diff')]:>4}  spec+diff {dist[(lab,'spec+diff')]:>4}  silent {dist[(lab,'silent')]:>4}")
    sb = CodeSandbox(wall_timeout_s=20.0)
    def hidden(c, code):
        try: return bool(sb.evaluate_with_test_script(code, c["eval_test_code"], entry_point=c["entry_point"]).pass_at_1)
        except Exception: return False
    out = Counter()
    for r in regs:
        c = by[r["uid"]]; d = bd[r["uid"]]; was = (c["label"] == "correct")
        ctx = dict(question=c["question"], entry_point=c["entry_point"])
        codes = [_extract(t) for t in r["texts"]]
        # (1) best-of-n by spec examples
        spec_ok = [cd for cd in codes if cd.strip() and CPF.spec_example_evidence(ctx, cd) is None]
        pick = spec_ok[0] if spec_ok else None
        # (2) cluster gate on random inputs: the pick must agree with the majority of {orig, alt, pick}
        cluster_ok = True
        if pick is not None and d.get("inputs") and d.get("outs_orig") and d.get("outs_alt"):
            op = run_on_inputs(CPF.compose_program(c["question"], pick, d["entry"]), d["entry"], d["inputs"])
            if op is None: cluster_ok = False
            else:
                agree_alt = sum(a == b for a, b in zip(op, d["outs_alt"])); agree_orig = sum(a == b for a, b in zip(op, d["outs_orig"]))
                cluster_ok = max(agree_alt, agree_orig) >= 0.8 * len(op)   # pick sides with at least one of them on ≥80% inputs
        for g, acc, code in (("code1-gate", all(CPF.spec_example_evidence(ctx, cd) is None for cd in codes) and bool(codes[0].strip()), codes[0]),
                             ("best-of-n", pick is not None, pick or ""),
                             ("best-of-n+cluster", pick is not None and cluster_ok, pick or "")):
            now = hidden(c, code) if acc else was
            out[f"{c['label']}:{g}:rescue:{d['source']}"] += (not was and now); out[f"{c['label']}:{g}:broke:{d['source']}"] += (was and not now)
            out[f"{c['label']}:{g}:rescue"] += (not was and now); out[f"{c['label']}:{g}:broke"] += (was and not now)
    tot = n["wrong"] + n["correct"]
    print("\n=== repair acceptance rules (hidden tests) ===")
    for g in ("code1-gate", "best-of-n", "best-of-n+cluster"):
        r_, b_ = out[f"wrong:{g}:rescue"], out[f"correct:{g}:broke"]
        print(f"  {g:<18} failing: rescue {r_:>3}/{n['wrong']}   passing: broke {b_:>3}/{n['correct']}   pass@1 {n['correct']/tot:.1%} → {(n['correct']+r_-b_)/tot:.1%}"
              f"   [by source spec/diff/both: {out[f'wrong:{g}:rescue:spec']}/{out[f'wrong:{g}:rescue:diff']}/{out[f'wrong:{g}:rescue:spec+diff']} rescue, "
              f"{out[f'correct:{g}:broke:spec']}/{out[f'correct:{g}:broke:diff']}/{out[f'correct:{g}:broke:spec+diff']} broke]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["alt", "diff", "regen", "score"])
    ap.add_argument("--inputs", default="random", choices=["random", "mutate"], help="diff stage: random typed inputs or mutated spec examples")
    ap.add_argument("--cases-tag", default="code1"); ap.add_argument("--tag", default="code3")
    ap.add_argument("--policy", default=_P.models.policy_code); ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel)
    ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization); ap.add_argument("--max-model-len", type=int, default=_P.harness.code.max_model_len)
    ap.add_argument("--max-tokens", type=int, default=_P.harness.code.max_tokens); ap.add_argument("--n", type=int, default=_P.harness.code.n_best_of)
    a = ap.parse_args()
    cases = [json.loads(l) for l in (OUT / f"cases_{a.cases_tag}.jsonl").open()]
    disp = [json.loads(l) for l in (OUT / f"dispatch_{a.cases_tag}.jsonl").open()]
    P = lambda nm: OUT / f"{nm}_{a.tag}.jsonl"
    if a.stage == "alt":
        r = stage_alt(cases, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens); P("alt").write_text("".join(json.dumps(x) + "\n" for x in r)); return
    alts = [json.loads(l) for l in P("alt").open()]
    if a.stage == "diff":
        r = (stage_diff_mutate if a.inputs == "mutate" else stage_diff)(cases, alts, disp); P("diff").write_text("".join(json.dumps(x) + "\n" for x in r)); print("[diff] ->", P("diff")); return
    diff = [json.loads(l) for l in P("diff").open()]
    if a.stage == "regen":
        r = stage_regen(cases, diff, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens, a.n); P("regen").write_text("".join(json.dumps(x) + "\n" for x in r)); return
    stage_score(cases, diff, [json.loads(l) for l in P("regen").open()])



# ── iteration 2b: in-domain inputs = mutants of the spec's own example inputs ──
# Random typed inputs leave the spec's precondition domain (car_race_collision(-9))
# and two solutions legitimately disagree there, so the signal fires about as
# often on passing solutions as on failing ones
# passing — pure noise. Example inputs are in-domain by construction; small
# mutations stay close to it.
_CALL = re.compile(r">>>\s*([A-Za-z_]\w*)\s*\((.*)\)\s*(?:#.*)?$", re.M)
_ASSERT_CALL = re.compile(r"^\s*assert\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*==", re.M)


def _mutate(v, rng: random.Random):
    if isinstance(v, bool): return not v
    if isinstance(v, int): return v + rng.choice([-1, 1, 2, -2]) if rng.random() < 0.7 else (0 if v else 1)
    if isinstance(v, float): return round(v + rng.choice([-1.0, 0.5, 1.0]), 2)
    if isinstance(v, str):
        if not v: return "a"
        ops = [lambda s: s[:-1], lambda s: s + rng.choice("ab1 "), lambda s: s.upper(), lambda s: s[::-1], lambda s: s[1:]]
        return rng.choice(ops)(v)
    if isinstance(v, list):
        if not v: return [_mutate_seed(rng)]
        ops = [lambda l: l[:-1], lambda l: l + [l[-1]], lambda l: l[1:] + l[:1], lambda l: [_mutate(l[0], rng)] + l[1:], lambda l: l + [_mutate(l[-1], rng)]]
        return rng.choice(ops)(list(v))
    if isinstance(v, tuple): return tuple(_mutate(list(v), rng))
    if isinstance(v, dict):
        d = dict(v)
        if d: k = rng.choice(list(d)); d[k] = _mutate(d[k], rng)
        return d
    return v


def _mutate_seed(rng): return rng.choice([0, 1, -1, 2])


def mutated_inputs(spec: str, entry: str, k: int = 12, seed: int = 0):
    calls = [(n, a) for n, a in _CALL.findall(spec)] + [(n, a) for n, a in _ASSERT_CALL.findall(spec)]
    calls = [a for n, a in calls if not entry or n == entry]
    seeds = []
    for a in calls:
        try:
            v = ast.literal_eval(f"({a},)") if a.strip() else ()
            seeds.append(list(v))
        except Exception:
            continue
    if not seeds: return entry, []
    rng = random.Random(seed); out = []
    for s in seeds: out.append(list(s))                       # the examples themselves
    while len(out) < k:
        s = rng.choice(seeds); out.append([_mutate(x, rng) for x in s])
    return entry, out[:k]


def stage_diff_mutate(cases, alts, disp):
    """Same as stage_diff but inputs = spec example inputs + mutants."""
    by_alt = {a["uid"]: a["alt"] for a in alts}; by_d = {d["uid"]: d for d in disp}
    out = []
    for c in cases:
        d = by_d.get(c["uid"], {}); fb = d.get("feedback", ""); src = "spec" if fb else ""
        entry, inputs = mutated_inputs(c["question"], c["entry_point"])
        diff_ev = ""; outs_o = outs_a = None; alt = by_alt.get(c["uid"], "")
        if entry and inputs and alt:
            co = CPF.compose_program(c["question"], c["answer"], entry); ca = CPF.compose_program(c["question"], alt, entry)
            outs_o = run_on_inputs(co, entry, inputs); outs_a = run_on_inputs(ca, entry, inputs)
            if outs_o and outs_a:
                for args, oo, oa in zip(inputs, outs_o, outs_a):
                    if oo != oa and not (oo.startswith("EXC") and oa.startswith("EXC")):
                        diff_ev = (f"[differential_test @input] on the input {entry}(*{args!r}) (a small variation of the specification's "
                                   f"example) your code returns {oo[:80]} but an independently written solution returns {oa[:80]}; "
                                   f"decide from the specification which is right, then fix the function")
                        break
        if diff_ev and not fb: fb, src = diff_ev, "diff"
        elif diff_ev: fb = fb + "\n\n" + diff_ev; src = "spec+diff"
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if fb else "A", feedback=fb, source=src,
                        entry=entry, inputs=inputs, outs_orig=outs_o, outs_alt=outs_a))
    return out

if __name__ == "__main__":
    main()
