"""Offline evaluation of the web evidence PFs by RE-ANSWERING FROM EXISTING EVIDENCE.

No live search: the trajectory's observations are replayed, so this measures
exactly the evidence_present channel (gold already inside an observation —
45% of wrong rollouts) plus answer_grounding_check.

build   (CPU)   episodes from Skills_Agent skill_eval_best (2Wiki / HotpotQA /
                Bamboogle × baseline / skills_top10 / +prompt) → correct / wrong
                by cover-EM of the parsed final answer vs gold_answers.
dispatch(CPU)   answer_grounding_check (deterministic) → Case C with evidence.
locate  (8B)    helper evidence for evidence_answer_consistency: given the
                question, the observations and the candidate answer, quote a
                passage that answers the question differently, or OK.
regen   (7B)    policy re-answers: [user: question + observations]
                [assistant: candidate] [user: [System Feedback] evidence →
                final answer only]  2 samples T=0.7.
score   (CPU)   gate = both samples agree ∧ new answer is grounded in the
                observations (answer_grounding silent); else keep original.
                rescue on wrong, broke on correct, cover-EM before/after.
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
_spec = importlib.util.spec_from_file_location("web_evidence", str(_HASP / "skills/executable/web/checkers.py"))
WPF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(WPF)

from hasp_paths import web_episodes_dir  # noqa: E402
ROOT = web_episodes_dir()
OUT = _HASP / "data" / "web_polished"
_GIVEUP = re.compile(r"^\s*$|unknown|cannot (?:be )?determine|not (?:enough|sufficient)|unable to|no information", re.I)


def _parse_final(f):
    s = str(f or "")
    if s.strip().startswith("{"):
        try: return str(ast.literal_eval(s).get("answer", ""))
        except Exception:
            m = re.search(r"'answer':\s*['\"](.+?)['\"],\s*'reasoning'", s); return m.group(1) if m else s
    return s


def cover_em(pred, golds):
    p = " ".join(WPF._norm(pred)); return bool(p) and any(" ".join(WPF._norm(g)) in p for g in golds)


def _trace_ctx(r):
    hist = []
    for s in r["trace"]:
        a = str(s.get("action", "")); m = re.match(r"\s*(SEARCH|READ|FINAL)\s*[\[\(](.*)[\]\)]\s*$", a, re.S)
        hist.append(dict(action=(m.group(1) if m else a[:10]), query=(m.group(2) if m else ""), observation=str(s.get("observation", ""))))
    return hist


def build_cases(cap_per_file: int) -> list[dict]:
    cases = []
    for f in sorted(ROOT.glob("*/*/base_clean_episodes.jsonl")):
        ds, setting = f.parts[-3], f.parts[-2]; k = Counter()
        for r in map(json.loads, f.open()):
            golds = r["gold_answers"]; golds = ast.literal_eval(golds) if isinstance(golds, str) else golds
            ans = _parse_final(r.get("final")); hist = _trace_ctx(r)
            obs = " ".join(h["observation"] for h in hist)
            if not obs.strip() or _GIVEUP.match(ans):
                continue
            lab = "correct" if cover_em(ans, golds) else "wrong"
            if k[lab] >= cap_per_file:
                continue
            k[lab] += 1
            cases.append(dict(uid=f"{lab[0].upper()}:{ds}:{setting}:{r['sample_id']}", label=lab, dataset=ds, setting=setting,
                              question=r["question"], golds=list(golds), answer=ans, history=hist,
                              gold_in_obs=any(WPF._contains(obs, g) for g in golds)))
    return cases


def _ctx(c):
    return dict(question=c["question"], action_history=c["history"], all_read_contents=" ".join(h["observation"] for h in c["history"]), has_read=any(h["action"] == "READ" for h in c["history"]))


def stage_dispatch(cases):
    out = []
    for c in cases:
        v = WPF.answer_grounding(_ctx(c), c["answer"])
        out.append(dict(uid=c["uid"], label=c["label"], case="C" if v else "A", feedback=(f"[answer_grounding_check @final] {v}" if v else "")))
    return out


LOCATE_TMPL = (
    "You audit a web-search agent's answer using ONLY the evidence it retrieved.\n\n"
    "Question:\n{question}\n\nRetrieved evidence (search results and pages, in order):\n{obs}\n\n"
    "Candidate answer: {answer}\n\n"
    "If a passage in the evidence explicitly answers the question with something DIFFERENT from the candidate, "
    "quote that passage (verbatim, ≤ 40 words) and state the answer it gives. If the evidence does not contradict the "
    "candidate, or does not answer the question at all, answer OK.\n\n"
    "Format:\nVERDICT: <ISSUE or OK>\nREASON: <quoted passage → the answer it supports>"
)
_V = re.compile(r"VERDICT:\s*(ISSUE|OK)", re.I); _R = re.compile(r"REASON:\s*(.+)", re.I | re.S)


def stage_locate(cases, disp, judge, tp, gpu_mem, max_model_len, thinking):
    from vllm import LLM, SamplingParams
    llm = LLM(model=judge, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); prompts, meta = [], []
    for c in cases:
        obs = "\n\n".join(f"[{i+1}] {h['action']} {h['query'][:80]}\n{h['observation'][:2500]}" for i, h in enumerate(c["history"]))[-14000:]
        msgs = [{"role": "user", "content": LOCATE_TMPL.format(question=c["question"], obs=obs, answer=c["answer"][:200])}]
        try: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        except TypeError: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tok(p).input_ids) > max_model_len - 2048:
            continue
        prompts.append(p); meta.append(c["uid"])
    print(f"[locate] {len(prompts)} audits", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.web.locate_temperature, max_tokens=_P.harness.web.locate_max_tokens), use_tqdm=True)
    res = []
    for u, o in zip(meta, outs):
        ans = (o.outputs[0].text if o.outputs else "").split("</think>")[-1]
        v = _V.search(ans); r = _R.search(ans)
        res.append(dict(uid=u, verdict=(v.group(1).upper() if v else "OK"), reason=(r.group(1).strip()[:400] if r else "")))
    return res


def stage_regen(cases, disp, locs, policy, tp, gpu_mem, max_model_len, n):
    from vllm import LLM, SamplingParams
    by = {c["uid"]: c for c in cases}; lb = {l["uid"]: l for l in locs}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); prompts, meta = [], []
    for d in disp:
        c = by[d["uid"]]; fb = d["feedback"]
        L = lb.get(d["uid"])
        if L and L["verdict"] == "ISSUE" and L["reason"]:
            fb = (fb + "\n\n" if fb else "") + f"[evidence_answer_consistency @final] the retrieved evidence answers the question differently: {L['reason']}"
        if not fb:
            continue
        obs = "\n\n".join(f"[{i+1}] {h['action']} {h['query'][:80]}\n{h['observation'][:2500]}" for i, h in enumerate(c["history"]))[-12000:]
        msgs = [{"role": "user", "content": f"Answer the question using the retrieved evidence.\n\nQuestion: {c['question']}\n\nEvidence:\n{obs}\n\nGive the final answer only."},
                {"role": "assistant", "content": c["answer"]},
                {"role": "user", "content": f"[System Feedback]\n{fb}\n\nReconsider using only the evidence above. Reply with the final answer only (a short phrase), nothing else."}]
        try: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tok(p).input_ids) > max_model_len - 256:
            continue
        prompts.append(p); meta.append((d["uid"], "both" if (d["case"] == "C" and L and L["verdict"] == "ISSUE") else ("ground" if d["case"] == "C" else "helper")))
    print(f"[regen] {len(prompts)} re-answers x {n}", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.harness.web.temperature, max_tokens=_P.harness.web.answer_max_tokens, n=n), use_tqdm=True)
    return [dict(uid=u, source=src, texts=[x.text.strip() for x in o.outputs]) for (u, src), o in zip(meta, outs)]


def stage_score(cases, disp, locs, regs):
    by = {c["uid"]: c for c in cases}; n = Counter(c["label"] for c in cases)
    print("=== population ===")
    for lab in ("wrong", "correct"):
        g = sum(1 for c in cases if c["label"] == lab and c["gold_in_obs"])
        print(f"  {lab:<8} n={n[lab]:>4}  gold inside an observation: {g} ({g/max(1,n[lab]):.0%})")
    dist = Counter((d["label"], d["case"]) for d in disp); lv = Counter((by[l["uid"]]["label"], l["verdict"]) for l in locs)
    print("=== fire rates ===")
    for lab in ("wrong", "correct"):
        print(f"  {lab:<8} answer_grounding_check {dist[(lab,'C')]:>4} ({dist[(lab,'C')]/max(1,n[lab]):.1%})   helper ISSUE {lv[(lab,'ISSUE')]:>4} ({lv[(lab,'ISSUE')]/max(1,n[lab]):.1%})")
    out = Counter()
    for r in regs:
        c = by[r["uid"]]; was = (c["label"] == "correct")
        a = [t.splitlines()[0].strip().strip('."') if t else "" for t in r["texts"]]; a0 = a[0]
        ctx = _ctx(c)
        gate = bool(a0) and all(" ".join(WPF._norm(x)) == " ".join(WPF._norm(a0)) for x in a) and WPF.answer_grounding(ctx, a0) is None
        for g, acc in (("nogate", bool(a0)), ("gate", gate)):
            fin = a0 if acc else c["answer"]; now = cover_em(fin, c["golds"])
            out[f"{c['label']}:{g}:rescue:{r['source']}"] += (not was and now); out[f"{c['label']}:{g}:broke:{r['source']}"] += (was and not now)
            out[f"{c['label']}:{g}:rescue"] += (not was and now); out[f"{c['label']}:{g}:broke"] += (was and not now)
    print("=== re-answer from evidence (Qwen2.5-7B-Instruct), cover-EM ===")
    for g in ("nogate", "gate"):
        print(f"  {g:<7} wrong: rescue {out[f'wrong:{g}:rescue']:>3}/{n['wrong']}   correct: broke {out[f'correct:{g}:broke']:>3}/{n['correct']}   "
              f"[by source — ground: {out[f'wrong:{g}:rescue:ground']}/{out[f'correct:{g}:broke:ground']}, helper: {out[f'wrong:{g}:rescue:helper']}/{out[f'correct:{g}:broke:helper']}, both: {out[f'wrong:{g}:rescue:both']}/{out[f'correct:{g}:broke:both']}]")
    tot = n["wrong"] + n["correct"]
    for g in ("nogate", "gate"):
        d = out[f"wrong:{g}:rescue"] - out[f"correct:{g}:broke"]
        print(f"  cover-EM on this population: {n['correct']/tot:.1%} → {(n['correct']+d)/tot:.1%}  ({g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "dispatch", "locate", "regen", "score"])
    ap.add_argument("--tag", default="web1"); ap.add_argument("--cap-per-file", type=int, default=_P.harness.web.cap_per_file)
    ap.add_argument("--judge", default=_P.models.judge); ap.add_argument("--policy", default=_P.models.policy_code)
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel); ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization)
    ap.add_argument("--max-model-len", type=int, default=_P.harness.web.max_model_len); ap.add_argument("--n", type=int, default=_P.harness.web.n)
    ap.add_argument("--no-thinking", action="store_true")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    P = lambda nm: OUT / f"{nm}_{a.tag}.jsonl"
    if a.stage == "build":
        cs = build_cases(a.cap_per_file); P("cases").write_text("".join(json.dumps(x) + "\n" for x in cs))
        print(f"[build] {len(cs)} cases {Counter(c['label'] for c in cs)} -> {P('cases')}"); return
    cases = [json.loads(l) for l in P("cases").open()]
    if a.stage == "dispatch":
        d = stage_dispatch(cases); P("dispatch").write_text("".join(json.dumps(x) + "\n" for x in d)); print("[dispatch] ->", P("dispatch")); return
    disp = [json.loads(l) for l in P("dispatch").open()]
    if a.stage == "locate":
        l = stage_locate(cases, disp, a.judge, a.tp, a.gpu_mem, a.max_model_len, not a.no_thinking)
        P("locate").write_text("".join(json.dumps(x) + "\n" for x in l)); return
    locs = [json.loads(l) for l in P("locate").open()] if P("locate").exists() else []
    if a.stage == "regen":
        r = stage_regen(cases, disp, locs, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.n)
        P("regen").write_text("".join(json.dumps(x) + "\n" for x in r)); return
    stage_score(cases, disp, locs, [json.loads(l) for l in P("regen").open()] if P("regen").exists() else [])


if __name__ == "__main__":
    main()
