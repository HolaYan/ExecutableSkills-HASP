"""LLM-based anchor: locate the first erroneous step with a judge model, then
regenerate the policy's answer FROM that step. Offline harness over the mined
committed-wrong cases plus a committed-correct control set.

Why an LLM locator
------------------
Both cheap anchors were falsified by the control set:
  per-step arithmetic regex : 8.7% coverage on wrong vs 5.6% on correct
  answer-drift              : 11.8% on wrong vs 9.0% on correct
and the error taxonomy shows 67% of committed-wrong answers are conceptual
(wrong setup / case analysis / deduction), i.e. not visible as a literal
`a op b = c` slip. Localizing those needs a semantic judge.

Stages (all batched through vLLM; one GPU is enough for 8B judge + 4B policy)
  locate : judge reads (problem, numbered steps) and returns the FIRST step
           with a definite error, or NONE. No gold is shown.
  regen  : policy (Qwen3-4B-Instruct-2507, the model the rollouts came from)
           continues from the anchored step with the judge's reason injected.
           Ablation arm: same reason appended at the END of the full rollout
           (= today's Case-C geometry, but with a specific reason). This
           isolates "truncate at the anchor" from "evidence appended at end".
  score  : repo's own extractor + EM matcher; fallback-to-original when the
           regeneration commits nothing parseable.

Headline numbers
  wrong set  : locator fire rate, rescue(anchored), rescue(endnote)
  correct set: locator fire rate (false-positive proxy), broke(anchored),
               broke(endnote)   <- must stay ~0 or the method is unsafe

Usage
  python anchor/llm_locate.py --stage build
  python anchor/llm_locate.py --stage locate --judge Qwen/Qwen3-8B
  python anchor/llm_locate.py --stage regen  --policy Qwen/Qwen3-4B-Instruct-2507
  python anchor/llm_locate.py --stage score
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HASP))

from hasp_paths import rollouts_dir, setup_compile_caches  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402
_P = _protocol()

# A quota-limited home directory fills up with vLLM's torch.compile / Triton
# caches and jobs die with "Disk quota exceeded". Redirect every compile cache
# BEFORE vllm/torch are imported; setdefault so an explicit value from the job
# script still wins.
setup_compile_caches()

# Resolved lazily: only the corpus-building stage reads the raw rollouts,
# so scoring an existing corpus needs no upstream checkout at all.

from anchor.anchor import segment_steps  # noqa: E402
from pf_select.react_prompts import build_react_user_prompt  # noqa: E402
from verifiers.reference_em import (  # noqa: E402
    em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
)

_COMMIT = re.compile(r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:", re.I | re.M)
DS = ["aime24", "amc23", "math500", "gsm8k", "olympiadbench"]

OUT = _HASP / "data" / "llm_anchor"


# ── stage: build ─────────────────────────────────────────────────────────

def build_cases(control_per_ds: int, wrong_per_ds: int = -1) -> list[dict]:
    cases = []
    for f in sorted((_HASP / "data/error_cases").glob("*.jsonl")):
        kept_w = 0
        for line in f.open():
            if wrong_per_ds >= 0 and kept_w >= wrong_per_ds:
                break
            kept_w += 1
            r = json.loads(line)
            cases.append(dict(uid=f"W:{r['dataset']}:{r['qid']}:{r['rollout_idx']}",
                              label="wrong", dataset=r["dataset"], qid=r["qid"],
                              question=r["question"], gold=r["gold"],
                              pred=r["wrong_pred"], response=r["orig_response"]))
    for ds in DS:
        off = json.loads((rollouts_dir() / "skills_off" / f"{ds}_results.json").read_text())["results"]
        k = 0
        per_q = 1 if control_per_ds <= 60 else 3   # larger control sets: a few per question
        for q in off:
            got = 0
            for i, (p, r) in enumerate(zip(q["all_predictions"], q["all_responses"])):
                if _COMMIT.search(r) and _em_match_multi(p, str(q["gold"])):
                    cases.append(dict(uid=f"C:{ds}:{q['id']}:{i}", label="correct",
                                      dataset=ds, qid=q["id"], question=q["question"],
                                      gold=str(q["gold"]), pred=p, response=r))
                    k += 1; got += 1
                    if got >= per_q:
                        break
            if k >= control_per_ds:
                break
    return cases


# ── stage: locate ────────────────────────────────────────────────────────

LOCATE_TMPL = (
    "You are auditing a student's solution to a math problem. Your job is to "
    "find the FIRST step that contains a definite error — a wrong computation, "
    "an invalid deduction, a misread condition, a missed case that the "
    "argument relies on. Do not flag stylistic issues, redundancy, or steps "
    "that are merely unfinished. If every step is valid, answer NONE.\n\n"
    "Problem:\n{question}\n\n"
    "Solution, split into numbered steps:\n{steps}\n\n"
    "Answer in exactly this format:\n"
    "STEP: <number or NONE>\n"
    "REASON: <one or two sentences stating concretely what is wrong in that step>"
)

_STEP_RE = re.compile(r"STEP:\s*(NONE|\d+)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)


def _strip_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def _numbered_steps(response: str, max_chars: int = 36000) -> tuple[list, str]:
    steps = segment_steps(response[:max_chars])
    body = "\n".join(f"[{s.idx + 1}] {s.text.strip()}" for s in steps)
    return steps, body


def _chat(tok, content: str, thinking: bool) -> str:
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _parse_locate(ans: str, n_steps: int):
    m = _STEP_RE.search(ans); r = _REASON_RE.search(ans)
    step = None
    if m and m.group(1).upper() != "NONE":
        k = int(m.group(1)) - 1
        if 0 <= k < n_steps:
            step = k
    return step, (r.group(1).strip()[:400] if r else "")


def stage_locate(cases: list[dict], judge: str, tp: int, gpu_mem: float,
                 max_model_len: int, thinking: bool, votes: int = 1) -> list[dict]:
    """votes=1: one greedy locate. votes>1: additionally sample `votes` locates
    at T=0.6; `votes_steps` is stored so `score` can apply a consistency gate
    (all votes fire on the same step ±2). The greedy run supplies `step` and
    `reason` so older gates keep working unchanged."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=judge, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for c in cases:
        steps, body = _numbered_steps(c["response"])
        if not steps:
            continue
        p = _chat(tok, LOCATE_TMPL.format(question=c["question"], steps=body), thinking)
        if len(tok(p).input_ids) > max_model_len - 4096:
            continue
        prompts.append(p); meta.append((c, len(steps)))
    print(f"[locate] {len(prompts)}/{len(cases)} cases fit the judge context", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.locator.locate_temperature, max_tokens=_P.locator.locate_max_tokens), use_tqdm=True)
    vote_outs = None
    if votes > 1:
        vote_outs = llm.generate(prompts, SamplingParams(temperature=_P.locator.vote_temperature, max_tokens=_P.locator.locate_max_tokens, n=votes),
                                 use_tqdm=True)
    res = []
    for i, ((c, n_steps), o) in enumerate(zip(meta, outs)):
        ans = _strip_think(o.outputs[0].text if o.outputs else "")
        step, reason = _parse_locate(ans, n_steps)
        rec = dict(uid=c["uid"], label=c["label"], step=step, n_steps=n_steps,
                   reason=reason, raw=ans[-600:])
        if vote_outs is not None:
            rec["votes_steps"] = [_parse_locate(_strip_think(v.text), n_steps)[0]
                                  for v in vote_outs[i].outputs]
        res.append(rec)
    return res


# ── stage: arbiter ───────────────────────────────────────────────────────
#
# The 8B-judge run broke 5 correct answers even under the agreement gate; 3 of
# the 5 were the judge MISREADING the problem statement (claiming a condition
# the problem does state, or answering a different question). An arbiter that
# re-reads the problem and compares the ORIGINAL vs REGENERATED conclusions
# is the targeted fix: it must actively prefer the regeneration, otherwise we
# keep the original.

ARBITER_TMPL = (
    "Two attempts at the same math problem reach different final answers. "
    "Read the problem carefully, then decide which final answer is correct.\n\n"
    "Problem:\n{question}\n\n"
    "Attempt A — final answer: {ans_a}\nEnd of attempt A's reasoning:\n{tail_a}\n\n"
    "Attempt B — final answer: {ans_b}\nEnd of attempt B's reasoning:\n{tail_b}\n\n"
    "Answer in exactly this format:\nVERDICT: <A or B or UNSURE>\nREASON: <one sentence>"
)
_VERDICT_RE = re.compile(r"VERDICT:\s*(A|B|UNSURE)", re.I)


def stage_arbiter(cases: list[dict], regens: list[dict], judge: str, tp: int, gpu_mem: float,
                  max_model_len: int, thinking: bool) -> list[dict]:
    from vllm import LLM, SamplingParams
    by_uid = {c["uid"]: c for c in cases}
    llm = LLM(model=judge, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for r in regens:
        c = by_uid[r["uid"]]
        new = _extract_answer_math(r["text"])
        if not new or new.strip() == str(c["pred"]).strip():
            continue  # nothing to arbitrate
        p = _chat(tok, ARBITER_TMPL.format(
            question=c["question"], ans_a=c["pred"], tail_a=c["response"][-1500:],
            ans_b=new, tail_b=r["text"][-1500:]), thinking)
        if len(tok(p).input_ids) > max_model_len - 2048:
            continue
        prompts.append(p); meta.append((r["uid"], r["arm"], new))
    print(f"[arbiter] {len(prompts)} comparisons", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.locator.arbiter_temperature, max_tokens=_P.locator.arbiter_max_tokens), use_tqdm=True)
    res = []
    for (uid, arm, new), o in zip(meta, outs):
        ans = _strip_think(o.outputs[0].text if o.outputs else "")
        m = _VERDICT_RE.search(ans)
        res.append(dict(uid=uid, arm=arm, new=new, verdict=(m.group(1).upper() if m else "UNSURE")))
    return res


# ── stage: regen ─────────────────────────────────────────────────────────

ANCHOR_NOTE = (
    "\n\n[Anchor Check] A verification step found a concrete error at this point "
    "in the solution: {reason}\nThe work before this point has not been flagged. "
    "Redo the solution from here, fixing this error, and carry it through to the "
    "end. You MUST end with `Action: finish[<answer>]`.\n\n"
)
END_NOTE = (
    "\n\n[System Feedback]\n{reason}\n\nPlease provide a revised answer. "
    "You MUST end with `Action: finish[<answer>]`.\n\nRevised answer:\n"
)


def stage_regen(cases: list[dict], locs: list[dict], policy: str, tp: int, gpu_mem: float,
                max_model_len: int, max_tokens: int) -> list[dict]:
    from vllm import LLM, SamplingParams
    by_uid = {c["uid"]: c for c in cases}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for L in locs:
        if L["step"] is None:
            continue
        c = by_uid[L["uid"]]
        steps, _ = _numbered_steps(c["response"])
        if L["step"] >= len(steps):
            continue
        base = tok.apply_chat_template(
            [{"role": "user", "content": build_react_user_prompt(c["question"])}],
            tokenize=False, add_generation_prompt=True)
        prefix = c["response"][: steps[L["step"]].char_start].rstrip()
        reason = L["reason"] or "this step contains an error"
        anchored = base + prefix + ANCHOR_NOTE.format(reason=reason)
        endnote = base + c["response"].rstrip() + END_NOTE.format(reason=reason)
        for arm, p in (("anchored", anchored), ("endnote", endnote)):
            if len(tok(p).input_ids) > max_model_len - 512:
                continue
            prompts.append(p); meta.append((L["uid"], arm))
    print(f"[regen] {len(prompts)} generations ({len(locs)} located cases x 2 arms)", flush=True)
    sp = SamplingParams(temperature=_P.locator.regen_temperature, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [dict(uid=u, arm=a, text=(o.outputs[0].text if o.outputs else ""))
            for (u, a), o in zip(meta, outs)]


# ── stage: score ─────────────────────────────────────────────────────────

_NUM_ANS = re.compile(r"^[-+]?\d+(\.\d+)?(/\d+)?$")
_REFUSE = re.compile(r"insufficient|cannot be determined|not enough|undefined|no solution", re.I)
# True committed population in the base eval (for weighting rescue vs broke).
_POP_WRONG, _POP_CORRECT = 2300, 2724


def _norm(s) -> str:
    return (s or "").strip().strip("$").replace(",", "").rstrip(".")


def _atype(s) -> str:
    s = _norm(s)
    if not s:
        return "none"
    if _REFUSE.search(s):
        return "refuse"
    return "num" if _NUM_ANS.match(s) else "expr"


def _votes_consistent(L: dict, tol: int = 2) -> bool:
    vs = L.get("votes_steps")
    if not vs:
        return True            # no vote data -> gate is a no-op
    if any(v is None for v in vs):
        return False
    return max(vs) - min(vs) <= tol


def stage_score(cases: list[dict], locs: list[dict], regens: list[dict],
                arbs: list[dict] | None = None) -> None:
    by_uid = {c["uid"]: c for c in cases}
    loc_by = {L["uid"]: L for L in locs}
    fire = Counter(); n = Counter()
    for c in cases:
        n[c["label"]] += 1
        L = loc_by.get(c["uid"])
        if L and L["step"] is not None:
            fire[c["label"]] += 1
            if _votes_consistent(L):
                fire[c["label"] + ":voted"] += 1
    print("=== locator ===")
    for lab in ("wrong", "correct"):
        extra = f"   voted-consistent {fire[lab + ':voted']}" if any('votes_steps' in L for L in locs) else ""
        print(f"  {lab:<8} fired {fire[lab]}/{n[lab]} = {fire[lab] / max(1, n[lab]):.1%}{extra}")
    pos = [loc_by[c['uid']]['step'] / max(1, loc_by[c['uid']]['n_steps'])
           for c in cases if c['uid'] in loc_by and loc_by[c['uid']]['step'] is not None]
    if pos:
        pos.sort(); print(f"  anchor position median {pos[len(pos) // 2]:.0%} of steps")

    # per-uid answers for each arm + arbiter verdicts
    ans: dict[str, dict[str, str]] = {}
    for r in regens:
        ans.setdefault(r["uid"], {})[r["arm"]] = _extract_answer_math(r["text"]) or ""
    arb: dict[tuple, str] = {(a["uid"], a["arm"]): a["verdict"] for a in (arbs or [])}
    have_arb = bool(arbs)

    out = Counter()
    for uid, arms in ans.items():
        c = by_uid[uid]; was = _em_match_multi(c["pred"], c["gold"])
        for arm, new in arms.items():
            key = f"{c['label']}:{arm}"
            final = new if new else c["pred"]
            out[key + ":n"] += 1
            ok = _em_match_multi(final, c["gold"])
            if not was and ok: out[key + ":rescue"] += 1
            if was and not ok: out[key + ":broke"] += 1
    print("\n=== regeneration, no gate (fallback only on unparseable) ===")
    for lab in ("wrong", "correct"):
        for arm in ("anchored", "endnote"):
            k = f"{lab}:{arm}"
            if out[k + ":n"]:
                print(f"  {lab:<8} {arm:<9} n={out[k + ':n']:>4}  rescue={out[k + ':rescue']:>3}  broke={out[k + ':broke']:>3}")

    # ── gate chain ──
    def gated(uid: str, use_votes: bool, use_arb: bool) -> str:
        c = by_uid[uid]; arms = ans.get(uid, {})
        a, e = arms.get("anchored", ""), arms.get("endnote", "")
        if not a or _norm(a) != _norm(e):                       # agreement
            return c["pred"]
        if _atype(a) != _atype(c["pred"]) or _atype(a) == "refuse":   # type
            return c["pred"]
        if use_votes and not _votes_consistent(loc_by.get(uid, {})):
            return c["pred"]
        if use_arb:
            v = arb.get((uid, "anchored")) or arb.get((uid, "endnote"))
            if v != "B":
                return c["pred"]
        return a

    print("\n=== gate chain (agreement + type [+ votes] [+ arbiter]) ===")
    print(f"  {'gate':<28}{'rescue':>7}{'broke':>7}{'rescue%':>9}{'broke%':>8}{'net (pop-weighted)':>20}")
    configs = [("agree+type", False, False)]
    if any('votes_steps' in L for L in locs):
        configs.append(("agree+type+votes", True, False))
    if have_arb:
        configs.append(("agree+type+arbiter", False, True))
        if any('votes_steps' in L for L in locs):
            configs.append(("agree+type+votes+arbiter", True, True))
    for name, uv, ua in configs:
        res = br = 0
        for uid in ans:
            c = by_uid[uid]
            was = _em_match_multi(c["pred"], c["gold"]); now = _em_match_multi(gated(uid, uv, ua), c["gold"])
            res += (not was and now); br += (was and not now)
        rr, bb = res / max(1, n["wrong"]), br / max(1, n["correct"])
        print(f"  {name:<28}{res:>7}{br:>7}{rr:>9.1%}{bb:>8.1%}{rr * _POP_WRONG - bb * _POP_CORRECT:>+17.0f} rollouts")
    print("\n  (compare against pf_select's own rescue / broke on this population)")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "locate", "regen", "arbiter", "score"])
    ap.add_argument("--votes", type=int, default=_P.locator.votes, help="locate: extra sampled votes (T=0.6)")
    ap.add_argument("--arb-tag", default="", help="suffix for arbiter output (cross-model arbiters)")
    ap.add_argument("--cases-tag", default="", help="alternate case set (e.g. 'ctrl600'): cases_<tag>.jsonl")
    ap.add_argument("--wrong-per-ds", type=int, default=_P.locator.wrong_per_ds,
                    help="build: cap committed-wrong cases per dataset (-1 = all mined; 0 = none)")
    ap.add_argument("--judge", default=_P.models.judge)
    ap.add_argument("--policy", default=_P.models.policy_math)
    ap.add_argument("--tag", default="q8b", help="run tag (judge identity)")
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel)
    ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization)
    ap.add_argument("--max-model-len", type=int, default=_P.locator.max_model_len)
    ap.add_argument("--max-tokens", type=int, default=_P.locator.max_tokens)
    ap.add_argument("--control-per-ds", type=int, default=_P.locator.control_per_ds)
    ap.add_argument("--no-thinking", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ct = f"_{a.cases_tag}" if a.cases_tag else ""
    cases_p = OUT / f"cases{ct}.jsonl"
    loc_p = OUT / f"locate_{a.tag}{ct}.jsonl"
    reg_p = OUT / f"regen_{a.tag}{ct}.jsonl"

    if a.stage == "build":
        cases = build_cases(a.control_per_ds, a.wrong_per_ds)
        cases_p.write_text("".join(json.dumps(c) + "\n" for c in cases))
        print(f"[build] {len(cases)} cases ({Counter(c['label'] for c in cases)}) -> {cases_p}")
        return
    arb_p = OUT / (f"arbiter_{a.tag}{('_' + a.arb_tag) if a.arb_tag else ''}{ct}.jsonl")
    cases = [json.loads(l) for l in cases_p.open()]
    if a.stage == "locate":
        locs = stage_locate(cases, a.judge, a.tp, a.gpu_mem, a.max_model_len,
                            not a.no_thinking, votes=a.votes)
        loc_p.write_text("".join(json.dumps(x) + "\n" for x in locs))
        print(f"[locate] -> {loc_p}")
        return
    locs = [json.loads(l) for l in loc_p.open()]
    if a.stage == "regen":
        regs = stage_regen(cases, locs, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens)
        reg_p.write_text("".join(json.dumps(x) + "\n" for x in regs))
        print(f"[regen] -> {reg_p}")
        return
    regs = [json.loads(l) for l in reg_p.open()] if reg_p.exists() else []
    if a.stage == "arbiter":
        arbs = stage_arbiter(cases, regs, a.judge, a.tp, a.gpu_mem, a.max_model_len, not a.no_thinking)
        arb_p.write_text("".join(json.dumps(x) + "\n" for x in arbs))
        print(f"[arbiter] -> {arb_p}")
        return
    arbs = [json.loads(l) for l in arb_p.open()] if arb_p.exists() else None
    stage_score(cases, locs, regs, arbs)


if __name__ == "__main__":
    main()
