"""Step-level dual-consent PF intervention (the user's design, 2026-08-22).

At EVERY ReAct action boundary of a rollout, two independent parties decide
whether a skill is needed at this step:

    anchor side : the PF's own step-level trigger — deterministic checkers
                  (arithmetic / power / mod / bounds) and family regexes —
                  evaluated on THIS step, not on the whole response at FINAL;
    model side  : the policy itself, shown the trajectory so far + this step +
                  the PF menu, selects <pf>id</pf> tags.

Only when BOTH agree (the model selects a PF whose anchor fired on this step)
do we act: a family-scoped locator produces concrete evidence for the step,
and the policy regenerates FROM this step with the evidence injected. The
result still goes through verify-and-fallback (two regeneration samples must
agree on a same-type answer; otherwise the original answer stands).

Superset constraint (user, 2026-08-22): pf_select already helps (stall
rescue channel), so this design must not regress it.
The step-level gate therefore applies only at action boundaries BEFORE the
final one; the FINAL boundary keeps today's pf_select dispatch unchanged
(including continuing stalled rollouts). Step-level interventions are
additive and fall back to the original trajectory whenever the gate fails,
so the existing gains are a floor, not a trade-off.

Offline harness over the paired base-model rollouts: the original trajectory
is REPLAYED step by step (no generation needed for the unmodified path); the
policy is only called for consent and for the regeneration branch.

Stages (each batched through vLLM; one L40S):
  anchor   CPU   — split rollouts into ReAct steps, run step triggers
  consent  4B    — model selection at each anchor-fired step (capped/rollout)
  evidence 8B    — family-scoped locate on the first consented steps
  regen    4B    — regenerate from the evidenced step, 2 samples
  score    CPU   — fire rates (anchor / consent / evidence) on wrong vs
                   correct, rescue / broke under the gate, step positions

PF skills, optimised form (v1, in-code table; SKILL.md port later): each PF
is (step trigger, family scope for the locator, execution = evidence +
regenerate). Three new families fill the measured blind spots: counting,
problem-condition recheck, geometry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HASP))
from hasp_paths import setup_compile_caches  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402
_P = _protocol()
setup_compile_caches()   # before torch/vllm are imported

from anchor.anchor import Step, check_arithmetic, check_bounds  # noqa: E402
from anchor.pf_checkers import (check_compute_observation, check_counting_trigger, check_interval_sign,  # noqa: E402
                                check_unique_solution_trigger, check_unsupported_final)
from anchor.sandbox import run_python  # noqa: E402
from pf_select.pf_select_eval import _PF_SELECT_INSTRUCTION_TMPL, _parse_pf_selection  # noqa: E402
from pf_select.react_prompts import build_react_user_prompt  # noqa: E402
from verifiers.reference_em import (  # noqa: E402
    em_match_multi as _em_match_multi, extract_answer_math as _extract_answer_math,
)

OUT = _HASP / "data" / "step_gate"


# ── optimised PF skills (v1 table) ───────────────────────────────────────
# trigger: regex on the step text (anchor side, cheap). checker: deterministic
# evidence producer (if it fires, the evidence is concrete and no LLM locate is
# needed). scope: instruction for the family-scoped LLM locator.
PF_SKILLS = {
    # ── hand-written from the wrong-case review (deterministic anchors) ──
    "compute_observation_verify": dict(
        trigger=re.compile(r"Action:\s*compute\[", re.I),
        checker=check_compute_observation,
        scope="Re-evaluate the compute[] expression in this step; flag only if the written Observation is numerically wrong, and state the true value.",
        summary="Re-check the value you wrote as the Observation of a compute[] action."),
    "unsupported_final_answer": dict(
        trigger=re.compile(r"Action:\s*finish\s*\[", re.I),
        checker=check_unsupported_final,
        scope="Flag only if the final answer is guessed / taken from a 'known result' instead of derived in the work above.",
        summary="Refuse to finish with an answer that was guessed rather than derived."),
    "counting_small_case_check": dict(
        trigger=re.compile(r"\\binom|\bC\(|[a-zA-Z]\s*!|\^\s*\{?[a-zA-Z]", re.I),
        checker=check_counting_trigger,        # anchor-side only; evidence via enumeration
        scope="(enumeration) test the counting formula stated in this step on small instances",
        summary="Test the counting formula in this step by brute force on a small case."),
    "interval_sign_check": dict(
        trigger=re.compile(r"[a-zA-Z]\s*(?:<|>|\\le|\\ge|≤|≥)\s*-?\d|-?\d\s*(?:<|\\le|≤)\s*[a-zA-Z]"),
        checker=check_interval_sign,
        scope="Only flag this step if a sign claimed for an expression on a stated interval is wrong; give a test point and the actual sign.",
        summary="Check the sign claims in this step's interval analysis at a test point."),
    "claimed_unique_solution_search": dict(
        trigger=re.compile(r"only (?:possible )?solution|the only solutions?|no other solutions?|only when\b|unique solution", re.I),
        checker=check_unique_solution_trigger,   # anchor-side only; evidence via counterexample search
        scope="(search) look for solutions the step claims do not exist",
        summary="Search small ranges for solutions this step claims do not exist."),
    "unsupported_known_result": dict(
        trigger=re.compile(r"known (?:result|fact|formula|problem|theorem)|well-known|it is known that|standard result|by a (?:known|classical) (?:result|theorem)", re.I),
        checker=None,
        scope="Only flag this step if the 'known result' it invokes is false, misstated, or does not apply under the problem's conditions; state the correct statement.",
        summary="Verify that a 'known result' cited in this step is true and applicable here."),
    "arithmetic_slip": dict(
        trigger=re.compile(r"\d\s*[+\-*/×÷]\s*\d|\\frac|\\times|\\cdot|=\s*-?\d"),
        checker=check_arithmetic,
        scope="Only flag this step if its literal arithmetic (sum, product, fraction, power, modular value) is wrong; recompute it and state the correct value.",
        summary="Catch arithmetic slips in this step's computation."),
    "algebraic_sign_error": dict(
        trigger=re.compile(r"[a-z]\s*[\^²³]|\bexpand|\bfactor|\bsimplif|distribut|\(-|-\(|\bsign\b", re.I),
        checker=None,
        scope="Only flag this step if an algebraic manipulation (sign, distribution, expansion, factoring, substitution) is invalid; state the correct expression.",
        summary="Catch sign / expansion / factoring errors in this step."),
    "boundary_violation": dict(
        trigger=re.compile(r"probabilit|\bsin\b|\bcos\b|must be (?:positive|an integer|at least|at most)|\brange\b|\bdomain\b", re.I),
        checker=check_bounds,
        scope="Only flag this step if a quantity violates a hard constraint (range, integrality, positivity, domain); quote the constraint.",
        summary="Catch values outside their allowed range in this step."),
    "case_incompleteness": dict(
        trigger=re.compile(r"\bcase\b|\bcases\b|\bif\b.*\bthen\b|\bwhen\b|\beither\b|\bwlog\b|without loss", re.I),
        checker=None,
        scope="Only flag this step if it omits a case the argument depends on, or assumes WLOG invalidly; name the missing case.",
        summary="Catch missing cases in this step's case analysis."),
    "counting_overcount": dict(
        trigger=re.compile(r"\\binom|\bchoose\b|\bC\(\d|\bways\b|\barrang|\bpermut|\bcombinat|\bcount", re.I),
        checker=None,
        scope="Only flag this step if it overcounts or undercounts (wrong binomial, ignores ordering/symmetry, double counts); state the correct count.",
        summary="Catch over/under-counting in this step's combinatorics."),
    "problem_condition_recheck": dict(
        trigger=re.compile(r"\bassume|\bgiven\b|\bthe problem\b|\bwe are told\b|\bcondition|\bconstraint|\bmust\b", re.I),
        checker=None,
        scope="Only flag this step if it uses a condition that the problem does NOT state, or contradicts one it does state; quote the problem text.",
        summary="Check this step against the problem's stated conditions."),
    "geometry_relation_check": dict(
        trigger=re.compile(r"\bangle|\btriangle|\bcircle|\bradius|\btangent|\bperpendicular|\bparallel|\barea\b|\bcoordinate", re.I),
        checker=None,
        scope="Only flag this step if a geometric relation it asserts (similarity, tangency, angle, length, area formula) is false; state the correct relation.",
        summary="Catch false geometric relations asserted in this step."),
}


def pf_menu() -> str:
    return "\n".join(f"- {sid}: {d['summary']}" for sid, d in PF_SKILLS.items())


# ── ReAct step splitting ─────────────────────────────────────────────────

_STEP_START = re.compile(r"(?m)^(?=(?:Thought|Action|Observation)\s*:)")
_ACTION = re.compile(r"(?m)^Action\s*:\s*(\w+)\s*\[")


def split_react_steps(text: str, min_len: int = _P.segmentation.production_min_len) -> list[Step]:
    """Reasoning steps at action granularity.

    The base model usually writes ONE huge `Thought:` block (the whole
    derivation) followed by a single Action, so splitting on ReAct markers
    alone yields ~2 steps per rollout (measured median = 2) and the gate would
    degenerate to FINAL-level again. We therefore cut at ReAct markers AND at
    paragraph boundaries inside a block, then merge short fragments forward
    so a step is a self-contained reasoning move (~350+ chars)."""
    from anchor.anchor import segment_steps
    cuts = {0, len(text)}
    for m in _STEP_START.finditer(text):
        cuts.add(m.start())
    for st in segment_steps(text, min_len=_P.segmentation.union_min_len):
        cuts.add(st.char_start)
    pts = sorted(cuts)
    raw = [(a, b) for a, b in zip(pts, pts[1:]) if b > a]
    merged: list[tuple[int, int]] = []
    for a, b in raw:
        if merged and (merged[-1][1] - merged[-1][0]) < min_len:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [Step(i, a, b, text[a:b]) for i, (a, b) in enumerate(merged)]


_MATHY = re.compile(r"\\d|=|\\\\[a-z]+|[+\-*/×÷^]")


# ── stage: anchor ────────────────────────────────────────────────────────

def stage_anchor(cases: list[dict], max_per_rollout: int) -> list[dict]:
    """Per rollout: steps where some PF trigger fires (+ deterministic evidence
    if a checker fires). The final step (the one that emits finish[]) is
    included — an error there is still 'this step'."""
    out = []
    for c in cases:
        steps = split_react_steps(c["response"])
        hits = []
        for st in steps:
            fired = []; ev = None; needs_enum = False
            if not _MATHY.search(st.text):      # prose-only step: no PF applies
                continue
            for sid, d in PF_SKILLS.items():
                if d["trigger"].search(st.text):
                    fired.append(sid)
                    if d["checker"] is not None and ev is None:
                        try:
                            e = d["checker"](st.text, c["response"], st.char_start)   # hand-written checkers
                        except TypeError:
                            e = d["checker"](st)                                      # anchor.py checkers (Step)
                        if e is not None:
                            if isinstance(e, dict) and e.get("verdict") is None:
                                needs_enum = e.get("enum_kind", "count")   # evidence must be executed
                            else:
                                ev = dict(pf=sid, verdict=(e["verdict"] if isinstance(e, dict) else e.verdict),
                                          fix=(e.get("fix") if isinstance(e, dict) else None))
            if fired:
                hits.append(dict(step=st.idx, pfs=fired, det_evidence=ev, needs_enum=needs_enum))
        # cap, but never drop the final step (the finish[] step is where
        # unsupported_final_answer anchors)
        kept = hits[:max_per_rollout]
        if hits and hits[-1]["step"] == len(steps) - 1 and hits[-1] not in kept:
            kept = kept[:-1] + [hits[-1]]
        out.append(dict(uid=c["uid"], label=c["label"], n_steps=len(steps), hits=kept))
    return out


# ── stage: consent (policy) ──────────────────────────────────────────────

def _chat(tok, content: str, thinking: bool = False) -> str:
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


CONSENT_TMPL = (
    "You are solving this problem step by step.\n\nProblem:\n{question}\n\n"
    "Your solution so far (earlier steps, abbreviated):\n{prefix}\n\n"
    "The step you just wrote:\n{step}\n\n"
    "Before continuing, you may invoke a program-function (PF) check on THIS step. "
    "Each PF inspects one failure mode of the step and, if it finds a concrete error, "
    "lets you redo the step.\n\nAvailable PFs (id: summary):\n{menu}\n\n"
    "Select the PFs that are genuinely relevant to this step. Output ONLY the "
    "selections, one tag per PF, in the form:\n<pf>skill_id</pf>\n"
    "If no check is needed for this step, output nothing.\n\nSelection:"
)


def stage_consent(cases, anchors, policy, tp, gpu_mem, max_model_len) -> list[dict]:
    from vllm import LLM, SamplingParams
    by_uid = {c["uid"]: c for c in cases}
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer(); menu = pf_menu()
    prompts, meta = [], []
    for a in anchors:
        c = by_uid[a["uid"]]; steps = split_react_steps(c["response"])
        for h in a["hits"]:
            st = steps[h["step"]]
            prefix = c["response"][: st.char_start]
            prefix = (prefix[-2500:] if len(prefix) > 2500 else prefix)
            p = _chat(tok, CONSENT_TMPL.format(question=c["question"], prefix=prefix.strip() or "(start)",
                                               step=st.text.strip()[:2500], menu=menu))
            if len(tok(p).input_ids) > max_model_len - 256:
                continue
            prompts.append(p); meta.append((a["uid"], h["step"], h["pfs"]))
    print(f"[consent] {len(prompts)} (rollout, step) queries", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.step_gate.consent_temperature, max_tokens=_P.step_gate.consent_max_tokens), use_tqdm=True)
    allowed = set(PF_SKILLS)
    res = []
    for (uid, step, fired), o in zip(meta, outs):
        sel = _parse_pf_selection(o.outputs[0].text if o.outputs else "", allowed)
        both = sorted(set(sel) & set(fired))
        res.append(dict(uid=uid, step=step, anchor_pfs=fired, model_pfs=sel, agreed=both))
    return res


# ── stage: evidence (judge, family-scoped, on THIS step) ─────────────────

EVIDENCE_TMPL = (
    "You are auditing ONE step of a student's math solution.\n{scope}\n"
    "Do not flag stylistic issues or an unfinished step. If the step is fine, answer OK.\n\n"
    "Problem:\n{question}\n\nSolution before this step:\n{prefix}\n\nThe step under audit:\n{step}\n\n"
    "Answer in exactly this format:\nVERDICT: <ISSUE or OK>\nREASON: <one or two sentences stating concretely what is wrong and what the correct value/relation is>"
)
CODE_TMPL = (
    "A student's solution step states a counting formula. Write a SHORT python snippet that tests it "
    "on small instances by direct enumeration of the objects the PROBLEM defines (do not re-derive a "
    "formula; enumerate with itertools).\n\nProblem:\n{question}\n\nThe step:\n{step}\n\n"
    "Define two functions and print a comparison:\n"
    "  def claimed(n): return <the step's formula evaluated at n>\n"
    "  def brute(n):   return <direct enumeration count for the same objects at n>\n"
    "for n in SMALL: print(n, claimed(n), brute(n))\n"
    "Use SMALL = 3 or 4 small values where enumeration is cheap. If the step's count is not a function "
    "of one parameter, pick the natural small parameter (board size, number of elements, etc.). "
    "Output ONLY a ```python code block, nothing else."
)
SEARCH_TMPL = (
    "A student's solution step claims that certain solutions are the ONLY ones (or that no other "
    "solutions exist). Write a SHORT python snippet that searches a small range by brute force for "
    "solutions of the problem's condition and prints any solution NOT covered by the step's claim.\n\n"
    "Problem:\n{question}\n\nThe step:\n{step}\n\n"
    "Search exhaustively over a small bounded range (e.g. all variables in 0..60, or the natural small "
    "bound for the problem). For each solution found, print one line: SOLUTION <values>. After the "
    "search print one line: COVERED <yes/no> — 'yes' iff every solution found is one the step claims. "
    "Output ONLY a ```python code block, nothing else."
)
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.S)
_V_RE = re.compile(r"VERDICT:\s*(ISSUE|OK)", re.I)
_R_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)


def stage_evidence(cases, anchors, consents, judge, tp, gpu_mem, max_model_len, thinking, max_steps_per_rollout) -> list[dict]:
    from vllm import LLM, SamplingParams
    by_uid = {c["uid"]: c for c in cases}
    det = {(a["uid"], h["step"]): h["det_evidence"] for a in anchors for h in a["hits"]}
    # first K agreed steps per rollout, in order
    per = defaultdict(list)
    for r in sorted(consents, key=lambda r: (r["uid"], r["step"])):
        if r["agreed"] and len(per[r["uid"]]) < max_steps_per_rollout:
            per[r["uid"]].append(r)
    llm = LLM(model=judge, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts, meta, res = [], [], []
    enum_prompts, enum_meta = [], []
    hits_by = {(a["uid"], h["step"]): h for a in anchors for h in a["hits"]}
    for uid, lst in per.items():
        c = by_uid[uid]; steps = split_react_steps(c["response"])
        for r in lst:
            d = det.get((uid, r["step"]))
            kind = hits_by.get((uid, r["step"]), {}).get("needs_enum")
            enum_pf = ("counting_small_case_check" if kind == "count" else
                       "claimed_unique_solution_search" if kind == "solutions" else None)
            if enum_pf and enum_pf in r["agreed"]:
                st = steps[r["step"]]
                tmpl = CODE_TMPL if kind == "count" else SEARCH_TMPL
                p = _chat(tok, tmpl.format(question=c["question"], step=st.text.strip()[:2500]), thinking)
                if len(tok(p).input_ids) <= max_model_len - 2048:
                    enum_prompts.append(p); enum_meta.append((uid, r["step"], enum_pf))
                continue
            if d is not None:   # deterministic checker already produced concrete evidence
                res.append(dict(uid=uid, step=r["step"], pf=d["pf"], verdict="ISSUE", reason=d["verdict"], source="checker"))
                continue
            pf = r["agreed"][0]
            st = steps[r["step"]]
            p = _chat(tok, EVIDENCE_TMPL.format(scope=PF_SKILLS[pf]["scope"], question=c["question"],
                                                prefix=c["response"][: st.char_start][-6000:].strip() or "(start)",
                                                step=st.text.strip()[:3000]), thinking)
            if len(tok(p).input_ids) > max_model_len - 2048:
                continue
            prompts.append(p); meta.append((uid, r["step"], pf))
    print(f"[evidence] {len(prompts)} LLM audits + {len(res)} deterministic + {len(enum_prompts)} enumerations", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.step_gate.evidence_temperature, max_tokens=_P.step_gate.evidence_max_tokens), use_tqdm=True)
    if enum_prompts:
        eouts = llm.generate(enum_prompts, SamplingParams(temperature=_P.step_gate.evidence_temperature, max_tokens=_P.step_gate.evidence_max_tokens), use_tqdm=True)
        for (uid, step, enum_pf), o in zip(enum_meta, eouts):
            ans = (o.outputs[0].text if o.outputs else "").split("</think>")[-1]
            m = _CODE_RE.search(ans)
            if not m:
                continue
            ok, out = run_python(m.group(1), timeout_s=10)
            if enum_pf == "claimed_unique_solution_search":
                if not ok:
                    continue
                sols = [ln for ln in out.splitlines() if ln.startswith("SOLUTION")]
                cov = [ln for ln in out.splitlines() if ln.startswith("COVERED")]
                uncovered = bool(cov) and cov[-1].strip().lower().endswith("no")
                res.append(dict(uid=uid, step=step, pf=enum_pf, source="search",
                                verdict="ISSUE" if (uncovered and sols) else "OK",
                                reason=("a brute-force search finds solutions the step claims do not exist: "
                                        + "; ".join(s_[:60] for s_ in sols[:4])) if (uncovered and sols) else ""))
                continue
            rows = [ln.split() for ln in out.strip().splitlines() if ln.strip()]
            rows = [r for r in rows if len(r) == 3 and all(x.lstrip("-").isdigit() for x in r)]
            if not ok or not rows:
                continue
            bad = [r for r in rows if r[1] != r[2]]
            res.append(dict(uid=uid, step=step, pf="counting_small_case_check",
                            verdict="ISSUE" if bad else "OK", source="enumeration",
                            reason=(f"the counting formula in this step disagrees with direct enumeration: "
                                    + "; ".join(f"n={r[0]}: formula {r[1]}, enumeration {r[2]}" for r in bad[:3]))
                                   if bad else ""))
    for (uid, step, pf), o in zip(meta, outs):
        ans = (o.outputs[0].text if o.outputs else "").split("</think>")[-1]
        v = _V_RE.search(ans); rs = _R_RE.search(ans)
        res.append(dict(uid=uid, step=step, pf=pf, verdict=(v.group(1).upper() if v else "OK"),
                        reason=(rs.group(1).strip()[:400] if rs else ""), source="llm"))
    return res


# ── stage: regen (policy, from the evidenced step) ───────────────────────

REGEN_NOTE = (
    "\n\n[PF {pf}] A check on the step above found a concrete error: {reason}\n"
    "Redo this step correctly and continue the solution to the end. "
    "You MUST end with `Action: finish[<answer>]`.\n\n"
)


def stage_regen(cases, evidence, policy, tp, gpu_mem, max_model_len, max_tokens, n_samples) -> list[dict]:
    from vllm import LLM, SamplingParams
    by_uid = {c["uid"]: c for c in cases}
    first = {}
    for e in sorted(evidence, key=lambda e: (e["uid"], e["step"])):
        if e["verdict"] == "ISSUE" and e["uid"] not in first:
            first[e["uid"]] = e
    llm = LLM(model=policy, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for uid, e in first.items():
        c = by_uid[uid]; steps = split_react_steps(c["response"]); st = steps[e["step"]]
        base = tok.apply_chat_template([{"role": "user", "content": build_react_user_prompt(c["question"])}],
                                       tokenize=False, add_generation_prompt=True)
        # keep the flagged step visible, inject the note right after it, regenerate from there
        p = base + c["response"][: st.char_end].rstrip() + REGEN_NOTE.format(pf=e["pf"], reason=e["reason"])
        if len(tok(p).input_ids) > max_model_len - 512:
            continue
        prompts.append(p); meta.append((uid, e["step"], e["pf"]))
    print(f"[regen] {len(prompts)} rollouts x {n_samples} samples", flush=True)
    outs = llm.generate(prompts, SamplingParams(temperature=_P.step_gate.regen_temperature, max_tokens=max_tokens, n=n_samples), use_tqdm=True)
    return [dict(uid=u, step=s, pf=pf, texts=[x.text for x in o.outputs]) for (u, s, pf), o in zip(meta, outs)]


# ── stage: score ─────────────────────────────────────────────────────────

_NUM = re.compile(r"^[-+]?\d+(\.\d+)?(/\d+)?$")


def _norm(s): return (s or "").strip().strip("$").replace(",", "").rstrip(".")
def _atype(s):
    s = _norm(s)
    return "none" if not s else ("num" if _NUM.match(s) else "expr")


def stage_score(cases, anchors, consents, evidence, regens) -> None:
    by_uid = {c["uid"]: c for c in cases}
    n = Counter(c["label"] for c in cases)
    anch = Counter(); cons = Counter(); evid = Counter()
    for a in anchors:
        if a["hits"]: anch[a["label"]] += 1
    agreed_uids = {r["uid"] for r in consents if r["agreed"]}
    for u in agreed_uids: cons[by_uid[u]["label"]] += 1
    issue_uids = {e["uid"] for e in evidence if e["verdict"] == "ISSUE"}
    for u in issue_uids: evid[by_uid[u]["label"]] += 1
    print("=== dual-consent funnel (rollouts) ===")
    print(f"  {'':<8}{'n':>5}{'anchor fired':>14}{'model agreed':>14}{'evidence=ISSUE':>16}")
    for lab in ("wrong", "correct"):
        print(f"  {lab:<8}{n[lab]:>5}{anch[lab]:>9} ({anch[lab]/max(1,n[lab]):.0%}){cons[lab]:>9} ({cons[lab]/max(1,n[lab]):.0%}){evid[lab]:>10} ({evid[lab]/max(1,n[lab]):.0%})")
    src = Counter(e["source"] for e in evidence if e["verdict"] == "ISSUE")
    print(f"  evidence sources: {dict(src)}")
    pfc = Counter(e["pf"] for e in evidence if e["verdict"] == "ISSUE")
    print(f"  PFs producing evidence: {dict(pfc)}")

    out = Counter(); pos = []
    for r in regens:
        c = by_uid[r["uid"]]; was = _em_match_multi(c["pred"], c["gold"])
        answers = [_extract_answer_math(t) or "" for t in r["texts"]]
        a0 = answers[0]
        # no gate
        fin = a0 if a0 else c["pred"]; now = _em_match_multi(fin, c["gold"])
        out[f"{c['label']}:nogate:rescue"] += (not was and now); out[f"{c['label']}:nogate:broke"] += (was and not now)
        # gate: all samples agree + same type as original
        ok = bool(a0) and all(_norm(x) == _norm(a0) for x in answers) and _atype(a0) == _atype(c["pred"])
        fin = a0 if ok else c["pred"]; now = _em_match_multi(fin, c["gold"])
        out[f"{c['label']}:gate:rescue"] += (not was and now); out[f"{c['label']}:gate:broke"] += (was and not now)
        n_steps = next(a["n_steps"] for a in anchors if a["uid"] == r["uid"])
        pos.append(r["step"] / max(1, n_steps))
    print("\n=== regeneration from the evidenced step ===")
    for g in ("nogate", "gate"):
        print(f"  {g:<7} wrong: rescue {out[f'wrong:{g}:rescue']:>3}/{n['wrong']}   correct: broke {out[f'correct:{g}:broke']:>3}/{n['correct']}")
    if pos:
        pos.sort(); print(f"  intervention step position median {pos[len(pos)//2]:.0%} of steps  (n={len(pos)})")
    print("\n  (reference — FINAL-level locator chain on the same cases: rescue 37 nogate / 10 gated, broke 15 / 0)")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["anchor", "consent", "evidence", "regen", "score"])
    ap.add_argument("--cases", default=str(_HASP / "data/llm_anchor/cases.jsonl"))
    ap.add_argument("--tag", default="sg1")
    ap.add_argument("--policy", default=_P.models.policy_math)
    ap.add_argument("--judge", default=_P.models.judge)
    ap.add_argument("--tp", type=int, default=_P.serving.tensor_parallel)
    ap.add_argument("--gpu-mem", type=float, default=_P.serving.gpu_memory_utilization)
    ap.add_argument("--max-model-len", type=int, default=_P.step_gate.max_model_len)
    ap.add_argument("--max-tokens", type=int, default=_P.step_gate.max_tokens)
    ap.add_argument("--max-hits", type=int, default=_P.step_gate.max_hits, help="anchor-fired steps kept per rollout")
    ap.add_argument("--max-audits", type=int, default=_P.step_gate.max_audits, help="agreed steps audited per rollout")
    ap.add_argument("--n-samples", type=int, default=_P.step_gate.n_samples)
    ap.add_argument("--no-thinking", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(a.cases)]
    P = lambda name: OUT / f"{name}_{a.tag}.jsonl"
    load = lambda name: [json.loads(l) for l in P(name).open()] if P(name).exists() else []
    dump = lambda name, rows: P(name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    if a.stage == "anchor":
        dump("anchor", stage_anchor(cases, a.max_hits)); print(f"[anchor] -> {P('anchor')}"); return
    anchors = load("anchor")
    if a.stage == "consent":
        dump("consent", stage_consent(cases, anchors, a.policy, a.tp, a.gpu_mem, a.max_model_len)); return
    consents = load("consent")
    if a.stage == "evidence":
        dump("evidence", stage_evidence(cases, anchors, consents, a.judge, a.tp, a.gpu_mem, a.max_model_len,
                                        not a.no_thinking, a.max_audits)); return
    evidence = load("evidence")
    if a.stage == "regen":
        dump("regen", stage_regen(cases, evidence, a.policy, a.tp, a.gpu_mem, a.max_model_len, a.max_tokens, a.n_samples)); return
    stage_score(cases, anchors, consents, evidence, load("regen"))


if __name__ == "__main__":
    main()
