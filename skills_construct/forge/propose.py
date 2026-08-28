"""Stage 2 — propose candidate PFs from real wrong cases with a local model.

The proposer is deliberately narrow. It is not asked "what could go wrong
here"; it is asked "what did the model WRITE DOWN that we can re-verify, and
what is the cheapest deterministic check that catches it". Every PF that ever
produced a rescue came out of that question, and every falsified one came out
of the looser one.

Runs offline on one L40S (no API, no key). Output is data — `PFSpec` records —
which the structural gate in spec.py filters before screen.py ever runs them.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from .spec import PFSpec, validate_spec, registered_pf_ids

_HASP = Path(__file__).resolve().parents[2]

_CONTRACT = '''You design PROGRAM FUNCTIONS (PFs) for a ReAct agent. A PF is a small,
deterministic Python check that runs while the agent is solving a problem. If it finds
hard evidence of a specific error, it injects that evidence so the agent redoes the work
from that point; otherwise it stays silent and the agent's original answer stands.

A PF is exactly three things:
  ANCHOR   — WHERE it attaches. level="step" (one reasoning step) or "final" (the
             committed answer), plus a trigger saying what the text must contain for
             the PF to look at all.
  EVIDENCE — HOW it decides. "deterministic" (recompute / test point / range check) or
             "executed" (run the code). Nothing else is allowed here.
  ACTION   — WHAT it says. A concrete verdict naming the wrong value and, when it can,
             the correct one.

THE RULES THAT DECIDE WHETHER A PF WORKS
1. A PF may only read WHAT THE MODEL ITSELF WROTE — its reasoning, its own
   `compute[...]` Observations, its committed answer, or the problem spec's own
   examples. It NEVER sees the correct answer. A check that needs the correct answer
   scores perfectly in testing and is worthless in production.
2. Anchor on a CHECKABLE CLAIM. The wins all re-verify something the model asserted:
   an arithmetic result it wrote, a uniqueness claim, a doctest it should satisfy.
   "The reasoning seems confused" is not checkable. Do not propose it.
3. SILENCE IS THE DEFAULT. The check runs on correct solutions too. Every false fire
   interrupts work that was already right. Prefer a narrow check that fires on 8% of
   wrong solutions and 0% of correct ones over a broad one that fires on 40% of both.
4. Recompute, do not pattern-match. `re.search("carry")` is not evidence; re-evaluating
   the model's own stated sum with sympy is.

CHECKER SIGNATURES — define exactly one function named `check`.

  checker_kind="step":
      def check(step_text, full_response, step_start):
          # step_text: one reasoning step. Return None to stay silent, or
          # {"pf": "<skill_id>", "verdict": "<what is wrong and the right value>",
          #  "fix": "<corrected step text, or None>"}

  checker_kind="answer":
      def check(text, arg, ctx):
          # text: the full response; arg: the committed answer;
          # ctx: {"question", "entry_point", "public_test_code"} — no answer key.
          # Return None to stay silent, or a verdict string.

Allowed imports: re, math, fractions, itertools, collections, sympy.
Forbidden: os, sys, subprocess, socket, pathlib, open(), exec(), eval().

A GOOD EXAMPLE (this one is in production and rescues with zero false fires):
  skill_id: compute_observation_verify
  anchor: step-level; trigger "the step contains a `compute[...]` action whose
          Observation the model wrote itself"
  evidence: deterministic — parse the expression, re-evaluate it with sympy, compare
          against the Observation the model wrote at 1e-2 tolerance
  action: "the Observation for compute[7*13] is 81, but 7*13 = 91" + the corrected value
  why it works: the model wrote both the expression and its result, so the claim is
          self-contained and machine-checkable, and a correct solution's Observations
          agree with sympy, so it almost never fires on them.
'''

_TASK = '''ERROR FAMILY: {family}
It covers {n_wrong} of the wrong solutions ({share:.0%} of all failures).
Checkable artifacts present in this family's responses:
{surfaces}
Already covered by existing PFs (do NOT duplicate these): {taken}

Here are real failures from this family. Each shows the problem, what the model wrote,
and what it committed. The correct answer is shown ONLY so you can see what went wrong —
your checker must never depend on it.

{examples}

Propose {k} DIFFERENT program functions for this family. Prefer narrow, provable checks.
If a genuinely checkable claim does not exist in these responses, return fewer — or an
empty list. Returning nothing is a valid and useful answer; a PF that fires on correct
solutions is worse than no PF.

Return ONLY a JSON array, no prose, no markdown fence:
[{{"skill_id": "snake_case_id",
   "family_scope": "Only flag <exactly what>; state <what the verdict must say>.",
   "anchor": {{"level": "step|final", "trigger": "what the text must contain",
              "evidence": "deterministic|executed"}},
   "checker_kind": "step|answer",
   "can_repair": true/false,
   "rationale": "which claim the model wrote, and why re-verifying it is decisive",
   "checker_src": "import re\\n\\ndef check(...):\\n    ..."}}]
'''


def _example_block(cases: List[Dict], uids: List[str], max_chars: int = 2600,
                   n: int = 4) -> str:
    by_uid = {c["uid"]: c for c in cases}
    out = []
    for i, uid in enumerate(uids[:n]):
        c = by_uid.get(uid)
        if not c:
            continue
        resp = (c["response"] or "")
        if len(resp) > max_chars:      # keep the commit, that is where anchors live
            resp = resp[:max_chars // 2] + "\n  ...[trimmed]...\n" + resp[-max_chars // 2:]
        out.append(
            f"--- case {i + 1} ({uid}) ---\nPROBLEM: {c['question'][:900]}\n"
            f"MODEL WROTE:\n{resp}\n"
            f"COMMITTED: {c['pred']}   |   CORRECT: {c['gold']}  (reference only)\n"
        )
    return "\n".join(out)


_LEDGER_NOTE = '''
ALREADY TRIED AND FALSIFIED — do not propose these again, and do not propose a
rewording of them. Each one was screened or run end to end and failed:
{falsified}

Read that list before you answer. The most common way these failed is firing on
correct solutions as often as on wrong ones — a check that cannot tell the two apart
is worse than no check, because it interrupts work that was already right.
'''


def build_prompts(families: List[Dict], cases: List[Dict], k: int,
                  falsified_note: str = "") -> List[Dict]:
    prompts = []
    for fam in families:
        surfaces = "\n".join(
            f"  - {r['surface']}: in {r['wrong_rate']:.0%} of this family's wrong responses "
            f"vs {r['correct_rate']:.0%} of correct ones (lift {r['lift']})"
            for r in fam["claim_surfaces"][:5]) or "  (none detected — say so and return [])"
        taken = ", ".join(sorted({r["taken_by"] for r in fam["claim_surfaces"] if r["taken_by"]})) or "none"
        note = _LEDGER_NOTE.format(falsified=falsified_note) if falsified_note.strip() else ""
        prompts.append(dict(
            family=fam["family"],
            text=_CONTRACT + note + "\n\n" + _TASK.format(
                family=fam["family"], n_wrong=fam["n_wrong"], share=fam["share_of_wrong"],
                surfaces=surfaces, taken=taken, k=k,
                examples=_example_block(cases, fam["examples"]))))
    return prompts


def _parse(raw: str, domain: str, family: str) -> List[PFSpec]:
    txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for d in arr if isinstance(arr, list) else []:
        if not isinstance(d, dict) or "checker_src" not in d:
            continue
        d.setdefault("anchor", {})
        out.append(PFSpec(
            skill_id=str(d.get("skill_id", "")).strip(),
            domain=domain,
            family_scope=str(d.get("family_scope", "")).strip(),
            anchor={k: str(v) for k, v in (d.get("anchor") or {}).items()},
            checker_kind=str(d.get("checker_kind", "step")).strip(),
            checker_src=str(d["checker_src"]),
            can_repair=bool(d.get("can_repair", False)),
            rationale=str(d.get("rationale", "")).strip(),
            source_uids=[],
        ))
    return out


def load_engine(model: str, tp: int = 1, gpu_mem: float = 0.88, max_model_len: int = 32768):
    """One engine per round: propose and refine share it."""
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, trust_remote_code=True)
    return llm, tok


def propose_with_engine(families: List[Dict], cases: List[Dict], domain: str, llm, tok,
                        k: int = 3, max_tokens: int = 4096, temperature: float = 0.7,
                        n: int = 2, falsified_note: str = "") -> List[PFSpec]:
    """Generate, then structurally gate. Returns only specs that parse clean."""
    from vllm import SamplingParams

    prompts = build_prompts(families, cases, k, falsified_note=falsified_note)
    if not prompts:
        return []
    chat = [tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts]
    outs = llm.generate(chat, SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n),
                        use_tqdm=True)

    known = registered_pf_ids(_HASP)
    specs, seen, dropped = [], set(), []
    for p, o in zip(prompts, outs):
        fam_uids = next((f["examples"] for f in families if f["family"] == p["family"]), [])
        for cand in o.outputs:
            for s in _parse(cand.text, domain, p["family"]):
                s.source_uids = fam_uids[:8]
                if s.skill_id in seen:
                    continue
                bad = validate_spec(s, known)
                if bad:
                    dropped.append((s.skill_id or "<unnamed>", bad[0]))
                    continue
                seen.add(s.skill_id)
                specs.append(s)

    print(f"[propose] {len(specs)} specs passed the structural gate, "
          f"{len(dropped)} dropped", flush=True)
    for sid, why in dropped[:12]:
        print(f"  dropped {sid}: {why}", flush=True)
    return specs


def propose(families: List[Dict], cases: List[Dict], domain: str, model: str,
            k: int = 3, tp: int = 1, gpu_mem: float = 0.88,
            max_model_len: int = 32768, max_tokens: int = 4096,
            temperature: float = 0.7, n: int = 2,
            falsified_note: str = "") -> List[PFSpec]:
    """Standalone entry: load an engine, propose, release it."""
    if not families:
        return []
    llm, tok = load_engine(model, tp, gpu_mem, max_model_len)
    return propose_with_engine(families, cases, domain, llm, tok, k=k,
                               max_tokens=max_tokens, temperature=temperature, n=n,
                               falsified_note=falsified_note)
