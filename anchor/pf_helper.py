"""PF helpers — plug the LLM locator into the existing PF runtime.

("helper" = the optional model a PF may consult for evidence, reached as
`ctx.pf_helper` or as the trailing `helper` argument of `intervene()`. A PF
written against the older interface may still name that parameter `teacher`;
`call_base_intervene` accepts both spellings so it keeps receiving one.)

The 51-PF library's verify-style PFs (`_MathVerifyPF` subclasses:
arithmetic_slip, algebraic_sign_error, case_incompleteness, boundary_violation,
…) already have a helper hook: `intervene()` calls `_teacher_verify()`, and an
"ISSUE: <text>" reply becomes the injected context, "OK" suppresses the PF.
In every eval so far `teacher_model=None`, so they only ever emitted their
fixed reminder string — which is why they fixed 3 of 2,300 committed-wrong
answers.

A helper here exposes `locate(question, reasoning, family_hint, skill_id,
uid)` and returns either
    "ISSUE: [step k] <concrete, evidence-backed verdict>"
or  "OK"
The PF runtime (patched in HASP's dynamic_program_functions.py) prefers this
method and passes the FULL reasoning, not the 2000-char truncation.

Two implementations:
  CachedEvidenceHelper — serves precomputed locate results (the
      `data/llm_anchor/locate_<tag>.jsonl` produced by llm_locate.py) keyed by
      case uid. Lets the whole pf_select dispatch run end-to-end over the
      mined cases with zero extra GPU.
  VLLMEvidenceHelper — live locator on a vLLM engine, family-scoped prompt.
      For the online pf_select loop later. Same output contract.

Neither helper decides whether the evidence is ACTED on — that is the gate
chain's job (agreement → type → votes → arbiter) at dispatch time. The PF helper
only produces localized evidence or stays silent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from anchor.anchor import segment_steps  # noqa: E402
from hasp_config import protocol as _protocol  # noqa: E402

_P = _protocol()

# Map a PF's error_hint onto the family the locator should restrict itself to.
# Keys are substrings of the PF's `error_hint`; the value is the instruction
# fragment spliced into the locate prompt. Unknown hints fall back to the
# general locator.
_FAMILY_SCOPE = {
    "arithmetic chain": "Only flag a step whose literal arithmetic (sums, products, fraction reductions, modular results) is wrong; recompute it and state the correct value.",
    "algebraic signs": "Only flag a step with a sign / distribution / factoring / expansion error; state the correct expression.",
    "boundary": "Only flag a step that violates a hard constraint of the problem (range, integrality, domain, positivity); quote the constraint.",
    "case": "Only flag a step that omits a case the argument depends on; name the missing case.",
    "units": "Only flag a step with a unit or dimension mismatch; state the mismatch.",
}


def _scope_for(family_hint: str) -> str:
    h = (family_hint or "").lower()
    for k, v in _FAMILY_SCOPE.items():
        if k in h:
            return v
    return ("Flag the FIRST step containing a definite mathematical error (wrong "
            "computation, invalid deduction, misread condition, missed case).")


LOCATE_TMPL = (
    "You are auditing a student's solution to a math problem.\n{scope}\n"
    "Do not flag stylistic issues or unfinished steps. If no such error exists, answer NONE.\n\n"
    "Problem:\n{question}\n\nSolution, split into numbered steps:\n{steps}\n\n"
    "Answer in exactly this format:\nSTEP: <number or NONE>\nREASON: <one or two sentences stating concretely what is wrong>"
)
_STEP_RE = re.compile(r"STEP:\s*(NONE|\d+)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)


def _format(step: Optional[int], reason: str) -> str:
    if step is None or not reason:
        return "OK"
    return f"ISSUE: [step {step + 1}] {reason.strip()}"


class CachedEvidenceHelper:
    """Serve locate results precomputed by llm_locate.py, keyed by case uid."""

    def __init__(self, locate_jsonl: str | Path):
        self._by_uid = {}
        for line in Path(locate_jsonl).open():
            r = json.loads(line)
            self._by_uid[r["uid"]] = r
        self.calls = 0
        self._served: set[str] = set()

    def locate(self, question: str, reasoning: str, family_hint: str = "",
               skill_id: str = "", uid: Optional[str] = None) -> str:
        self.calls += 1
        r = self._by_uid.get(uid or "")
        if r is None:
            return "OK"
        # The cached result is a single general locate per rollout, not a
        # family-scoped one, so hand it to the FIRST PF that asks and answer
        # "OK" to the rest — otherwise every selected PF injects the same
        # evidence and the feedback block repeats itself.
        if uid in self._served:
            return "OK"
        self._served.add(uid)
        return _format(r.get("step"), r.get("reason", ""))

    # legacy entry point some PFs may still call
    def generate_from_messages(self, messages, max_tokens=200, temperature=0.0) -> str:
        return "OK"


class VLLMEvidenceHelper:
    """Live family-scoped locator on a vLLM engine (one call per PF firing).

    Use for the online pf_select loop. For offline batch evaluation prefer
    llm_locate.py + CachedEvidenceHelper — same prompt family, batched.
    """

    def __init__(self, llm, thinking: bool = _P.helper.thinking,
                 max_tokens: int = _P.helper.max_tokens,
                 max_model_len: int = _P.helper.max_model_len):
        self.llm = llm
        self.tok = llm.get_tokenizer()
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self._cache: dict[tuple, str] = {}

    def locate(self, question: str, reasoning: str, family_hint: str = "",
               skill_id: str = "", uid: Optional[str] = None) -> str:
        from vllm import SamplingParams
        key = (uid or hash(reasoning), _scope_for(family_hint))
        if key in self._cache:
            return self._cache[key]
        steps = segment_steps(reasoning[:36000])
        if not steps:
            return "OK"
        body = "\n".join(f"[{s.idx + 1}] {s.text.strip()}" for s in steps)
        msgs = [{"role": "user", "content": LOCATE_TMPL.format(
            scope=_scope_for(family_hint), question=question, steps=body)}]
        try:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=self.thinking)
        except TypeError:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(self.tok(p).input_ids) > self.max_model_len - self.max_tokens:
            return "OK"
        out = self.llm.generate([p], SamplingParams(temperature=_P.helper.temperature, max_tokens=self.max_tokens), use_tqdm=False)
        ans = out[0].outputs[0].text if out and out[0].outputs else ""
        ans = ans.split("</think>")[-1]
        m = _STEP_RE.search(ans); r = _REASON_RE.search(ans)
        step = None
        if m and m.group(1).upper() != "NONE":
            k = int(m.group(1)) - 1
            if 0 <= k < len(steps):
                step = k
        res = _format(step, r.group(1)[:400] if r else "")
        self._cache[key] = res
        return res

    def generate_from_messages(self, messages, max_tokens=200, temperature=0.0) -> str:
        return "OK"
