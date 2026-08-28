"""Verifiers score generated rollouts for rejection sampling.

Two verifier kinds:
  * PFVerifier  — replays the rollout against the PF layer and uses S1..S4
  * TeacherVerifier — asks GPT-4o to score the generation (Q_skill or action quality)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    prompt_index: int
    group_index: int
    score: float
    breakdown: Dict[str, float]
    raw: str = ""


# ----------------------------------------------------------------------
# PF-based verifier (Objective A rollouts)
# ----------------------------------------------------------------------

_ACTION_PAT = re.compile(r"Action:\s*(SEARCH|READ|FINAL)\s*\((.*?)\)\s*$", re.IGNORECASE | re.DOTALL)


class PFVerifier:
    """Score a generated action by (a) syntactic validity and (b) alignment with the PF-corrected action.

    For use when the rollout dataset was produced from `objA_prompts.jsonl`
    and we have access to the expected PF-corrected action in a sibling
    `objA_sft.jsonl` (keyed by sample_id + step_index).
    """

    def __init__(self, reference_sft_path: str):
        self._ref: Dict = {}
        with open(reference_sft_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r.get("sample_id"), r.get("step_index"))
                self._ref[key] = r
        logger.info("PFVerifier loaded %d reference steps from %s", len(self._ref), reference_sft_path)

    def score(self, prompt: Dict[str, Any], generation: str, prompt_index: int = 0, group_index: int = 0) -> VerifyResult:
        m = _ACTION_PAT.search(generation)
        if not m:
            return VerifyResult(prompt_index, group_index, 0.0, {"syntactic": 0.0}, raw=generation)

        gen_type = m.group(1).upper()
        gen_arg = m.group(2).strip()

        ref_row = self._ref.get((prompt.get("sample_id"), prompt.get("step_index")))
        if ref_row is None:
            # Syntactic only
            return VerifyResult(prompt_index, group_index, 0.3, {"syntactic": 1.0, "reference": 0.0}, raw=generation)

        ref_assistant = ref_row["messages"][-1]["content"]
        ref_m = _ACTION_PAT.search(ref_assistant)
        if not ref_m:
            return VerifyResult(prompt_index, group_index, 0.3, {"syntactic": 1.0, "reference": 0.0}, raw=generation)
        ref_type = ref_m.group(1).upper()
        score = 1.0 if gen_type == ref_type else 0.3
        breakdown = {"syntactic": 1.0, "type_match": float(gen_type == ref_type)}
        return VerifyResult(prompt_index, group_index, score, breakdown, raw=generation)


# ----------------------------------------------------------------------
# PF helper (GPT) verifier (Objective B skill rollouts or fine-grained action quality)
# ----------------------------------------------------------------------

TEACHER_SKILL_PROMPT = (
    "You are an expert reviewer of PF skills for a ReAct web-search agent. "
    "Given a failure pattern and a candidate skill (SKILL.md + PF code), "
    "rate its overall quality in [0, 1]. Return JSON: {\"score\": float, \"reason\": str}."
)


class TeacherVerifier:
    def __init__(self, model_name: str = "", max_concurrent: int = 8):
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        # Lazy import of openai client
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def score(self, prompt: Dict[str, Any], generation: str, prompt_index: int = 0, group_index: int = 0) -> VerifyResult:
        client = self._get_client()
        user_msg = (
            f"Failure pattern: {prompt.get('target_failure_pattern', '')[:800]}\n\n"
            f"Candidate skill:\n{generation[:3000]}"
        )
        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": TEACHER_SKILL_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            s = float(data.get("score", 0.0))
            return VerifyResult(prompt_index, group_index, s, {"teacher": s}, raw=generation)
        except Exception as e:
            logger.warning("PF helper verifier failed: %s", e)
            return VerifyResult(prompt_index, group_index, 0.0, {"teacher": 0.0}, raw=generation)


# ----------------------------------------------------------------------

def verify_rollout_file(rollout_path: str, verifier, output_path: str) -> str:
    """Score every rollout in a jsonl file and write results to `output_path`."""
    out = open(output_path, "w", encoding="utf-8")
    n = 0
    current_prompt_index = -1
    last_key = None
    group_index = 0
    with open(rollout_path, "r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            # Group rollouts that share the same prompt identity
            key = row.get("sample_id"), row.get("step_index"), row.get("target_failure_pattern")
            if key != last_key:
                current_prompt_index += 1
                group_index = 0
                last_key = key
            res = verifier.score(row, row.get("generation", ""), current_prompt_index, group_index)
            row["verifier_score"] = res.score
            row["verifier_breakdown"] = res.breakdown
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            group_index += 1
    out.close()
    logger.info("Verified %d rollouts → %s", n, output_path)
    return output_path


class SpecExampleVerifier:
    """HASP (2026-08-24): deterministic code-domain verifier — run the spec's
    own examples / public tests on the generated solution in the sandbox.

    This is the training-time form of the same acceptance rule used at inference —
    keep the first sample that passes the spec's own examples: in rejection
    sampling, a
    candidate that fails the spec's examples is rejected outright; candidates
    with no checkable examples get a neutral 0.5 so they are ranked by the
    other signals instead of being dropped.

    Zero API cost. Prompt rows must carry `question` (the spec) and optionally
    `entry_point` / `public_test_code` (the runner's step_context fields).
    """

    def __init__(self, timeout_s: float = 8.0):
        import importlib.util as _iu
        from pathlib import Path as _P
        p = _P(__file__).resolve().parents[2] / "skills" / "code" / "evidence_pfs.py"
        spec = _iu.spec_from_file_location("_hasp_code_evidence", str(p))
        self._cpf = _iu.module_from_spec(spec); spec.loader.exec_module(self._cpf)

    def score(self, prompt: Dict[str, Any], generation: str,
              prompt_index: int = 0, group_index: int = 0) -> VerifyResult:
        ctx = {"question": prompt.get("question", ""),
               "entry_point": prompt.get("entry_point", ""),
               "public_test_code": prompt.get("public_test_code", "")}
        try:
            ev = self._cpf.spec_example_evidence(ctx, generation)
        except Exception as e:
            return VerifyResult(prompt_index, group_index, 0.5,
                               {"spec_example": f"error:{type(e).__name__}"}, raw=generation)
        if ev is None:
            # distinguish "passed the examples" from "no examples to run"
            has_ex = bool(self._cpf._DOCTEST.findall(ctx["question"]) or
                          self._cpf._ASSERT.findall(ctx["question"]) or ctx["public_test_code"])
            s = 1.0 if has_ex else 0.5
            return VerifyResult(prompt_index, group_index, s, {"spec_example": "pass" if has_ex else "no_examples"}, raw=generation)
        return VerifyResult(prompt_index, group_index, 0.0, {"spec_example": ev[:200]}, raw=generation)
