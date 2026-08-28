"""Score model outputs against EvalPlus/BCB sandbox tests.

Reuses the project's `CodeSandbox.evaluate_with_test_script` and
`CodeAnswerEvaluator.extract` so the scoring matches what
`scripts/run_code_judge_eval.py` produces for the agent runs.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.skills_agent.eval.code_sandbox import CodeSandbox  # noqa: E402
from src.skills_agent.eval.metrics import CodeAnswerEvaluator  # noqa: E402


def score_one(row: Dict[str, Any], raw_output: str,
              sandbox: CodeSandbox, sandbox_bcb: CodeSandbox) -> Dict[str, Any]:
    code = CodeAnswerEvaluator.extract(raw_output)
    eval_test_code = row.get("eval_test_code", "")
    entry_point = row.get("entry_point") or row.get("metadata", {}).get("entry_point")
    if not eval_test_code:
        return {"sample_id": row["sample_id"], "passed": False, "reason": "no_eval_test_code"}
    sb = sandbox_bcb if (row.get("benchmark") == "BigCodeBench") else sandbox
    res = sb.evaluate_with_test_script(code, eval_test_code, entry_point)
    return {
        "sample_id": row["sample_id"],
        "variant": row.get("variant"),
        "passed": bool(res.pass_at_1),
        "first_failure_msg": res.first_failure_msg,
        "extracted_len": len(code),
    }


def score_batch(
    rows: List[Dict[str, Any]],
    raw_outputs: List[str],
    workers: int = 8,
) -> List[Dict[str, Any]]:
    assert len(rows) == len(raw_outputs)
    sandbox = CodeSandbox()
    # BCB problems need ~5-10s just for matplotlib+sklearn imports — keep a
    # longer-timeout sandbox for them while HumanEval/MBPP stay on defaults.
    sandbox_bcb = CodeSandbox(cpu_seconds=30, wall_timeout_s=30.0)
    results: List[Dict[str, Any]] = [None] * len(rows)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(score_one, row, out, sandbox, sandbox_bcb): i
            for i, (row, out) in enumerate(zip(rows, raw_outputs))
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = {"sample_id": rows[i]["sample_id"], "passed": False, "reason": f"score_err:{e}"}
    return results


def pass_at_1(scored: List[Dict[str, Any]]) -> float:
    if not scored:
        return 0.0
    return sum(1 for r in scored if r.get("passed")) / len(scored)
