"""Re-judge every FAILING code solution locally with its hidden test driver and
classify the first failure (full traceback, not the 300-char stub stored in
the error files). CPU-heavy (hundreds of sandbox runs) — submit via
scripts/slurm/cpu.sbatch, never on the login node.

Output: logs/code_error_taxonomy.log (table) + data/code_error_taxonomy.jsonl
"""
from __future__ import annotations
import ast, json, re, sys
from collections import Counter
from pathlib import Path
_HASP = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(_HASP))
from src.skills_agent.eval.code_sandbox import CodeSandbox  # noqa: E402

from hasp_paths import code_episodes_dir  # noqa: E402
ROOT = code_episodes_dir()
FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _code(f):
    s = str(f or "")
    if s.strip().startswith("{"):
        try: s = str(ast.literal_eval(s).get("answer", s))
        except Exception: pass
    m = FENCE.findall(s); return m[-1] if m else s


def main():
    lookup = {}
    for bm in ("humaneval_plus", "mbpp_plus", "bigcodebench"):
        for row in map(json.loads, (S / f"data/code/{bm}.jsonl").open()):
            lookup[str(row["sample_id"])] = dict(test=row.get("eval_test_code", ""), entry=row.get("entry_point"), bm=bm)
    sb = CodeSandbox(wall_timeout_s=20.0)
    tax = Counter(); ex = {}; rows = []
    for bm in ("humaneval_plus", "mbpp_plus", "bigcodebench"):
        for arm in ("baseline", "pf_no_teacher", "pf_with_teacher"):
            try:
                jj = json.load(open(ROOT / bm / arm / "base_clean_episodes.codejudge.json"))["results"]
                eps = {str(e["sample_id"]): e for e in map(json.loads, (ROOT / bm / arm / "base_clean_episodes.jsonl").open())}
            except Exception as e:
                print("skip", bm, arm, e); continue
            for it in jj:
                if it.get("passed"): continue
                sid = str(it["sample_id"]); e = eps.get(sid); info = lookup.get(sid)
                if not e or not info or not info["test"]: continue
                try:
                    res = sb.evaluate_with_test_script(_code(e.get("final")), info["test"], entry_point=info["entry"])
                    msg = res.first_failure_msg or ""
                except Exception as ex_:
                    msg = f"HARNESS_ERROR {type(ex_).__name__}"
                mm = re.findall(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b", msg)
                if "timeout" in msg.lower(): k = "timeout"
                elif mm: k = mm[-1]
                elif "FAIL" in msg or "assert" in msg.lower(): k = "unittest_FAIL"
                elif not msg: k = "empty_msg"
                else: k = "other"
                tax[(bm, k)] += 1; ex.setdefault((bm, k), msg[-220:].replace("\n", " | "))
                rows.append(dict(benchmark=bm, arm=arm, sample_id=sid, family=k, first_failure=msg[-600:]))
    (_HASP / "data" / "code_error_taxonomy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n=== failing solutions re-judged locally: first-failure taxonomy ===")
    for (bm, k), v in sorted(tax.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"  {bm:<16}{k:<24}{v:>4}   e.g. {ex[(bm, k)][:140]}")
    print("DONE")


if __name__ == "__main__":
    main()
