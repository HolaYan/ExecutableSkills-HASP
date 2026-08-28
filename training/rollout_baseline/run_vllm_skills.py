"""Skills + PF helper pipeline on top of direct-answer baseline.

Architecture (per problem):
  Stage 1: base_model.generate(direct prompt) → candidate_v1
  Stage 2: PF helper.review(question, candidate_v1, SKILL CATALOG)
              → {selected_skills, improved_v2}
              PF helper reads the skill catalog (the SKILL.md system_summary
              for each registered code skill) and decides which patterns
              apply, then rewrites the code accordingly.
  Stage 3 (optional, gated): if PF helper flagged "needs_full_rewrite",
              call base_model.generate(question + teacher_feedback) and
              feed result through Stage 2 again. Off by default — the
              user said "如需，如果问题不大则直接进入下一步".
  Stage 4: PF helper.final_review(question, current_candidate)
              → FINAL Python code

Score FINAL with the same `evaluate_with_test_script` path as
`baseline/scripts/run_vllm.py` — so numbers compare apples-to-apples.

Outputs (mirrors run_vllm.py layout, plus an extra `stages/` dir):
  baseline/outputs/{name}/raw/{dataset}.jsonl       (Stage 1 output)
  baseline/outputs/{name}/stages/{dataset}.jsonl    (per-stage trace)
  baseline/outputs/{name}/scored/{dataset}.jsonl
  baseline/outputs/{name}/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from .data_utils import iter_subsets, DATASETS  # noqa: E402
from .prompt import build_messages  # noqa: E402
from .sandbox_eval import score_batch, pass_at_1  # noqa: E402

from src.skills_agent.eval.metrics import CodeAnswerEvaluator  # noqa: E402
from src.skills_agent.eval.model_loader import load_model_api  # noqa: E402
from src.skills_agent.eval.code_sandbox import CodeSandbox  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Skill catalog — read from SKILL.md frontmatter under skills_code/seed/
# ----------------------------------------------------------------------

def load_skill_catalog(skill_dir: Path) -> List[Dict[str, str]]:
    """Return [{id, name, summary}] for every code skill that has SKILL.md.

    Used by the PF helper prompt as a "diagnostic dictionary" — the PF helper
    decides which patterns apply to the candidate code.
    """
    import re, yaml
    out: List[Dict[str, str]] = []
    for child in sorted(skill_dir.iterdir()):
        if not child.is_dir():
            continue
        md = child / "SKILL.md"
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1))
        except Exception:
            continue
        sid = fm.get("skill_id") or child.name
        out.append({
            "id": sid,
            "name": fm.get("name", sid),
            "summary": (fm.get("system_summary") or "").strip(),
        })
    return out


def format_skill_catalog_for_prompt(skills: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"- {s['id']}: {s['summary']}" for s in skills if s["summary"]
    )


# ----------------------------------------------------------------------
# Stage 2 + 4 prompts
# ----------------------------------------------------------------------

REVIEW_SYSTEM = (
    "You are an expert Python code reviewer. You will be given a coding "
    "problem and a student's draft solution. The skill catalog below "
    "describes common bug patterns. Apply any catalog patterns that match "
    "the student's code AND fix any other bugs you spot. Output ONLY a "
    "JSON object with two keys:\n"
    '  "selected_skills": list of skill_ids you applied (may be empty),\n'
    '  "improved_code":   the corrected Python source as a single string '
    '(no fences, no commentary, just the code body — must be self-'
    "contained and define the requested function/class).\n"
    "If the student's code is already correct, copy it verbatim into "
    "improved_code and return an empty selected_skills list."
)

FINAL_REVIEW_SYSTEM = (
    "You are the final reviewer. Confirm or revise the candidate solution "
    "for the coding problem. Output ONLY raw Python source — no JSON, no "
    "fences, no commentary. The output must define the requested function/"
    "class with the same name and signature, be self-contained, and pass "
    "any docstring examples. Make minimal edits if the candidate is already "
    "correct; rewrite if you spot a bug."
)


def teacher_review(
    teacher,
    question: str,
    candidate: str,
    skill_catalog_text: str,
) -> Tuple[List[str], str]:
    """Stage 2: PF helper reads candidate, selects skills, returns improved code.

    Returns (selected_skill_ids, improved_code). On any failure falls back
    to ([], candidate).
    """
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Student's draft solution:\n```python\n{candidate.strip()}\n```\n\n"
        f"Skill catalog (each line is `skill_id: when_it_applies`):\n"
        f"{skill_catalog_text}\n\n"
        f"Output the JSON described in the system message."
    )
    try:
        out = teacher.generate(
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=2048, temperature=0.0,
        )
    except Exception as e:
        logger.warning("teacher_review call failed: %s", e)
        return [], candidate
    if not out:
        return [], candidate
    # Parse JSON. Be lenient: the model may wrap in ```json ... ```.
    import re
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return [], candidate
    try:
        d = json.loads(m.group(0))
    except Exception:
        # Sometimes models emit Python literal — try ast
        try:
            import ast
            d = ast.literal_eval(m.group(0))
        except Exception:
            return [], candidate
    skills = d.get("selected_skills") or []
    improved = d.get("improved_code") or candidate
    if not isinstance(skills, list):
        skills = []
    if not isinstance(improved, str) or not improved.strip():
        improved = candidate
    return skills, improved


def teacher_final(
    teacher,
    question: str,
    candidate: str,
) -> str:
    """Stage 4: PF helper does a final pass and emits the FINAL Python code."""
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Candidate solution after Stage-2 review:\n"
        f"```python\n{candidate.strip()}\n```\n\n"
        f"Confirm or fix. Output the final Python source only."
    )
    try:
        out = teacher.generate(
            messages=[
                {"role": "system", "content": FINAL_REVIEW_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=2048, temperature=0.0,
        )
    except Exception as e:
        logger.warning("teacher_final call failed: %s", e)
        return candidate
    out = (out or "").strip()
    # Strip any accidental fences.
    import re
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", out, re.DOTALL | re.IGNORECASE)
    if m:
        out = m.group(1).strip()
    return out or candidate


# ----------------------------------------------------------------------
# Per-row pipeline driver
# ----------------------------------------------------------------------

def run_pipeline_one(
    row: Dict[str, Any],
    base_raw: str,
    teacher,
    skill_catalog_text: str,
) -> Dict[str, Any]:
    """Run Stage 2 + Stage 4 for one row. Returns full per-stage trace."""
    candidate_v1 = CodeAnswerEvaluator.extract(base_raw)
    if not candidate_v1.strip():
        return {
            "sample_id": row["sample_id"],
            "stage": "stage1_empty",
            "candidate_v1": "",
            "selected_skills": [],
            "candidate_v2": "",
            "final_code": "",
        }

    selected_skills, candidate_v2 = teacher_review(
        teacher, row["question"], candidate_v1, skill_catalog_text,
    )
    final_code = teacher_final(teacher, row["question"], candidate_v2)
    return {
        "sample_id": row["sample_id"],
        "stage": "stage4_done",
        "candidate_v1": candidate_v1,
        "selected_skills": selected_skills,
        "candidate_v2": candidate_v2,
        "final_code": final_code,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-adapter", default=None)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "code"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--score-workers", type=int, default=8)
    p.add_argument("--teacher-workers", type=int, default=8,
                   help="Parallel PF helper API calls per dataset")
    p.add_argument("--teacher-provider", default="openai")
    p.add_argument("--teacher-model", default="")
    p.add_argument("--skill-dir", default=str(
        PROJECT_ROOT / "skills" / "code"))
    p.add_argument("--resume", action="store_true",
                   help="Resume from prior raw / stages outputs if present")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Reuse existing raw/{ds}.jsonl from prior run_vllm")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s")

    out_root = Path(args.out_dir)
    raw_dir = out_root / "raw"
    stages_dir = out_root / "stages"
    scored_dir = out_root / "scored"

    # ---- Stage 1: base-model direct (vLLM) -----------------------------
    stage1_raws: Dict[str, Dict[str, str]] = {}  # ds → {sid: raw}
    by_dataset: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in DATASETS}
    for ds, _v, rows in iter_subsets(args.data_dir):
        by_dataset[ds].extend(rows)

    if args.skip_stage1:
        logger.info("--skip-stage1: reusing %s/{ds}.jsonl", raw_dir)
        for ds in DATASETS:
            f = raw_dir / f"{ds}.jsonl"
            if not f.exists():
                logger.error("Missing %s — cannot skip Stage 1", f)
                sys.exit(1)
            d = {}
            with open(f) as fp:
                for ln in fp:
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    d[r["sample_id"]] = r.get("raw", "")
            stage1_raws[ds] = d
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        tokenizer_src = args.tokenizer_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

        llm_kwargs: Dict[str, Any] = dict(
            model=args.model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        lora_req = None
        if args.lora_adapter:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            lora_req = LoRARequest("sft_vanilla", 1, args.lora_adapter)
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )

        for ds, rows in by_dataset.items():
            if not rows:
                continue
            raw_path = raw_dir / f"{ds}.jsonl"
            prior: Dict[str, str] = {}
            if args.resume and raw_path.exists():
                with open(raw_path) as f:
                    for ln in f:
                        if not ln.strip():
                            continue
                        d = json.loads(ln)
                        if d.get("raw"):
                            prior[d["sample_id"]] = d["raw"]
            todo = [r for r in rows if r["sample_id"] not in prior]
            if todo:
                prompts = []
                for r in todo:
                    msgs = build_messages(r)
                    prompt_str = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt_str)
                gen_kwargs: Dict[str, Any] = {"sampling_params": sampling}
                if lora_req is not None:
                    gen_kwargs["lora_request"] = lora_req
                outs = llm.generate(prompts, **gen_kwargs)
                for r, o in zip(todo, outs):
                    prior[r["sample_id"]] = o.outputs[0].text if o.outputs else ""
            ordered = [{"sample_id": r["sample_id"], "raw": prior.get(r["sample_id"], "")}
                       for r in rows]
            write_jsonl(raw_path, ordered)
            stage1_raws[ds] = prior
            logger.info("[%s] stage1 done: %d raws", ds, len(prior))
        # Free vLLM GPU memory before PF helper API calls (CPU-bound).
        del llm

    # ---- Stage 2 + 4: helper review and final ------------------------
    teacher, _ = load_model_api(
        provider=args.teacher_provider,
        model_name=args.teacher_model,
        max_tokens=2048,
        temperature=0.0,
    )
    skill_catalog = load_skill_catalog(Path(args.skill_dir))
    skill_catalog_text = format_skill_catalog_for_prompt(skill_catalog)
    logger.info("Loaded %d skill catalog entries from %s",
                len(skill_catalog), args.skill_dir)

    summary: Dict[str, Dict[str, float]] = {}
    for ds, rows in by_dataset.items():
        if not rows:
            continue
        logger.info("[%s] running Stage 2+4 over %d rows (workers=%d)",
                    ds, len(rows), args.teacher_workers)
        raws = stage1_raws.get(ds, {})

        stage_path = stages_dir / f"{ds}.jsonl"
        # Resume support: skip rows already in the stage trace.
        prior_stage: Dict[str, Dict[str, Any]] = {}
        if args.resume and stage_path.exists():
            with open(stage_path) as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    d = json.loads(ln)
                    prior_stage[d["sample_id"]] = d

        results: Dict[str, Dict[str, Any]] = dict(prior_stage)
        todo = [r for r in rows if r["sample_id"] not in results]
        if todo:
            with ThreadPoolExecutor(max_workers=args.teacher_workers) as pool:
                futs = {
                    pool.submit(
                        run_pipeline_one, r, raws.get(r["sample_id"], ""),
                        teacher, skill_catalog_text,
                    ): r["sample_id"] for r in todo
                }
                for fut in as_completed(futs):
                    sid = futs[fut]
                    try:
                        results[sid] = fut.result()
                    except Exception as e:
                        logger.warning("[%s] %s pipeline error: %s", ds, sid, e)
                        results[sid] = {
                            "sample_id": sid,
                            "stage": "error",
                            "candidate_v1": CodeAnswerEvaluator.extract(raws.get(sid, "")),
                            "final_code": CodeAnswerEvaluator.extract(raws.get(sid, "")),
                        }
        ordered = [results[r["sample_id"]] for r in rows]
        write_jsonl(stage_path, ordered)

        # ---- Score Stage-4 final code -----------------------------------
        finals = [results[r["sample_id"]].get("final_code", "") for r in rows]
        # `score_batch` expects a list of raw outputs and re-extracts via
        # CodeAnswerEvaluator. Wrap each final_code in a fenced block so the
        # extractor returns it cleanly.
        wrapped = [f"```python\n{c}\n```" for c in finals]
        scored = score_batch(rows, wrapped, workers=args.score_workers)
        for s, r in zip(scored, rows):
            s["variant"] = r.get("variant")
        write_jsonl(scored_dir / f"{ds}.jsonl", scored)

        per_v: Dict[str, List[Dict[str, Any]]] = {}
        for s in scored:
            per_v.setdefault(s["variant"], []).append(s)
        summary[ds] = {v: pass_at_1(items) for v, items in sorted(per_v.items())}
        logger.info("[%s] pass@1 by variant: %s", ds, summary[ds])

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "lora_adapter": args.lora_adapter,
            "teacher_provider": args.teacher_provider,
            "teacher_model": args.teacher_model,
            "skill_dir": args.skill_dir,
            "n_skills": len(skill_catalog),
            "data_dir": str(args.data_dir),
            "subsets": summary,
        }, f, indent=2)
    logger.info("Wrote summary to %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
