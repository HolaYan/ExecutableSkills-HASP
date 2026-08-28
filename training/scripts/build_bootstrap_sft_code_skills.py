"""CODE-domain bootstrap with the new Skills+PF helper pipeline (Option B).

Replaces the old SkillAgentRunner-based rollout (`build_bootstrap_sft_code.py`)
which used ReAct + PF interventions. ReAct was empirically regressing the
code-domain student; the eval-side replacement is `run_vllm_skills.py`'s
direct + teacher pipeline. This script mirrors that flow so training data
matches the eval distribution.

Per problem:
  Stage 1: Qwen2.5-7B-Instruct direct answer (vLLM bf16, greedy)
              → candidate_v1
  Stage 2: GPT-4o reads the SKILL catalog + candidate, returns
              {selected_skills, improved_code}
  Stage 4: GPT-4o final review → final_code
  Score:   final_code vs eval_test_code → exact_match (pass@1)

Trajectory schema: 1 synthetic step per problem.
  - proposed_action_arg = candidate_v1 (base model's raw code)
  - final_action_arg    = final_code   (teacher's polished code)
  - was_modified        = (proposed != final)
  - pf_activations      = synthesized records, one per selected_skill
                          (so DPO chosen=teacher / rejected=base wires up).
This makes the resulting `objA_sft.jsonl` a teacher-distillation dataset:
the base model is trained to imitate the teacher's improved code on the
same problem distribution.

Test-set isolation: `data/code/{ds}.jsonl` rows 0-199 are testset (100
unique problems × 2 variants). We hard-code `test_reserve = 200` per
dataset so rollouts NEVER touch testset rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "scripts"))

from training.scripts.build_bootstrap_sft import (  # noqa: E402
    _load_done_sample_ids,
    build_sft,
    load_training_pool,
)
from training.rollout_baseline.run_vllm_skills import (  # noqa: E402
    load_skill_catalog,
    format_skill_catalog_for_prompt,
    teacher_review,
    teacher_final,
)

# Direct-answer chat prompt
from training.rollout_baseline.prompt import build_messages  # noqa: E402


def _rollout_helpers():
    return (load_skill_catalog, format_skill_catalog_for_prompt,
            teacher_review, teacher_final)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def synthesize_trajectory(
    sample: Dict[str, Any],
    candidate_v1: str,
    selected_skills: List[str],
    final_code: str,
    passed: bool,
    pass_rate: float,
    epoch: int = 0,
) -> Dict[str, Any]:
    """Build the EpisodeTrajectory dict for one problem.

    One synthetic step:
      proposed_action_arg = candidate_v1   (base model's direct answer)
      final_action_arg    = final_code     (PF helper's polished code)
      pf_activations      = synthesized from selected_skills (each as
                            modify_action with original=v1, modified=final)
    Downstream `UsePFsBuilder` will produce:
      • SFT target = PF helper's final_code (distillation)
      • DPO chosen=final_code / rejected=candidate_v1 (when modified)
    """
    from training.signals.trajectory import (
        EpisodeTrajectory, StepRecord, PFActivationRecord,
    )

    was_modified = (candidate_v1.strip() != final_code.strip())
    pf_recs: List[PFActivationRecord] = []
    for skill in selected_skills or []:
        pf_recs.append(PFActivationRecord(
            pf_id=skill,
            activated=True,
            intervention_type="modify_action" if was_modified else "noop",
            reason="teacher_selected" if was_modified else "skill_check_passed",
            original_action="FINAL" if was_modified else None,
            original_arg=candidate_v1 if was_modified else None,
            modified_action="FINAL" if was_modified else None,
            modified_arg=final_code if was_modified else None,
            injected_text=None,
        ))

    step = StepRecord(
        step_index=0,
        proposed_action_type="FINAL",
        proposed_action_arg=candidate_v1,
        proposed_reasoning="",
        final_action_type="FINAL",
        final_action_arg=final_code,
        was_modified=was_modified,
        pf_activations=pf_recs,
        context_injections=[],
        observation_summary="",
        step_context_snapshot={
            "domain": "code",
            "entry_point": sample.get("entry_point", ""),
            "variant": sample.get("variant", ""),
        },
    )
    traj = EpisodeTrajectory(
        sample_id=str(sample["sample_id"]),
        question=sample["question"],
        gold_answers=list(sample.get("gold_answers", [])),
        dataset_name=sample.get("dataset_name", ""),
        difficulty_score=0,
        skills_enabled=True,
        selected_pf_ids=sorted(set(selected_skills or [])),
        steps=[step],
        final_answer=final_code,
        exact_match=bool(passed),
        f1_score=float(pass_rate),
        epoch=epoch,
        ablation="skills_top10_teacher",
    )
    traj.compute_stats()
    return traj.to_dict()


def run_stage1_vllm(
    pool: List[Dict[str, Any]],
    model_path: str,
    tp_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, str]:
    """Stage 1: Qwen direct answer per pool row. Returns {sample_id: raw_text}."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
    )
    prompts = []
    for r in pool:
        msgs = build_messages(r)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))
    outs = llm.generate(prompts, sampling_params=sampling)
    raws = {r["sample_id"]: (o.outputs[0].text if o.outputs else "")
            for r, o in zip(pool, outs)}
    # Free vLLM memory before PF helper API calls.
    del llm
    import gc; gc.collect()
    return raws


def run_stages_2_4(
    pool: List[Dict[str, Any]],
    raws: Dict[str, str],
    teacher_provider: str,
    teacher_model: str,
    skill_catalog_text: str,
    workers: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """For each row: Stage 2 (review+select skills) + Stage 4 (final review).

    Returns {sample_id: {candidate_v1, selected_skills, candidate_v2, final_code}}.
    """
    from src.skills_agent.eval.model_loader import load_model_api
    from src.skills_agent.eval.metrics import CodeAnswerEvaluator

    teacher, _ = load_model_api(
        provider=teacher_provider, model_name=teacher_model,
        max_tokens=2048, temperature=0.0,
    )

    _, _, teacher_review, teacher_final = _rollout_helpers()

    def _one(row):
        sid = row["sample_id"]
        raw = raws.get(sid, "")
        v1 = CodeAnswerEvaluator.extract(raw)
        if not v1.strip():
            return sid, {"candidate_v1": "", "selected_skills": [],
                         "candidate_v2": "", "final_code": ""}
        skills, v2 = teacher_review(teacher, row["question"], v1, skill_catalog_text)
        v4 = teacher_final(teacher, row["question"], v2)
        return sid, {
            "candidate_v1": v1, "selected_skills": skills,
            "candidate_v2": v2, "final_code": v4,
        }

    out: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, r): r["sample_id"] for r in pool}
        for fut in as_completed(futs):
            sid, rec = fut.result()
            out[sid] = rec
    return out


def score_one(final_code: str, row: Dict[str, Any], sandbox) -> Dict[str, Any]:
    from src.skills_agent.eval.code_sandbox import CodeSandbox
    if not final_code.strip():
        return {"passed": False, "pass_rate": 0.0}
    eval_test_code = row.get("eval_test_code", "")
    entry_point = row.get("entry_point") or (row.get("metadata") or {}).get("entry_point", "")
    if not eval_test_code:
        return {"passed": False, "pass_rate": 0.0}
    try:
        res = sandbox.evaluate_with_test_script(final_code, eval_test_code, entry_point)
    except Exception:
        return {"passed": False, "pass_rate": 0.0}
    return {"passed": bool(res.pass_at_1), "pass_rate": float(res.pass_rate)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--datasets", nargs="+",
                    default=["humaneval_plus", "mbpp_plus", "bigcodebench"])
    ap.add_argument("--raw-dir", default="data/code")
    # Hard-coded testset reserve (rows 0-199 of every dataset jsonl).
    # Overrides default of 100 in build_bootstrap_sft.py.
    ap.add_argument("--test-reserve-overrides", type=str,
                    default='{"humaneval_plus":200,"mbpp_plus":200,"bigcodebench":200}',
                    help="MUST be 200 per dataset — testset is 100 unique × 2 variants.")
    ap.add_argument("--per-dataset-cap-overrides", type=str,
                    default='{"humaneval_plus":128,"mbpp_plus":300,"bigcodebench":300}',
                    help="Option B caps: HE+ uses full 128 train pool; MBPP/BCB sampled "
                         "to 300 each → 728 total → ~$30-50 in teacher API.")
    ap.add_argument("--traj-out", default="outputs/bootstrap_rollouts_code/")
    ap.add_argument("--sft-out", default="training/outputs/_shared_data_code/")
    ap.add_argument("--skill-dir", default="skills/code")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--teacher-provider", default="openai")
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--teacher-workers", type=int, default=16)
    ap.add_argument("--score-workers", type=int, default=8)
    ap.add_argument("--signals", default="all")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--skip-rollout", action="store_true")
    ap.add_argument("--skip-stage1", action="store_true",
                    help="Reuse existing raw/{ds}.jsonl from a prior run_vllm pass")
    ap.add_argument("--force-fresh", action="store_true")
    args = ap.parse_args()

    test_reserve_overrides = json.loads(args.test_reserve_overrides) if args.test_reserve_overrides else {}
    per_dataset_cap_overrides = json.loads(args.per_dataset_cap_overrides) if args.per_dataset_cap_overrides else {}

    pool = load_training_pool(
        args.datasets, args.raw_dir, test_reserve=200,
        per_dataset_cap=None,
        test_reserve_overrides=test_reserve_overrides,
        per_dataset_cap_overrides=per_dataset_cap_overrides,
    )
    logger.info("Total bootstrap pool: %d rows", len(pool))

    traj_path = Path(args.traj_out) / "epoch_0" / "trajectories" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force_fresh and traj_path.exists():
        logger.info("--force-fresh: removing %s", traj_path)
        traj_path.unlink()

    if args.skip_rollout:
        logger.info("--skip-rollout: reusing existing trajectories")
    else:
        done_ids = _load_done_sample_ids(traj_path)
        if done_ids:
            logger.info("Resume: %d sample_ids already in trajectories", len(done_ids))
        pending = [r for r in pool if str(r["sample_id"]) not in done_ids]
        logger.info("Pending: %d / %d", len(pending), len(pool))

        if pending:
            # ---- Stage 1: vLLM direct ----
            raws_dir = Path(args.traj_out) / "raw_stage1"
            raws_dir.mkdir(parents=True, exist_ok=True)
            raws_per_ds: Dict[str, Dict[str, str]] = {}
            if not args.skip_stage1:
                logger.info("Stage 1: vLLM direct answer (TP=%d)", args.tp_size)
                raws_all = run_stage1_vllm(
                    pending, args.model_path,
                    tp_size=args.tp_size,
                    gpu_memory_utilization=args.gpu_mem_util,
                    max_model_len=args.max_model_len,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )
                # Save Stage 1 outputs grouped by dataset for resume support.
                by_ds: Dict[str, List[Dict[str, str]]] = {}
                for r in pending:
                    by_ds.setdefault(r["dataset_name"], []).append({
                        "sample_id": r["sample_id"],
                        "raw": raws_all.get(r["sample_id"], ""),
                    })
                for ds, rows in by_ds.items():
                    with open(raws_dir / f"{ds}.jsonl", "w", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row) + "\n")
            else:
                logger.info("--skip-stage1: loading raws from %s", raws_dir)
                raws_all = {}
                for ds in args.datasets:
                    fp = raws_dir / f"{ds}.jsonl"
                    if not fp.exists():
                        continue
                    with open(fp) as f:
                        for ln in f:
                            d = json.loads(ln)
                            raws_all[d["sample_id"]] = d.get("raw", "")

            # ---- Stages 2 + 4: helper review/finalize ----
            logger.info("Stages 2+4: PF helper (%s)", args.teacher_model)
            load_skill_catalog, format_skill_catalog_for_prompt, _, _ = _rollout_helpers()
            skill_catalog = load_skill_catalog(Path(args.skill_dir))
            skill_catalog_text = format_skill_catalog_for_prompt(skill_catalog)
            stage_records = run_stages_2_4(
                pending, raws_all,
                teacher_provider=args.teacher_provider,
                teacher_model=args.teacher_model,
                skill_catalog_text=skill_catalog_text,
                workers=args.teacher_workers,
            )

            # ---- Score + write trajectories ----
            from src.skills_agent.eval.code_sandbox import CodeSandbox
            sandbox = CodeSandbox(cpu_seconds=20, wall_timeout_s=30.0)
            n_passed = 0
            with open(traj_path, "a", encoding="utf-8") as f:
                for r in pending:
                    rec = stage_records.get(r["sample_id"], {})
                    score = score_one(rec.get("final_code", ""), r, sandbox)
                    traj_dict = synthesize_trajectory(
                        sample=r,
                        candidate_v1=rec.get("candidate_v1", ""),
                        selected_skills=rec.get("selected_skills", []),
                        final_code=rec.get("final_code", ""),
                        passed=score["passed"],
                        pass_rate=score["pass_rate"],
                    )
                    f.write(json.dumps(traj_dict, ensure_ascii=False) + "\n")
                    if score["passed"]:
                        n_passed += 1
            logger.info("Bootstrap done: %d / %d passed", n_passed, len(pending))

    build_sft(
        traj_out_dir=args.traj_out,
        sft_out_dir=args.sft_out,
        enabled_signals=args.signals,
        threshold=args.threshold,
    )
    logger.info("Done. trajectories=%s, SFT=%s", args.traj_out, args.sft_out)


if __name__ == "__main__":
    main()
