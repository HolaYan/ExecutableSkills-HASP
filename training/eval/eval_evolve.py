"""Evaluate Objective B — the student's ability to evolve.

Pipeline:
  1. Load held-out failure clusters (jsonl with `cluster_id` + `pattern_summary`).
  2. Prompt the trained student to generate a candidate skill via vLLM.
  3. Parse the generation into SKILL.md + PF code, wrap as `CandidateSkill`.
  4. Call `SkillReviewer.review(list)` to get 5-dim scores.
  5. Aggregate Q_skill distribution + acceptance rate → json summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


_MD_CODE_PAT = re.compile(
    r"###\s*SKILL\.md\s*(?:```)?\s*(.*?)(?:```)?\s*###\s*PF\s*Code\s*(?:```(?:python)?)?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def _split_md_code(generation: str):
    m = _MD_CODE_PAT.search(generation or "")
    if not m:
        return generation.strip(), ""
    md = m.group(1).strip().strip("`").strip()
    code = m.group(2).strip().rstrip("`").strip()
    return md, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clusters-jsonl", required=True, help="Jsonl with cluster_id + pattern_summary")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--acceptance-threshold", type=float, default=0.6)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from skills_construct.candidate import CandidateSkill
    from self_improving.skill_reviewer import SkillReviewer
    from self_improving.configs import SkillReviewConfig
    from training.rejection_sampling.rollout import Rollouter, RolloutConfig
    from training.data.prompt_templates import (
        SYSTEM_EVOLVE, build_skillgen_prompt, to_chat_prompt_only,
    )

    # 1. Build per-cluster prompts
    prompts_path = out_dir / "clusters_prompts.jsonl"
    with open(args.clusters_jsonl, "r", encoding="utf-8") as fin, \
         open(prompts_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            pattern = row.get("pattern_summary", "")[:1000]
            user = build_skillgen_prompt(pattern)
            out = to_chat_prompt_only(SYSTEM_EVOLVE, user)
            out["cluster_id"] = row.get("cluster_id", "")
            out["target_failure_pattern"] = pattern[:500]
            fout.write(json.dumps(out) + "\n")

    # 2. Generate skills with trained student
    rc = RolloutConfig(
        model_path=args.ckpt,
        prompts_path=str(prompts_path),
        output_dir=str(out_dir / "rollouts"),
        group_size=1,
        temperature=0.7,
        max_new_tokens=2048,
        mode="skill",
    )
    rollout_path = Rollouter(rc).run()

    # 3. Parse + wrap as CandidateSkill
    candidates = []
    with open(rollout_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            md, code = _split_md_code(r.get("generation", ""))
            cand = CandidateSkill(
                skill_id=f"eval_{r.get('cluster_id', 'x')}",
                name=f"eval_{r.get('cluster_id', 'x')}",
                category="eval",
                target_failure_pattern=r.get("target_failure_pattern", ""),
                md_spec=md,
                pf_code=code,
                raw_response=r.get("generation", ""),
            )
            candidates.append(cand)
    logger.info("Parsed %d candidates for review", len(candidates))

    # 4. Helper review
    reviewer = SkillReviewer(SkillReviewConfig(
        teacher_model=args.teacher_model,
        acceptance_threshold=args.acceptance_threshold,
    ))
    reviews = reviewer.review(candidates)

    # 5. Summary
    per_row = []
    for cand, review in zip(candidates, reviews):
        per_row.append({
            "cluster_id": cand.skill_id,
            "q_concept": review.q_concept,
            "q_trigger": review.q_trigger,
            "q_intervene": review.q_intervene,
            "q_exec": review.q_exec,
            "q_val": review.q_val,
            "q_skill": review.q_skill,
            "decision": review.decision,
        })
    n = max(len(per_row), 1)
    summary = {
        "n_clusters": len(per_row),
        "accept_rate": sum(1 for r in per_row if r["decision"] == "accept") / n,
        "avg_q_skill": sum(r["q_skill"] for r in per_row) / n,
        "avg_q_concept": sum(r["q_concept"] for r in per_row) / n,
        "avg_q_trigger": sum(r["q_trigger"] for r in per_row) / n,
        "avg_q_intervene": sum(r["q_intervene"] for r in per_row) / n,
        "avg_q_exec": sum(r["q_exec"] for r in per_row) / n,
        "avg_q_val": sum(r["q_val"] for r in per_row) / n,
        "per_cluster": per_row,
    }
    summary_path = out_dir / "evolve_eval.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evolve eval complete → %s", summary_path)


if __name__ == "__main__":
    main()
