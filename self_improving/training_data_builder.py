"""
Training data builder — converts pseudo-gradients and trajectories into
training samples for post-training (SFT, DPO, offline RL).

Three types of training data:
  1. Student action-correction data (SFT/DPO from PF rescues)
  2. PF helper skill-selection data (SFT from selection outcomes)
  3. Skill-generation / skill-judging data (SFT for student + PF helper)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .pseudo_gradient import StudentGradient, TeacherGradient
from training.signals.trajectory import EpisodeTrajectory
from .skill_proposer import CandidateSkill
from .skill_reviewer import ReviewResult

logger = logging.getLogger(__name__)


class TrainingDataBuilder:
    """Builds training data from self-improving pipeline outputs."""

    def __init__(self, output_dir: str, formats: List[str] = None):
        self.output_dir = Path(output_dir) / "training_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["sft", "dpo"]

    # ------------------------------------------------------------------
    # 1. Student action-correction data
    # ------------------------------------------------------------------

    def build_action_correction_data(
        self,
        student_grad: StudentGradient,
        epoch: int = 0,
    ) -> Dict[str, Path]:
        """Build SFT and DPO data from PF rescue action corrections.

        SFT: train student to directly produce the PF-corrected action.
        DPO: preference pair (corrected > proposed) weighted by advantage.
        """
        outputs = {}

        # SFT format
        if "sft" in self.formats:
            sft_samples = []
            for corr in student_grad.action_corrections:
                if corr["advantage"] <= 0:
                    continue  # Only use positive corrections

                sft_samples.append({
                    "instruction": self._build_action_instruction(corr),
                    "input": self._build_action_context(corr),
                    "output": self._build_corrected_output(corr),
                    "metadata": {
                        "source": "pf_rescue",
                        "advantage": corr["advantage"],
                        "pf_ids": corr["pf_ids"],
                        "epoch": epoch,
                    },
                })

            path = self._save_jsonl(f"student_action_sft_epoch{epoch}.jsonl", sft_samples)
            outputs["sft"] = path
            logger.info("Built %d SFT action-correction samples", len(sft_samples))

        # DPO format
        if "dpo" in self.formats:
            dpo_samples = []
            for corr in student_grad.action_corrections:
                if corr["advantage"] <= 0:
                    continue

                dpo_samples.append({
                    "prompt": self._build_action_context(corr),
                    "chosen": self._build_corrected_output(corr),
                    "rejected": self._build_proposed_output(corr),
                    "metadata": {
                        "source": "pf_rescue",
                        "advantage": corr["advantage"],
                        "epoch": epoch,
                    },
                })

            path = self._save_jsonl(f"student_action_dpo_epoch{epoch}.jsonl", dpo_samples)
            outputs["dpo"] = path
            logger.info("Built %d DPO action-correction samples", len(dpo_samples))

        # Risk avoidance SFT
        if "sft" in self.formats and student_grad.risk_signals:
            risk_samples = []
            for risk in student_grad.risk_signals:
                hint = risk.get("safer_action_hint", "")
                if not hint:
                    continue
                risk_samples.append({
                    "instruction": "You are a ReAct web search agent. Choose the safest next action.",
                    "input": self._build_risk_context(risk),
                    "output": f"Action: {hint}",
                    "metadata": {
                        "source": "risk_avoidance",
                        "risk_score": risk["risk_score"],
                        "epoch": epoch,
                    },
                })

            path = self._save_jsonl(f"student_risk_sft_epoch{epoch}.jsonl", risk_samples)
            outputs["risk_sft"] = path
            logger.info("Built %d risk-avoidance SFT samples", len(risk_samples))

        return outputs

    # ------------------------------------------------------------------
    # 2. PF helper skill-selection data
    # ------------------------------------------------------------------

    def build_selection_data(
        self,
        teacher_grad: TeacherGradient,
        epoch: int = 0,
    ) -> Dict[str, Path]:
        """Build SFT data for PF helper's skill selection improvement."""
        outputs = {}

        if "sft" not in self.formats:
            return outputs

        sft_samples = []
        for sig in teacher_grad.selection_signals:
            # Keep any episode where at least one PF actually fired and
            # helped (useful_pf_ids non-empty). Success preferred but not
            # required — a firing PF on a failure episode still teaches
            # the selector that the PF is a candidate for this question.
            if not sig["useful_pf_ids"]:
                continue
            source = ("skill_selection_positive" if sig["episode_success"]
                      else "skill_selection_weak")
            sft_samples.append({
                "instruction": (
                    "You are a skill selector for a ReAct web search agent. "
                    "Given a question, select the most relevant Program Functions."
                ),
                "input": f"Question: {sig['question']}",
                "output": f"Selected PFs: {', '.join(sig['useful_pf_ids'])}",
                "metadata": {
                    "source": source,
                    "net_signal": sig["net_signal"],
                    "episode_success": sig["episode_success"],
                    "epoch": epoch,
                },
            })

        path = self._save_jsonl(f"teacher_selection_sft_epoch{epoch}.jsonl", sft_samples)
        outputs["selection_sft"] = path
        logger.info("Built %d PF helper selection SFT samples", len(sft_samples))
        return outputs

    # ------------------------------------------------------------------
    # 3. Skill-generation and skill-judging data
    # ------------------------------------------------------------------

    def build_skillgen_data(
        self,
        candidates: List[CandidateSkill],
        reviews: List[ReviewResult],
        epoch: int = 0,
    ) -> Dict[str, Path]:
        """Build training data for skill generation (student) and judging (PF helper)."""
        outputs = {}
        review_map = {r.skill_id: r for r in reviews}

        # Student: skill generation SFT
        if "sft" in self.formats:
            gen_samples = []
            for cand in candidates:
                review = review_map.get(cand.skill_id)
                if not review or review.q_skill < 0.5:
                    continue  # Only train on good skill proposals

                gen_samples.append({
                    "instruction": (
                        "You are a skill designer for a ReAct web search agent. "
                        "Given a failure pattern cluster, propose a new skill with "
                        "SKILL.md and ProgramFunction code."
                    ),
                    "input": f"Failure pattern: {cand.target_failure_pattern[:500]}",
                    "output": f"### SKILL.md\n```\n{cand.md_spec[:1000]}\n```\n\n"
                              f"### PF Code\n```python\n{cand.pf_code[:2000]}\n```",
                    "metadata": {
                        "source": "skill_generation",
                        "quality_score": review.q_skill,
                        "decision": review.decision,
                        "epoch": epoch,
                    },
                })

            if gen_samples:
                path = self._save_jsonl(f"student_skillgen_sft_epoch{epoch}.jsonl", gen_samples)
                outputs["skillgen_sft"] = path
                logger.info("Built %d skill-generation SFT samples", len(gen_samples))

        # PF helper: skill judging SFT
        if "sft" in self.formats:
            judge_samples = []
            for cand in candidates:
                review = review_map.get(cand.skill_id)
                if not review:
                    continue

                judge_samples.append({
                    "instruction": (
                        "You are an expert skill quality reviewer. "
                        "Evaluate this candidate skill on 5 dimensions."
                    ),
                    "input": (
                        f"Skill ID: {cand.skill_id}\n"
                        f"MD Spec:\n{cand.md_spec[:500]}\n"
                        f"PF Code:\n{cand.pf_code[:1000]}"
                    ),
                    "output": (
                        f"Q_concept: {review.q_concept:.2f}\n"
                        f"Q_trigger: {review.q_trigger:.2f}\n"
                        f"Q_intervene: {review.q_intervene:.2f}\n"
                        f"Q_exec: {review.q_exec:.2f}\n"
                        f"Q_val: {review.q_val:.2f}\n"
                        f"DECISION: {review.decision.upper()}\n"
                        f"OVERALL FEEDBACK: {review.feedback[:300]}"
                    ),
                    "metadata": {
                        "source": "skill_judging",
                        "q_skill": review.q_skill,
                        "epoch": epoch,
                    },
                })

            if judge_samples:
                path = self._save_jsonl(f"teacher_judging_sft_epoch{epoch}.jsonl", judge_samples)
                outputs["judging_sft"] = path
                logger.info("Built %d skill-judging SFT samples", len(judge_samples))

        # DPO for skill generation: accepted > rejected
        if "dpo" in self.formats:
            accepted = [c for c in candidates if review_map.get(c.skill_id, None) and review_map[c.skill_id].decision == "accept"]
            rejected = [c for c in candidates if review_map.get(c.skill_id, None) and review_map[c.skill_id].decision == "reject"]

            dpo_pairs = []
            for good in accepted:
                for bad in rejected:
                    dpo_pairs.append({
                        "prompt": f"Failure pattern: {good.target_failure_pattern[:300]}",
                        "chosen": f"```python\n{good.pf_code[:1500]}\n```",
                        "rejected": f"```python\n{bad.pf_code[:1500]}\n```",
                        "metadata": {
                            "source": "skill_generation_dpo",
                            "chosen_q": review_map[good.skill_id].q_skill,
                            "rejected_q": review_map[bad.skill_id].q_skill,
                            "epoch": epoch,
                        },
                    })
                    if len(dpo_pairs) >= 50:
                        break
                if len(dpo_pairs) >= 50:
                    break

            if dpo_pairs:
                path = self._save_jsonl(f"student_skillgen_dpo_epoch{epoch}.jsonl", dpo_pairs)
                outputs["skillgen_dpo"] = path
                logger.info("Built %d skill-generation DPO pairs", len(dpo_pairs))

        return outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_action_instruction(self, corr: Dict[str, Any]) -> str:
        return (
            "You are a ReAct web search agent. Based on the current context, "
            "choose the best next action (SEARCH/READ/FINAL)."
        )

    def _build_action_context(self, corr: Dict[str, Any]) -> str:
        ctx = corr.get("step_context", {})
        return (
            f"Question: {corr['question']}\n"
            f"Step: {corr['step_index']}\n"
            f"Searches done: {ctx.get('search_count', '?')}\n"
            f"Reads done: {ctx.get('read_count', '?')}\n"
            f"Has read: {ctx.get('has_read', '?')}\n"
            f"Last reasoning: {corr.get('proposed_reasoning', '')[:200]}"
        )

    def _build_corrected_output(self, corr: Dict[str, Any]) -> str:
        return f"Action: {corr['corrected_action']}({corr['corrected_arg'][:200]})"

    def _build_proposed_output(self, corr: Dict[str, Any]) -> str:
        return f"Action: {corr['proposed_action']}({corr['proposed_arg'][:200]})"

    def _build_risk_context(self, risk: Dict[str, Any]) -> str:
        ctx = risk.get("step_context", {})
        return (
            f"Question: {risk['question']}\n"
            f"Step: {risk['step_index']}\n"
            f"Current action: {risk['action']}({risk['arg'][:100]})\n"
            f"Searches done: {ctx.get('search_count', '?')}\n"
            f"Has read: {ctx.get('has_read', '?')}\n"
            f"Empty results: {ctx.get('empty_results', '?')}"
        )

    def _save_jsonl(self, filename: str, samples: List[Dict]) -> Path:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return path

    def get_summary(self, epoch: int = 0) -> Dict[str, Any]:
        """Summarize training data generated for an epoch."""
        summary = {"epoch": epoch, "files": {}}
        for f in self.output_dir.glob(f"*epoch{epoch}*"):
            count = sum(1 for _ in open(f))
            summary["files"][f.name] = count
        return summary
