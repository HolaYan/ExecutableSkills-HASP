"""
Skill reviewer — PF helper evaluates candidate skills on 5 quality dimensions.

Scoring dimensions:
  Q_concept   — Is the skill concept sound and generalizable?
  Q_trigger   — Is should_activate() well-designed?
  Q_intervene — Is intervene() appropriate and safe?
  Q_exec      — Is the PF code executable and interface-compliant?
  Q_val       — Will this skill likely improve validation performance?

Final: Q_skill = weighted sum, compared to acceptance threshold.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from .skill_proposer import CandidateSkill
from .configs import SkillReviewConfig

logger = logging.getLogger(__name__)


# The review record moved to skills_construct/candidate.py, alongside the
# candidate it judges. Re-exported for this module's own API.
from skills_construct.candidate import ReviewResult  # noqa: F401



REVIEW_SYSTEM = """\
You are an expert skill quality reviewer for a ReAct web search agent system.

The agent uses Program Functions (PFs) — deterministic Python hooks that run every step.
Each PF has:
  - should_activate(step_context, action_type, arg) → bool
  - intervene(step_context, action_type, arg, helper=None) → Intervention

Intervention types: MODIFY_ACTION (change action), INJECT_CONTEXT (add text to observation), NOOP.

step_context fields: question, step_count, has_read, search_count, read_count, \
empty_results, contradictory_sources, max_steps, action_history, last_search_results_text, \
all_read_contents, thought.

You must evaluate candidate skills rigorously across 5 dimensions.
"""

REVIEW_PROMPT = """\
## Candidate Skill

**Skill ID:** {skill_id}
**Name:** {name}
**Category:** {category}
**Target failure pattern:** {target_failure_pattern}

### Trigger Conditions:
{trigger_conditions}

### Intervention Design:
{intervention_description}

### SKILL.md:
```
{md_spec}
```

### PF Code:
```python
{pf_code}
```

### Source Failure Cluster:
- Cluster: {source_cluster_id}
- Frequency: {evidence_count} failures
- Novelty: {novelty_explanation}

---

## Review Instructions

Score each dimension from 0.0 to 1.0 and provide brief feedback.

### Dimension 1: Concept (Q_concept)
- Is this a real, recurrent failure type?
- Is it generalizable, not a single-case patch?
- Is it clearly distinct from existing skills?

### Dimension 2: Trigger (Q_trigger)
- Is should_activate() condition specific and deterministic?
- Does it depend on available step_context fields?
- Is it neither too broad (many false positives) nor too narrow?

### Dimension 3: Intervention (Q_intervene)
- Does intervene() appropriately address the failure?
- Could it cause harmful side effects?
- Is the intervention proportional (not over-blocking)?

### Dimension 4: Executability (Q_exec)
- Does the code follow ProgramFunction interface?
- Would it import without errors?
- Are return types correct (Intervention)?

### Dimension 5: Validation Utility (Q_val)
- Will this likely reduce the target failure type?
- Will it transfer to unseen questions?
- Will it interact well with existing PFs?

Respond in EXACTLY this format:

Q_concept: [0.0-1.0]
Concept feedback: [brief]

Q_trigger: [0.0-1.0]
Trigger feedback: [brief]

Q_intervene: [0.0-1.0]
Intervention feedback: [brief]

Q_exec: [0.0-1.0]
Executability feedback: [brief]

Q_val: [0.0-1.0]
Validation feedback: [brief]

DECISION: [ACCEPT / REVISE / REJECT]
OVERALL FEEDBACK: [1-3 sentences]
"""


class SkillReviewer:
    """PF helper reviews candidate skills with 5-dimensional quality scoring."""

    def __init__(
        self,
        teacher_model,  # APIModelWrapper
        config: SkillReviewConfig,
        existing_skill_ids: List[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.teacher_model = teacher_model
        self.config = config
        self.existing_skill_ids = set(existing_skill_ids or [])
        self.output_dir = Path(output_dir) if output_dir else None

    def review(self, candidates: List[CandidateSkill]) -> List[ReviewResult]:
        """Review all candidate skills and return results."""
        results = []
        for candidate in candidates:
            result = self._review_one(candidate)
            # Apply result back to candidate
            candidate.review_scores = {
                "q_concept": result.q_concept,
                "q_trigger": result.q_trigger,
                "q_intervene": result.q_intervene,
                "q_exec": result.q_exec,
                "q_val": result.q_val,
                "q_skill": result.q_skill,
            }
            candidate.review_decision = result.decision
            candidate.review_feedback = result.feedback
            results.append(result)

            logger.info(
                "Reviewed %s: Q=%.2f, decision=%s",
                candidate.skill_id, result.q_skill, result.decision,
            )

        if self.output_dir:
            self._save_reviews(results)

        return results

    def _review_one(self, candidate: CandidateSkill) -> ReviewResult:
        """Review a single candidate skill."""
        prompt = REVIEW_PROMPT.format(
            skill_id=candidate.skill_id,
            name=candidate.name,
            category=candidate.category,
            target_failure_pattern=candidate.target_failure_pattern[:300],
            trigger_conditions="\n".join(f"- {c}" for c in candidate.trigger_conditions),
            intervention_description=candidate.intervention_description[:300],
            md_spec=candidate.md_spec[:1000],
            pf_code=candidate.pf_code[:2000],
            source_cluster_id=candidate.source_cluster_id,
            evidence_count=len(candidate.source_evidence_ids),
            novelty_explanation=candidate.novelty_explanation[:200],
        )

        try:
            response = self.teacher_model.generate(
                prompt=prompt,
                system=REVIEW_SYSTEM,
                temperature=self.config.temperature,
                max_tokens=1500,
            )
        except Exception as e:
            logger.error("Helper review failed for %s: %s", candidate.skill_id, e)
            return ReviewResult(skill_id=candidate.skill_id, decision="reject",
                                feedback=f"Review failed: {e}")

        result = self._parse_review(candidate.skill_id, response)
        result.raw_response = response
        return result

    def _parse_review(self, skill_id: str, response: str) -> ReviewResult:
        """Parse PF helper's review response into structured scores."""
        result = ReviewResult(skill_id=skill_id)

        # Extract scores
        score_patterns = {
            "q_concept": r"Q_concept:\s*([\d.]+)",
            "q_trigger": r"Q_trigger:\s*([\d.]+)",
            "q_intervene": r"Q_intervene:\s*([\d.]+)",
            "q_exec": r"Q_exec:\s*([\d.]+)",
            "q_val": r"Q_val:\s*([\d.]+)",
        }

        for field_name, pattern in score_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = min(max(float(match.group(1)), 0.0), 1.0)
                setattr(result, field_name, score)

        # Extract per-dimension feedback
        feedback_patterns = {
            "concept_feedback": r"Concept feedback:\s*(.+?)(?=\n\nQ_|$)",
            "trigger_feedback": r"Trigger feedback:\s*(.+?)(?=\n\nQ_|$)",
            "intervene_feedback": r"Intervention feedback:\s*(.+?)(?=\n\nQ_|$)",
            "exec_feedback": r"Executability feedback:\s*(.+?)(?=\n\nQ_|$)",
            "val_feedback": r"Validation feedback:\s*(.+?)(?=\n\nDECISION|$)",
        }

        for field_name, pattern in feedback_patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                setattr(result, field_name, match.group(1).strip()[:300])

        # Compute weighted score
        result.q_skill = (
            self.config.weight_concept * result.q_concept
            + self.config.weight_trigger * result.q_trigger
            + self.config.weight_intervene * result.q_intervene
            + self.config.weight_exec * result.q_exec
            + self.config.weight_validation * result.q_val
        )

        # Extract decision
        decision_match = re.search(r"DECISION:\s*(ACCEPT|REVISE|REJECT)", response, re.IGNORECASE)
        if decision_match:
            result.decision = decision_match.group(1).lower()
        else:
            # Fall back to threshold
            if result.q_skill >= self.config.acceptance_threshold:
                result.decision = "accept"
            elif result.q_skill >= self.config.acceptance_threshold * 0.7:
                result.decision = "revise"
            else:
                result.decision = "reject"

        # Override: if Q_exec < 0.3, always reject (code won't run)
        if result.q_exec < 0.3:
            result.decision = "reject"
            result.feedback += " [Auto-rejected: Q_exec too low, code likely non-functional]"

        # Extract overall feedback
        feedback_match = re.search(r"OVERALL FEEDBACK:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            result.feedback = feedback_match.group(1).strip()[:500]

        return result

    def _save_reviews(self, results: List[ReviewResult]) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "skill_reviews.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        logger.info("Saved %d reviews to %s", len(results), path)
