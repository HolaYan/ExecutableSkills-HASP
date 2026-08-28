"""What a proposed skill looks like before it is a skill.

`CandidateSkill` is a proposal — the failure it targets, when it should fire,
what it would do, and the SKILL.md and code it would become. `ReviewResult` is
a judgement of one: five dimensions, a composite, and a decision.

They live here rather than inside the module that happens to produce them,
because several parts of training read them as a data format — loading a
finished proposal round, scoring it, turning accepted candidates into training
data — and none of those should import a proposal pipeline to get a type.

`skills_construct/forge/spec.py::PFSpec` is the other representation of a
proposed skill, used by the forge pipeline. The two exist because they came
from different generations of this work; a candidate carries prose rationale
for a reviewer to read, a PFSpec carries a checker for a gate to run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateSkill:
    """A candidate skill proposed by the student model."""
    skill_id: str
    name: str
    category: str

    # Part 1: Failure abstraction
    target_failure_pattern: str = ""
    failure_description: str = ""

    # Part 2: Trigger design
    trigger_conditions: List[str] = field(default_factory=list)
    trigger_rationale: str = ""

    # Part 3: Intervention design
    intervention_type: str = ""  # "modify_action" or "inject_context"
    intervention_description: str = ""
    intervention_rationale: str = ""

    # Part 4: Generated artifacts
    md_spec: str = ""      # SKILL.md content
    pf_code: str = ""      # Python PF class code

    # Raw student model response (preserved for audit)
    raw_response: str = ""

    # Metadata
    source_cluster_id: str = ""
    source_evidence_ids: List[str] = field(default_factory=list)
    novelty_explanation: str = ""
    expected_gain: str = ""

    # Review results (filled by SkillReviewer)
    review_scores: Dict[str, float] = field(default_factory=dict)
    review_decision: str = ""  # "accept" / "revise" / "reject"
    review_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "category": self.category,
            "target_failure_pattern": self.target_failure_pattern,
            "trigger_conditions": self.trigger_conditions,
            "intervention_type": self.intervention_type,
            "intervention_description": self.intervention_description,
            "md_spec": self.md_spec,
            "pf_code": self.pf_code,
            "source_cluster_id": self.source_cluster_id,
            "raw_response": self.raw_response,
            "novelty_explanation": self.novelty_explanation,
            "expected_gain": self.expected_gain,
            "review_scores": self.review_scores,
            "review_decision": self.review_decision,
            "review_feedback": self.review_feedback,
        }


@dataclass
class ReviewResult:
    """Result of helper review for a candidate skill."""
    skill_id: str
    # 5-dimensional scores (0-1)
    q_concept: float = 0.0
    q_trigger: float = 0.0
    q_intervene: float = 0.0
    q_exec: float = 0.0
    q_val: float = 0.0
    # Weighted composite
    q_skill: float = 0.0
    # Decision
    decision: str = "reject"  # accept / revise / reject
    # Detailed feedback
    feedback: str = ""
    # Per-dimension feedback
    concept_feedback: str = ""
    trigger_feedback: str = ""
    intervene_feedback: str = ""
    exec_feedback: str = ""
    val_feedback: str = ""
    # Raw PF helper response (preserved for audit)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "q_concept": self.q_concept,
            "q_trigger": self.q_trigger,
            "q_intervene": self.q_intervene,
            "q_exec": self.q_exec,
            "q_val": self.q_val,
            "q_skill": self.q_skill,
            "decision": self.decision,
            "feedback": self.feedback,
            "concept_feedback": self.concept_feedback,
            "trigger_feedback": self.trigger_feedback,
            "intervene_feedback": self.intervene_feedback,
            "exec_feedback": self.exec_feedback,
            "val_feedback": self.val_feedback,
            "raw_response": self.raw_response,
        }
