"""
Skill proposer — Student model generates candidate skills (SKILL.md + PF code)
from failure clusters.

The student generates structured 4-part proposals:
  1. Failure abstraction (what error pattern does this address)
  2. Trigger design (when should the PF activate)
  3. Intervention design (what the PF does)
  4. PF code (executable Python matching ProgramFunction interface)
"""

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from .failure_analyzer import FailureCluster, FailurePattern

logger = logging.getLogger(__name__)


# The proposal record moved to skills_construct/candidate.py: several parts of
# training read it as a data format and should not import this pipeline for a
# type. Re-exported because this module's API is built on it.
from skills_construct.candidate import CandidateSkill  # noqa: F401



# ======================================================================
# Prompt templates
# ======================================================================

SKILL_PROPOSAL_SYSTEM = """\
You are a skill designer for a ReAct web search agent.

The agent operates in a SEARCH → READ → FINAL loop. At each step, Program Functions (PFs) \
can check conditions and intervene by modifying the agent's action or injecting context.

Your job: given a cluster of recurring failures, propose a NEW skill that will prevent \
this failure pattern in the future. The skill consists of:
1. A SKILL.md specification (YAML frontmatter + markdown body)
2. A ProgramFunction Python class with should_activate() and intervene() methods

IMPORTANT RULES:
- The PF must be deterministic (no LLM calls in should_activate).
- should_activate receives: step_context (dict), action_type (str: "SEARCH"/"READ"/"FINAL"), arg (str).
- intervene receives the same + optional PF helper, returns an Intervention.
- The PF class must inherit from ProgramFunction and use @register_pf decorator.
- should_activate must return bool. intervene must return an Intervention object.
- step_context contains: question, step_count, has_read, search_count, read_count, \
empty_results, contradictory_sources, max_steps, action_history, last_search_results_text, \
all_read_contents, thought (agent reasoning).

EXACT Intervention interface (use ONLY these keyword arguments):

    class InterventionType(Enum):
        NOOP = "noop"
        MODIFY_ACTION = "modify_action"
        INJECT_CONTEXT = "inject_context"

    @dataclass
    class Intervention:
        type: InterventionType = InterventionType.NOOP
        new_action_type: Optional[str] = None      # For MODIFY_ACTION: "SEARCH", "READ", or "FINAL"
        new_action_arg: Optional[str] = None        # For MODIFY_ACTION: the new argument
        context_text: str = ""                      # For INJECT_CONTEXT: text to inject
        reason: str = ""                            # Why this intervention was triggered
        skill_id: str = ""                          # Must match the @register_pf id

Example MODIFY_ACTION intervention (blocks premature FINAL, forces SEARCH):

    return Intervention(
        type=InterventionType.MODIFY_ACTION,
        new_action_type="SEARCH",
        new_action_arg="reformulated query here",
        reason="Agent tried FINAL without reading any documents",
        skill_id="my_skill_id",
    )

Example INJECT_CONTEXT intervention:

    return Intervention(
        type=InterventionType.INJECT_CONTEXT,
        context_text="WARNING: You have not verified this claim against any source.",
        reason="No source verification detected",
        skill_id="my_skill_id",
    )

DO NOT use any keyword arguments not listed above (e.g. do NOT use 'new_action', 'action_type', etc.).
"""

SKILL_PROPOSAL_PROMPT = """\
## Failure Cluster

**Cluster ID:** {cluster_id}
**Category:** {suggested_category}
**Frequency:** {total_frequency} occurrences
**Is new category:** {is_new_category}

### Failure Patterns:
{pattern_descriptions}

### Representative Examples:
{representative_examples}

### Existing Skills in This Area:
{existing_skills}

---

## Your Task

Propose a new skill to address this failure cluster. Output EXACTLY this format:

### Part 1: Failure Abstraction
```
This skill addresses failures where ...
Typical trajectory pattern: ...
Existing skills fail because ...
```

### Part 2: Trigger Design
```
Activate when:
- condition 1 (referencing specific step_context fields)
- condition 2
Do NOT activate when:
- exclusion condition
Minimum evidence required: ...
```

### Part 3: Intervention Design
```
Intervention type: MODIFY_ACTION or INJECT_CONTEXT
If activated:
- specific action to take
Only under: ...
Avoid intervening when: ...
```

### Part 4: SKILL.md
```yaml
---
skill_id: {suggested_skill_id}
name: "Descriptive Name"
version: 1
priority: 0.8
error_category: "{suggested_category}"
system_summary: "One-line summary of what this skill does"
detection_triggers:
  - trigger1
  - trigger2
avoidance_strategies:
  - strategy1
  - strategy2
applicable_modes: ["clean"]
phases:
  pre_final:
    conditions: ["always"]
    action: "verify_{suggested_skill_id}"
---
# Skill Name
Detailed description...
```

### Part 5: PF Code
```python
@register_pf("{suggested_skill_id}")
class SomePF(ProgramFunction):
    def should_activate(self, step_context, action_type, arg):
        # Deterministic check using step_context fields
        # Must return True or False
        if action_type == "FINAL" and not step_context.get("has_read", False):
            return True
        return False

    def intervene(self, step_context, action_type, arg, helper=None):
        # Must return an Intervention object
        # Use ONLY these kwargs: type, new_action_type, new_action_arg, context_text, reason, skill_id
        return Intervention(
            type=InterventionType.MODIFY_ACTION,
            new_action_type="SEARCH",
            new_action_arg="reformulated query",
            reason="Explanation of why this intervention is needed",
            skill_id="{suggested_skill_id}",
        )
```
"""


class SkillProposer:
    """Uses student model to generate candidate skills from failure clusters."""

    def __init__(
        self,
        student_model,  # APIModelWrapper
        existing_skill_ids: List[str],
        max_candidates: int = 5,
        temperature: float = 0.7,
        output_dir: Optional[str] = None,
    ):
        self.student_model = student_model
        self.existing_skill_ids = existing_skill_ids
        self.max_candidates = max_candidates
        self.temperature = temperature
        self.output_dir = Path(output_dir) if output_dir else None

    def propose(self, clusters: List[FailureCluster]) -> List[CandidateSkill]:
        """Generate candidate skills for the top failure clusters."""
        candidates = []

        # Sort clusters by frequency, take top N
        sorted_clusters = sorted(clusters, key=lambda c: c.total_frequency, reverse=True)
        target_clusters = sorted_clusters[:self.max_candidates]

        for cluster in target_clusters:
            # NOTE: Do NOT skip clusters just because they map to existing categories.
            # The fact that failures persist despite existing skills means the current
            # skills are insufficient — that's exactly what we want to improve.
            # We log coverage info but always attempt a proposal.

            candidate = self._propose_for_cluster(cluster)
            if candidate:
                candidates.append(candidate)
                logger.info("Proposed skill: %s (%s)", candidate.skill_id, candidate.name)

        if self.output_dir:
            self._save_candidates(candidates)

        return candidates

    def _propose_for_cluster(self, cluster: FailureCluster) -> Optional[CandidateSkill]:
        """Generate a single candidate skill for a failure cluster."""
        # Build prompt context
        pattern_descs = "\n".join(
            f"- **{p.category}** (freq={p.frequency}): {p.description}"
            for p in cluster.patterns
        )

        representative = []
        for p in cluster.patterns[:3]:
            for step in p.representative_steps[:2]:
                representative.append(json.dumps(step, indent=2)[:500])
        rep_text = "\n".join(representative) if representative else "(no detailed examples)"

        # Find existing skills in this category
        existing_in_cat = []
        for group_name, skill_ids in {
            "exploration_control": ["insufficient_exploration", "retrieval_failure", "search_depth_controller"],
            "reasoning_guard": ["hallucination", "reasoning_error", "multi_hop_reasoning_failure"],
            "entity_verification": ["wrong_entity_confusion", "temporal_confusion", "numerical_reasoning_error"],
            "query_strategy": ["query_decomposition", "constraint_search", "iterative_refinement"],
            "format_output": ["format_extraction_error", "answer_completeness", "citation_mismatch"],
            "information_synthesis": ["claim_triangulation", "evidence_synthesis", "misinformation_detector"],
        }.items():
            if group_name == cluster.suggested_category:
                existing_in_cat = [s for s in skill_ids if s in self.existing_skill_ids]
                break

        existing_text = ", ".join(existing_in_cat) if existing_in_cat else "(none)"

        # Suggest a skill_id
        suggested_id = self._suggest_skill_id(cluster)

        prompt = SKILL_PROPOSAL_PROMPT.format(
            cluster_id=cluster.cluster_id,
            suggested_category=cluster.suggested_category or "uncategorized",
            total_frequency=cluster.total_frequency,
            is_new_category=cluster.is_new_category,
            pattern_descriptions=pattern_descs,
            representative_examples=rep_text,
            existing_skills=existing_text,
            suggested_skill_id=suggested_id,
        )

        # Call student model
        try:
            response = self.student_model.generate(
                prompt=prompt,
                system=SKILL_PROPOSAL_SYSTEM,
                temperature=self.temperature,
                max_tokens=3000,
            )
        except Exception as e:
            logger.error("Student model failed for cluster %s: %s", cluster.cluster_id, e)
            return None

        # Parse response into CandidateSkill
        candidate = self._parse_response(response, cluster, suggested_id)
        if candidate:
            candidate.raw_response = response
        return candidate

    def _parse_response(
        self,
        response: str,
        cluster: FailureCluster,
        suggested_id: str,
    ) -> Optional[CandidateSkill]:
        """Parse the student model's response into a CandidateSkill."""
        candidate = CandidateSkill(
            skill_id=suggested_id,
            name=suggested_id.replace("_", " ").title(),
            category=cluster.suggested_category or "uncategorized",
            source_cluster_id=cluster.cluster_id,
            source_evidence_ids=[
                eid for p in cluster.patterns for eid in p.evidence_ids[:5]
            ],
        )

        # Extract Part 1: Failure Abstraction
        part1 = self._extract_section(response, "Part 1", "Part 2")
        if part1:
            candidate.target_failure_pattern = part1[:500]
            candidate.failure_description = part1[:500]

        # Extract Part 2: Trigger Design
        part2 = self._extract_section(response, "Part 2", "Part 3")
        if part2:
            candidate.trigger_rationale = part2[:500]
            # Parse conditions
            conditions = re.findall(r"- (.+)", part2)
            candidate.trigger_conditions = conditions[:10]

        # Extract Part 3: Intervention Design
        part3 = self._extract_section(response, "Part 3", "Part 4")
        if part3:
            candidate.intervention_description = part3[:500]
            candidate.intervention_rationale = part3[:500]
            if "MODIFY_ACTION" in part3:
                candidate.intervention_type = "modify_action"
            elif "INJECT_CONTEXT" in part3:
                candidate.intervention_type = "inject_context"

        # Extract Part 4: SKILL.md
        md_match = re.search(r"```ya?ml\s*\n(---\n.*?\n---\n.*?)```", response, re.DOTALL)
        if md_match:
            candidate.md_spec = md_match.group(1).strip()
        else:
            # Try to find any YAML frontmatter
            yaml_match = re.search(r"(---\n.*?\n---)", response, re.DOTALL)
            if yaml_match:
                candidate.md_spec = yaml_match.group(1).strip()

        # Extract Part 5: PF Code
        pf_match = re.search(r"```python\s*\n(@register_pf.*?)```", response, re.DOTALL)
        if pf_match:
            candidate.pf_code = pf_match.group(1).strip()
        else:
            # Try any python code block with class definition
            code_match = re.search(r"```python\s*\n(class\s+\w+.*?)```", response, re.DOTALL)
            if code_match:
                candidate.pf_code = code_match.group(1).strip()

        # Extract novelty explanation
        novelty_match = re.search(r"existing skills fail because(.*?)(?:\n\n|\n###)", response, re.DOTALL | re.IGNORECASE)
        if novelty_match:
            candidate.novelty_explanation = novelty_match.group(1).strip()[:300]

        # Validate minimum content
        if not candidate.pf_code:
            logger.warning("No PF code extracted for cluster %s", cluster.cluster_id)
            return None

        return candidate

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Extract text between two section markers."""
        pattern = rf"###?\s*{start_marker}.*?\n(.*?)(?=###?\s*{end_marker}|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _suggest_skill_id(self, cluster: FailureCluster) -> str:
        """Generate a suggested skill_id from the cluster."""
        # Use the most frequent pattern's category
        if cluster.patterns:
            base = cluster.patterns[0].category
        else:
            base = cluster.suggested_category or "unknown"

        # Ensure it doesn't conflict with existing
        candidate_id = base.replace(" ", "_").lower()
        if candidate_id in self.existing_skill_ids:
            candidate_id = f"{candidate_id}_v2"
        counter = 2
        while candidate_id in self.existing_skill_ids:
            counter += 1
            candidate_id = f"{base}_v{counter}"

        return candidate_id

    def _save_candidates(self, candidates: List[CandidateSkill]) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "candidate_skills.json"
        with open(path, "w") as f:
            json.dump([c.to_dict() for c in candidates], f, indent=2, ensure_ascii=False)
        logger.info("Saved %d candidate skills to %s", len(candidates), path)
