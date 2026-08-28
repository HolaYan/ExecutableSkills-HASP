"""
Prompt templates for skill injection into the ReAct agent loop.

Two main template categories:
1. System prompt injection — format selected skills for the system prompt
2. Step-level reminders — format skill reminders appended to observations
"""

from typing import List, Dict, Any, Optional, Tuple

from .skill import Skill, PhaseInstruction


# ============================================================================
# 1. System Prompt Injection Templates
# ============================================================================

SKILLS_SECTION_HEADER = """

## Error Avoidance Skills
Before each Thought, review these skills to avoid common mistakes:
"""

SKILL_TEMPLATE = """### Skill: {name}
**When to watch out**: {triggers}
**Strategies**:
{strategies}
"""


def format_skills_section(skills: List[Skill]) -> str:
    """
    Format selected skills as a section to append to the system prompt.

    Args:
        skills: List of selected skills

    Returns:
        Formatted skills section string
    """
    if not skills:
        return ""

    parts = [SKILLS_SECTION_HEADER.strip()]

    for skill in skills:
        triggers = "; ".join(skill.detection_triggers[:3])  # Limit triggers for brevity
        strategies = "\n".join(
            f"- {s}" for s in skill.avoidance_strategies[:4]  # Limit strategies
        )
        part = SKILL_TEMPLATE.format(
            name=skill.name,
            triggers=triggers,
            strategies=strategies,
        )
        parts.append(part.strip())

    return "\n\n".join(parts)


def format_skills_compact(skills: List[Skill]) -> str:
    """
    Compact format for skills (fewer tokens, for constrained contexts).

    Uses system_summary (preferred, ~15 tokens/skill) or falls back to
    first 2 avoidance strategies.  Total overhead: ~30 tokens/skill vs
    ~150 tokens/skill for the verbose format.

    Args:
        skills: List of selected skills

    Returns:
        Compact skills string
    """
    if not skills:
        return ""

    lines = ["## Search Strategy Guidelines:"]
    for skill in skills:
        # Prefer system_summary (concise one-liner from SKILL.md frontmatter)
        summary = getattr(skill, "system_summary", None)
        if summary:
            lines.append(f"- {summary.strip()}")
        else:
            top_strategies = "; ".join(skill.avoidance_strategies[:2])
            lines.append(f"- **{skill.name}**: {top_strategies}")

    return "\n".join(lines)


# ============================================================================
# 1b. Phase-Gated System Prompt (compact awareness priming)
# ============================================================================

def format_skills_awareness(skills: List[Skill]) -> str:
    """Format compact skill awareness block for system prompt.

    Uses system_summary (if available) for ~30 tokens/skill instead of
    the verbose format_skills_section (~60 tokens/skill).

    Args:
        skills: List of selected skills

    Returns:
        Compact awareness section string
    """
    if not skills:
        return ""

    lines = [
        "## Active Skills",
        "Detailed guidance will appear after relevant observations.",
    ]
    for skill in skills:
        summary = skill.system_summary
        if not summary:
            # Fallback: first avoidance strategy truncated
            summary = (
                skill.avoidance_strategies[0][:80]
                if skill.avoidance_strategies
                else skill.description[:80]
            )
        lines.append(f"- **{skill.name}**: {summary}")

    return "\n".join(lines)


def format_phase_instruction(skill_name: str, instruction: str) -> str:
    """Format a phase-specific instruction block to append to observation.

    Args:
        skill_name: Human-readable skill name
        instruction: The operational instruction text

    Returns:
        Formatted instruction block string
    """
    return f"\n\n[SKILL CHECK — {skill_name}]\n{instruction}"


# ============================================================================
# 2. Step-Level Reminder Templates
# ============================================================================

STEP_REMINDER_TEMPLATE = "\n[Skill Reminder: {reminder}]"


def format_step_reminder(skill: Skill, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Format a skill as a step-level reminder to append after an observation.

    Args:
        skill: The skill to remind about
        context: Optional context for dynamic reminder generation

    Returns:
        Formatted reminder string
    """
    # Choose the most relevant strategy based on context
    reminder = _select_reminder_text(skill, context)
    return STEP_REMINDER_TEMPLATE.format(reminder=reminder)


def _select_reminder_text(skill: Skill, context: Optional[Dict[str, Any]] = None) -> str:
    """Select the most relevant reminder text based on skill and context."""
    skill_reminders = {
        "adversarial_distraction": (
            "Multiple conflicting sources detected — cross-reference across 2+ "
            "independent sources before proceeding"
        ),
        "retrieval_failure": (
            "Search returned poor results — try reformulating the query with "
            "synonyms or decompose the question"
        ),
        "insufficient_exploration": (
            "Limited exploration so far — try more search queries and READ "
            "at least 2 documents before answering"
        ),
        "wrong_entity_confusion": (
            "Multiple similar entities found — verify you have the correct one "
            "by checking dates, locations, or other distinguishing attributes"
        ),
        "format_extraction_error": (
            "Remember to provide ONLY the requested entity as your answer — "
            "no full sentences or extra explanation"
        ),
        "reasoning_error": (
            "Complex reasoning required — verify each logical step explicitly "
            "before concluding"
        ),
        "reading_comprehension_error": (
            "Carefully re-read the key sentences and quote the exact text "
            "that answers the question"
        ),
        "hallucination": (
            "Ensure your answer is grounded in retrieved evidence — do not "
            "fabricate facts"
        ),
        "answer_completeness": (
            "Verify your answer addresses all parts of the question at the correct scope"
        ),
    }

    return skill_reminders.get(
        skill.skill_id,
        skill.avoidance_strategies[0] if skill.avoidance_strategies else "Review this skill's strategies"
    )
