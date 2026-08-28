"""Unified prompt templates for SFT / RS / GRPO."""

from __future__ import annotations

from typing import Any, Dict

SYSTEM_USE_PFS = (
    "You are a ReAct web search agent. Given the current question, history and "
    "step context, decide the next action. Available actions are SEARCH(query), "
    "READ(url_or_index), FINAL(answer). Respond with a single Action line only."
)

SYSTEM_EVOLVE = (
    "You are a skill designer for a ReAct agent framework. Given a cluster of "
    "failure patterns observed across episodes, propose exactly one new skill "
    "comprising (a) SKILL.md with trigger/intervention specification and (b) a "
    "ProgramFunction subclass implementing should_activate() and intervene() "
    "using the Intervention API. Return both sections in order."
)


def build_react_step_prompt(sample: Dict[str, Any]) -> str:
    """Prompt for an individual ReAct step (Objective A)."""
    ctx = sample.get("step_context", {}) or {}
    return (
        f"Question: {sample.get('question', '')[:400]}\n"
        f"Step index: {sample.get('step_index', '?')}\n"
        f"Searches done: {ctx.get('search_count', '?')}  "
        f"Reads done: {ctx.get('read_count', '?')}  "
        f"Has read: {ctx.get('has_read', '?')}\n"
        f"Empty results flag: {ctx.get('empty_results', '?')}\n"
        f"Prior reasoning: {sample.get('proposed_reasoning', '')[:300]}\n\n"
        f"Respond with a single line: Action: TYPE(arg)"
    )


def build_action_target(action_type: str, action_arg: str) -> str:
    return f"Action: {action_type}({action_arg[:300]})"


def build_skillgen_prompt(failure_pattern: str, cluster_samples: str = "") -> str:
    """Prompt for skill generation (Objective B)."""
    body = f"Failure pattern summary:\n{failure_pattern[:800]}\n"
    if cluster_samples:
        body += f"\nExample failures:\n{cluster_samples[:1200]}\n"
    body += (
        "\nPropose one new skill as:\n"
        "### SKILL.md\n```\n<concept, triggers, intervention spec>\n```\n"
        "### PF Code\n```python\n<ProgramFunction subclass>\n```"
    )
    return body


def build_skillgen_target(md_spec: str, pf_code: str) -> str:
    return (
        f"### SKILL.md\n```\n{md_spec[:1500]}\n```\n\n"
        f"### PF Code\n```python\n{pf_code[:2500]}\n```"
    )


def to_chat(system: str, user: str, assistant: str) -> Dict[str, Any]:
    """TRL chat-format sample."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def to_chat_prompt_only(system: str, user: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
