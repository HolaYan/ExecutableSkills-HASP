"""
MarkdownSkillLoader — loads Skill objects from folder-per-skill Markdown files.

Each skill lives in ``configs/skills/{skill_id}/SKILL.md`` with YAML frontmatter
for machine-readable metadata and Markdown body for human-readable content.

All public methods are ``@staticmethod`` so the loader is stateless and thread-safe.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import logging
import re

import yaml

from .skill import Skill, PhaseInstruction

logger = logging.getLogger(__name__)

# Regex for splitting on ``## `` headings (keeps the heading text)
_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Phase heading pattern: ``## Phase: post_search``
_PHASE_HEADING_RE = re.compile(r"^Phase:\s*(.+)$")

# Example sub-heading pattern: ``### Example Title``
_EXAMPLE_HEADING_RE = re.compile(r"^### (.+)$", re.MULTILINE)


class MarkdownSkillLoader:
    """Load :class:`Skill` objects from ``SKILL.md`` files.

    Directory layout expected::

        skills_dir/
        ├── adversarial_distraction/SKILL.md
        ├── format_extraction_error/SKILL.md
        └── ...
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def load_directory(skills_dir: str) -> List[Skill]:
        """Scan *skills_dir* for ``{skill_id}/SKILL.md`` and return parsed Skills."""
        skills_path = Path(skills_dir)
        if not skills_path.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {skills_dir}")

        skills: List[Skill] = []
        for child in sorted(skills_path.iterdir()):
            md_file = child / "SKILL.md"
            if child.is_dir() and md_file.exists():
                try:
                    skill = MarkdownSkillLoader.load_single(str(md_file))
                    skills.append(skill)
                except Exception:
                    logger.exception("Failed to load %s", md_file)

        logger.info("Loaded %d skills from %s", len(skills), skills_dir)
        return skills

    @staticmethod
    def load_single(path: str) -> Skill:
        """Parse a single ``SKILL.md`` file and return a :class:`Skill`."""
        text = Path(path).read_text(encoding="utf-8")
        frontmatter, body = MarkdownSkillLoader._parse_frontmatter(text)
        body_data = MarkdownSkillLoader._parse_markdown_body(body)
        return MarkdownSkillLoader._build_skill(frontmatter, body_data)

    @staticmethod
    def compile_to_json(skills_dir: str, output: str) -> None:
        """Generate a ``default_skills.json``-compatible file from SKILL.md files."""
        skills = MarkdownSkillLoader.load_directory(skills_dir)
        data = {"skills": [s.to_dict() for s in skills]}
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Compiled %d skills to %s", len(skills), output)

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
        """Split ``---`` delimited YAML frontmatter from the Markdown body.

        Returns ``(frontmatter_dict, body_string)``.
        """
        content = content.lstrip("\ufeff")  # strip BOM if present
        stripped = content.strip()
        if not stripped.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter (---)")

        # Find the closing ``---``
        second = stripped.index("---", 3)
        yaml_text = stripped[3:second].strip()
        body = stripped[second + 3:].strip()

        frontmatter = yaml.safe_load(yaml_text) or {}
        return frontmatter, body

    @staticmethod
    def _parse_markdown_body(body: str) -> Dict[str, Any]:
        """Parse the Markdown body into structured sections.

        Returns a dict with keys:
        - ``description``: text under the H1 heading (before first ``## ``)
        - ``detection_triggers``: bullet list from ``## Detection Triggers``
        - ``avoidance_strategies``: bullet list from ``## Avoidance Strategies``
        - ``phase_instructions``: ``{phase_name: instruction_text}``
        - ``examples``: list of ``{scenario, wrong_behavior, correct_behavior}``
        """
        result: Dict[str, Any] = {
            "description": "",
            "detection_triggers": [],
            "avoidance_strategies": [],
            "phase_instructions": {},
            "examples": [],
        }

        # Split into sections by ``## `` headings
        sections = MarkdownSkillLoader._split_sections(body)

        for heading, content in sections:
            if heading is None:
                # Preamble: skip H1 title line, rest is description
                lines = content.strip().splitlines()
                desc_lines = []
                for line in lines:
                    if line.startswith("# "):
                        continue  # skip H1
                    desc_lines.append(line)
                result["description"] = "\n".join(desc_lines).strip()

            elif heading.lower() == "detection triggers":
                result["detection_triggers"] = MarkdownSkillLoader._parse_bullet_list(content)

            elif heading.lower() == "avoidance strategies":
                result["avoidance_strategies"] = MarkdownSkillLoader._parse_bullet_list(content)

            elif _PHASE_HEADING_RE.match(heading):
                phase_name = _PHASE_HEADING_RE.match(heading).group(1).strip()
                result["phase_instructions"][phase_name] = content.strip()

            elif heading.lower() == "examples":
                result["examples"] = MarkdownSkillLoader._parse_examples(content)

        return result

    @staticmethod
    def _build_skill(frontmatter: Dict[str, Any], body_data: Dict[str, Any]) -> Skill:
        """Merge frontmatter metadata and body content into a :class:`Skill`."""
        # Build phase_instructions by merging frontmatter conditions/boost with body text
        phase_instructions: Dict[str, PhaseInstruction] = {}
        fm_phases = frontmatter.get("phases", {})
        body_phases = body_data.get("phase_instructions", {})

        # Collect all phase names from both sources
        all_phases = set(fm_phases.keys()) | set(body_phases.keys())
        for phase_name in all_phases:
            fm_data = fm_phases.get(phase_name, {})
            instruction_text = body_phases.get(phase_name, "")
            phase_instructions[phase_name] = PhaseInstruction(
                conditions=fm_data.get("conditions", []),
                instruction=instruction_text,
                action=fm_data.get("action"),
                action_params=fm_data.get("action_params", {}),
                priority_boost=fm_data.get("priority_boost", 0.0),
            )

        # Use description from body, falling back to frontmatter
        description = body_data.get("description", "") or frontmatter.get("description", "")

        return Skill(
            skill_id=frontmatter["skill_id"],
            version=frontmatter.get("version", 1),
            name=frontmatter.get("name", ""),
            error_category=frontmatter.get("error_category", ""),
            description=description,
            detection_triggers=body_data.get("detection_triggers", []),
            avoidance_strategies=body_data.get("avoidance_strategies", []),
            examples=body_data.get("examples", []),
            system_summary=frontmatter.get("system_summary", ""),
            phase_instructions=phase_instructions,
            priority=frontmatter.get("priority", 0.5),
            applicable_modes=frontmatter.get("applicable_modes", ["all"]),
            applicable_phases=frontmatter.get("applicable_phases", ["think", "search", "read", "answer"]),
            created_at="",
            updated_at="",
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sections(body: str) -> List[Tuple[Optional[str], str]]:
        """Split body into ``[(heading_or_None, content), ...]`` by ``## `` headings."""
        parts: List[Tuple[Optional[str], str]] = []
        matches = list(_HEADING_RE.finditer(body))

        if not matches:
            # No headings at all — entire body is preamble
            parts.append((None, body))
            return parts

        # Preamble before the first heading
        if matches[0].start() > 0:
            parts.append((None, body[: matches[0].start()]))

        for i, m in enumerate(matches):
            heading = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            parts.append((heading, body[start:end]))

        return parts

    @staticmethod
    def _parse_bullet_list(text: str) -> List[str]:
        """Extract top-level ``- `` bullet items from text."""
        items: List[str] = []
        for line in text.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                items.append(stripped[2:].strip())
        return items

    @staticmethod
    def _parse_examples(text: str) -> List[Dict[str, str]]:
        """Parse example blocks from ``## Examples`` section.

        Each example starts with ``### Title`` and contains
        ``**Scenario:**``, ``**Wrong:**``, ``**Correct:**`` fields.
        """
        examples: List[Dict[str, str]] = []
        # Split by ### headings
        sub_matches = list(_EXAMPLE_HEADING_RE.finditer(text))
        if not sub_matches:
            return examples

        for i, m in enumerate(sub_matches):
            start = m.end()
            end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(text)
            block = text[start:end].strip()

            example: Dict[str, str] = {}
            example["scenario"] = MarkdownSkillLoader._extract_field(block, "Scenario")
            example["wrong_behavior"] = MarkdownSkillLoader._extract_field(block, "Wrong")
            example["correct_behavior"] = MarkdownSkillLoader._extract_field(block, "Correct")
            examples.append(example)

        return examples

    @staticmethod
    def _extract_field(block: str, field_name: str) -> str:
        """Extract ``**FieldName:** value`` from a text block."""
        pattern = re.compile(
            rf"\*\*{re.escape(field_name)}:\*\*\s*(.+?)(?=\n\*\*|\Z)",
            re.DOTALL,
        )
        match = pattern.search(block)
        return match.group(1).strip() if match else ""
