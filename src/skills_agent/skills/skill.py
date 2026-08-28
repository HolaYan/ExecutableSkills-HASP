"""
Skill data structure and SkillLibrary for JSON-based CRUD operations.

A Skill encapsulates knowledge about a specific type of error and
strategies to avoid it. The SkillLibrary manages persistence via JSON files.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json
import copy
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PhaseInstruction:
    """A phase-specific instruction block for a skill.

    Injected at specific ReAct decision points (post_search, post_read, pre_final)
    when the given conditions are met.

    Two modes of operation:
    - **Text injection** (V2): ``instruction`` is appended to the observation.
    - **Programmatic action** (V3): ``action`` triggers a direct action override
      (e.g., force READ instead of another SEARCH). When ``action`` is set,
      ``instruction`` is ignored for prompt injection and the system directly
      executes the specified action.
    """

    conditions: List[str] = field(default_factory=list)
    # e.g., ["search_has_conflicts", "result_count_gte_2"]
    instruction: str = ""
    # 3-8 sentence operational guide (text injection, V2)
    action: Optional[str] = None
    # Programmatic action ID (V3): "force_read_best_doc", "reformulate_search",
    # "force_final", "postprocess_answer"
    action_params: Dict[str, Any] = field(default_factory=dict)
    priority_boost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "conditions": self.conditions,
            "instruction": self.instruction,
            "priority_boost": self.priority_boost,
        }
        if self.action is not None:
            d["action"] = self.action
        if self.action_params:
            d["action_params"] = self.action_params
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseInstruction":
        return cls(
            conditions=data.get("conditions", []),
            instruction=data.get("instruction", ""),
            action=data.get("action"),
            action_params=data.get("action_params", {}),
            priority_boost=data.get("priority_boost", 0.0),
        )


@dataclass
class Skill:
    """A single error-avoidance skill."""

    skill_id: str  # e.g. "adversarial_distraction"
    version: int = 1
    name: str = ""  # Human-readable name
    error_category: str = ""  # Corresponding error category
    description: str = ""  # Brief description

    # Core content
    detection_triggers: List[str] = field(default_factory=list)
    avoidance_strategies: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    # Each example: {scenario, wrong_behavior, correct_behavior}

    # Phase-gated injection (new)
    system_summary: str = ""  # Compact 1-sentence description for system prompt priming
    phase_instructions: Dict[str, PhaseInstruction] = field(default_factory=dict)
    # Keyed by phase name: "post_search", "post_read", "pre_final"

    # Metadata
    priority: float = 0.5  # 0-1, used for ranking
    applicable_modes: List[str] = field(default_factory=lambda: ["all"])
    applicable_phases: List[str] = field(
        default_factory=lambda: ["think", "search", "read", "answer"]
    )

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "skill_id": self.skill_id,
            "version": self.version,
            "name": self.name,
            "error_category": self.error_category,
            "description": self.description,
            "detection_triggers": self.detection_triggers,
            "avoidance_strategies": self.avoidance_strategies,
            "examples": self.examples,
            "priority": self.priority,
            "applicable_modes": self.applicable_modes,
            "applicable_phases": self.applicable_phases,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.system_summary:
            d["system_summary"] = self.system_summary
        if self.phase_instructions:
            d["phase_instructions"] = {
                k: v.to_dict() for k, v in self.phase_instructions.items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        # Parse phase_instructions with backward compat (missing → empty dict)
        raw_pi = data.get("phase_instructions", {})
        phase_instructions = {
            k: PhaseInstruction.from_dict(v) for k, v in raw_pi.items()
        }

        return cls(
            skill_id=data["skill_id"],
            version=data.get("version", 1),
            name=data.get("name", ""),
            error_category=data.get("error_category", ""),
            description=data.get("description", ""),
            detection_triggers=data.get("detection_triggers", []),
            avoidance_strategies=data.get("avoidance_strategies", []),
            examples=data.get("examples", []),
            system_summary=data.get("system_summary", ""),
            phase_instructions=phase_instructions,
            priority=data.get("priority", 0.5),
            applicable_modes=data.get("applicable_modes", ["all"]),
            applicable_phases=data.get(
                "applicable_phases", ["think", "search", "read", "answer"]
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

class SkillLibrary:
    """JSON file-backed Skill library with CRUD operations."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._skills: Dict[str, Skill] = {}
        if self.path.exists():
            self.load()

    @classmethod
    def load_from_directory(cls, skills_dir: str) -> "SkillLibrary":
        """Create a SkillLibrary from a directory of ``{skill_id}/SKILL.md`` folders.

        Uses :class:`~skills_agent.skills.loader.MarkdownSkillLoader` to parse
        each SKILL.md file. The library is in-memory only (no JSON file backing).
        """
        from .loader import MarkdownSkillLoader

        skills = MarkdownSkillLoader.load_directory(skills_dir)
        lib = cls.__new__(cls)
        lib.path = Path(skills_dir)
        lib._skills = {s.skill_id: s for s in skills}
        logger.info("Loaded %d skills from directory %s", len(lib._skills), skills_dir)
        return lib

    def load(self) -> None:
        """Load skills from JSON file."""
        if not self.path.exists():
            logger.warning(f"Skill library file not found: {self.path}")
            return

        # Handle empty files gracefully
        if self.path.stat().st_size == 0:
            logger.warning(f"Skill library file is empty: {self.path}")
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        skills_data = data if isinstance(data, list) else data.get("skills", [])
        self._skills = {}
        for item in skills_data:
            skill = Skill.from_dict(item)
            self._skills[skill.skill_id] = skill

        logger.info(f"Loaded {len(self._skills)} skills from {self.path}")

    def save(self) -> None:
        """Save skills to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"skills": [s.to_dict() for s in self._skills.values()]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self._skills)} skills to {self.path}")

    def get(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_all(self) -> List[Skill]:
        """Get all skills."""
        return list(self._skills.values())

    def update(self, skill: Skill) -> None:
        """Update an existing skill."""
        if skill.skill_id not in self._skills:
            raise KeyError(f"Skill not found: {skill.skill_id}")
        self._skills[skill.skill_id] = skill

    def add(self, skill: Skill) -> None:
        """Add a new skill."""
        if skill.skill_id in self._skills:
            raise KeyError(f"Skill already exists: {skill.skill_id}")
        self._skills[skill.skill_id] = skill

    def add_or_update(self, skill: Skill) -> None:
        """Add or update a skill."""
        self._skills[skill.skill_id] = skill

    def remove(self, skill_id: str) -> None:
        """Remove a skill by ID."""
        if skill_id in self._skills:
            del self._skills[skill_id]

    def snapshot(self) -> str:
        """Return a JSON snapshot of the current library (for experiment reproducibility)."""
        data = {"skills": [s.to_dict() for s in self._skills.values()]}
        return json.dumps(data, ensure_ascii=False, indent=2)

    def clone(self) -> "SkillLibrary":
        """Create a deep copy of this library (in-memory, no file backing)."""
        new_lib = SkillLibrary.__new__(SkillLibrary)
        new_lib.path = self.path
        new_lib._skills = {
            k: Skill.from_dict(v.to_dict()) for k, v in self._skills.items()
        }
        return new_lib

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills
