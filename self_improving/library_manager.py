"""
Library manager — manages skill library evolution across epochs.

The library is stored entirely under ``self_improving/skills/``:

    self_improving/skills/
    ├── seed/                  # Initial skills copied from configs/skills/ (read-only reference)
    │   ├── hallucination/SKILL.md
    │   └── ...
    ├── generated/             # New skills produced during self-improving epochs
    │   ├── {new_skill_id}/
    │   │   ├── SKILL.md
    │   │   └── metadata.json
    │   └── ...
    ├── snapshots/             # Per-epoch full library snapshots
    │   ├── epoch_0/
    │   └── ...
    ├── dynamic_program_functions.py   # PF code for generated skills
    ├── pending_revisions/     # Skills needing revision
    └── library_history.json   # Acceptance/rejection history

The "active library" at any point = seed + generated.
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from .configs import LibraryConfig
from .skill_proposer import CandidateSkill
from .skill_reviewer import ReviewResult
from .skill_validator import ValidationResult

logger = logging.getLogger(__name__)

# Resolve self_improving/ module directory
_MODULE_DIR = Path(__file__).resolve().parent


class LibraryManager:
    """Manages the evolving skill library under self_improving/skills/."""

    def __init__(
        self,
        config: LibraryConfig,
        seed_skill_dir: str,
        generated_skill_dir: str,
        snapshots_dir: str,
    ):
        self.config = config
        # All paths relative to project root, but resolved to absolute
        self.seed_dir = Path(seed_skill_dir).resolve() if Path(seed_skill_dir).is_absolute() else (Path.cwd() / seed_skill_dir).resolve()
        self.generated_dir = Path(generated_skill_dir).resolve() if Path(generated_skill_dir).is_absolute() else (Path.cwd() / generated_skill_dir).resolve()
        self.snapshots_dir = Path(snapshots_dir).resolve() if Path(snapshots_dir).is_absolute() else (Path.cwd() / snapshots_dir).resolve()

        # Dynamic PF code file lives next to generated/
        self.dynamic_pf_file = self.generated_dir.parent / "dynamic_program_functions.py"

        # Revision / history also under self_improving/skills/
        self._history_file = self.generated_dir.parent / "library_history.json"
        self._revision_dir = self.generated_dir.parent / "pending_revisions"

        # Current skill IDs (seed + generated)
        self._seed_ids: Set[str] = set()
        self._generated_ids: Set[str] = set()

        # History
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, original_skill_source: str = "configs/skills/") -> None:
        """Initialize the library.

        Copies seed skills from *original_skill_source* into ``seed_dir``
        if seed_dir is empty. Creates generated/ and snapshots/ dirs.
        """
        # Ensure directories exist
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Copy seed skills if seed_dir is empty
        seed_children = [c for c in self.seed_dir.iterdir() if c.is_dir() and (c / "SKILL.md").exists()]
        if not seed_children:
            src = Path(original_skill_source)
            if not src.is_absolute():
                src = Path.cwd() / src
            if src.is_dir():
                logger.info("Copying seed skills from %s to %s", src, self.seed_dir)
                for child in sorted(src.iterdir()):
                    if child.is_dir() and (child / "SKILL.md").exists():
                        dst = self.seed_dir / child.name
                        if not dst.exists():
                            shutil.copytree(child, dst)
            else:
                logger.warning("Original skill source not found: %s", src)

        # Initialize dynamic PF file
        if not self.dynamic_pf_file.exists():
            self.dynamic_pf_file.write_text(
                '"""\nDynamically generated Program Functions from self-improving pipeline.\n"""\n\n'
                'import re\n'
                'import logging\n'
                'from typing import Dict, Any, Optional, List\n\n'
                '# Import base types from the main PF module\n'
                'from src.skills_agent.skills.program_functions import (\n'
                '    ProgramFunction, Intervention, InterventionType,\n'
                '    register_pf, PFRecord,\n'
                ')\n\n'
                'logger = logging.getLogger(__name__)\n\n'
                '# === Dynamically added PFs below ===\n\n',
                encoding="utf-8",
            )

        # Load history if exists
        if self._history_file.exists():
            try:
                self._history = json.loads(self._history_file.read_text())
            except Exception:
                self._history = []

        # Scan both directories
        self._scan_all()
        logger.info(
            "Library initialized: %d seed + %d generated = %d total skills",
            len(self._seed_ids), len(self._generated_ids), len(self.skill_ids),
        )

    def _scan_all(self) -> None:
        """Scan seed/ and generated/ for skill IDs."""
        self._seed_ids = self._scan_dir(self.seed_dir)
        self._generated_ids = self._scan_dir(self.generated_dir)

    @staticmethod
    def _scan_dir(directory: Path) -> Set[str]:
        ids = set()
        if not directory.exists():
            return ids
        for child in directory.iterdir():
            if child.is_dir() and (child / "SKILL.md").exists():
                ids.add(child.name)
        return ids

    @property
    def skill_ids(self) -> List[str]:
        """All active skill IDs (seed + generated), sorted. Includes all versions."""
        return sorted(self._seed_ids | self._generated_ids)

    @property
    def skill_ids_latest(self) -> List[str]:
        """Active skill IDs deduped by base — one (latest) version per base_id.

        Use this at runtime/inference: keep every version on disk for
        audit, but the live library should never expose two versions of
        the same base skill.
        """
        by_base: Dict[str, str] = {}
        for sid in self._seed_ids | self._generated_ids:
            base = self._strip_version_suffix(sid)
            cur = by_base.get(base)
            if cur is None:
                by_base[base] = sid
                continue
            # Keep whichever version is higher
            if self._parse_version(sid, base) > self._parse_version(cur, base):
                by_base[base] = sid
        return sorted(by_base.values())

    @property
    def seed_skill_ids(self) -> List[str]:
        return sorted(self._seed_ids)

    @property
    def generated_skill_ids(self) -> List[str]:
        return sorted(self._generated_ids)

    def get_library_dirs(self) -> List[str]:
        """Return list of directories that together form the active library.

        Useful for SkillLibrary loaders that need to scan multiple dirs.
        """
        dirs = [str(self.seed_dir)]
        if self._generated_ids:
            dirs.append(str(self.generated_dir))
        return dirs

    # ------------------------------------------------------------------
    # Accept / Reject skills
    # ------------------------------------------------------------------

    def process_candidates(
        self,
        candidates: List[CandidateSkill],
        reviews: List[ReviewResult],
        validations: List[ValidationResult],
        epoch: int = 0,
    ) -> Dict[str, str]:
        """Process reviewed candidates: accept, revise, or reject.

        Returns:
            Dict mapping skill_id -> decision ("accept"/"revise"/"reject")
        """
        decisions = {}

        review_map = {r.skill_id: r for r in reviews}
        val_map = {v.skill_id: v for v in validations}

        for candidate in candidates:
            review = review_map.get(candidate.skill_id)
            validation = val_map.get(candidate.skill_id)

            if not review or not validation:
                decisions[candidate.skill_id] = "reject"
                continue

            decision = self._decide(candidate, review, validation)
            # Treat "revise" as accept-with-version-bump. The versioning
            # logic in _accept_skill keeps every prior version on disk and
            # emits a new versioned skill_id; runtime selects only the
            # latest version per base_id (see skill_ids_latest).
            effective_decision = decision
            stored_skill_id = candidate.skill_id

            if decision in ("accept", "revise"):
                stored_skill_id = self._accept_skill(candidate, epoch)
                effective_decision = "accept"
                decisions[candidate.skill_id] = "accept"
            else:
                decisions[candidate.skill_id] = "reject"
                self._store_rejected(candidate, review, validation, epoch)

            self._history.append({
                "epoch": epoch,
                "skill_id": candidate.skill_id,
                "stored_skill_id": stored_skill_id,
                "decision": effective_decision,
                "review_decision": decision,
                "q_skill": review.q_skill if review else 0.0,
                "validation_passed": validation.passed if validation else False,
            })

        self._save_history()
        return decisions

    def _decide(
        self,
        candidate: CandidateSkill,
        review: ReviewResult,
        validation: ValidationResult,
    ) -> str:
        """Make accept/revise/reject decision."""
        if not validation.passed:
            return "reject"

        total = len(self._seed_ids) + len(self._generated_ids)
        if total >= self.config.max_library_size:
            logger.warning("Library at max size (%d), rejecting %s",
                           self.config.max_library_size, candidate.skill_id)
            return "reject"

        is_new_category = candidate.category not in self._get_existing_categories()
        threshold = (
            self.config.new_group_threshold if is_new_category
            else self.config.same_group_threshold
        )

        if not self.config.allow_new_categories and is_new_category:
            return "reject"

        if review.q_skill >= threshold:
            return "accept"
        elif review.q_skill >= threshold * 0.7:
            return "revise"
        else:
            return "reject"

    def _accept_skill(self, candidate: CandidateSkill, epoch: int) -> str:
        """Write accepted skill to generated/ directory.

        If another skill with the same base_id already exists in seed/ or
        generated/, the new skill is stored with a ``__v{N}`` suffix so
        every version is retained on disk. Returns the effective stored
        skill_id (may differ from candidate.skill_id).
        """
        base_id, stored_skill_id = self._resolve_versioned_id(candidate.skill_id)

        skill_dir = self.generated_dir / stored_skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Rewrite md_spec frontmatter to use the stored skill_id so the
        # loaded library records it under the versioned id.
        md_content = candidate.md_spec
        if not md_content:
            md_content = self._generate_default_md(candidate)
        if stored_skill_id != candidate.skill_id:
            md_content = self._rewrite_skill_id_in_md(md_content,
                                                     candidate.skill_id,
                                                     stored_skill_id)

        (skill_dir / "SKILL.md").write_text(md_content, encoding="utf-8")

        # Write metadata (record both base_id and stored_skill_id for later
        # dedup by base).
        version = self._parse_version(stored_skill_id, base_id)
        meta = {
            "skill_id": stored_skill_id,
            "base_skill_id": base_id,
            "version": version,
            "name": candidate.name,
            "category": candidate.category,
            "epoch_added": epoch,
            "source": "self_improving",
            "review_scores": candidate.review_scores,
        }
        (skill_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Append PF code (rewrite register_pf(...) to versioned id)
        if candidate.pf_code:
            self._append_pf_code(candidate, epoch, stored_skill_id)

        self._generated_ids.add(stored_skill_id)
        logger.info(
            "Accepted skill %s as %s (v%d) into generated/ (epoch %d), library = %d seed + %d generated",
            candidate.skill_id, stored_skill_id, version, epoch,
            len(self._seed_ids), len(self._generated_ids),
        )
        return stored_skill_id

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    _VERSION_RE = re.compile(r"^(?P<base>.+?)__v(?P<n>\d+)$")

    @classmethod
    def _strip_version_suffix(cls, skill_id: str) -> str:
        m = cls._VERSION_RE.match(skill_id)
        return m.group("base") if m else skill_id

    @classmethod
    def _parse_version(cls, skill_id: str, base_id: str) -> int:
        if skill_id == base_id:
            return 1
        m = cls._VERSION_RE.match(skill_id)
        return int(m.group("n")) if m else 1

    def _resolve_versioned_id(self, proposed_id: str) -> tuple:
        """Given a proposed skill_id, return (base_id, stored_id).

        If no collision, stored_id == proposed_id. On collision, stored_id
        becomes ``{base}__v{N+1}`` where N is the highest existing version.
        """
        base_id = self._strip_version_suffix(proposed_id)
        all_ids = self._seed_ids | self._generated_ids

        if proposed_id not in all_ids:
            return base_id, proposed_id

        # Find max version among ids sharing this base
        max_v = 1
        for sid in all_ids:
            if self._strip_version_suffix(sid) == base_id:
                max_v = max(max_v, self._parse_version(sid, base_id))
        return base_id, f"{base_id}__v{max_v + 1}"

    @staticmethod
    def _rewrite_skill_id_in_md(md: str, old_id: str, new_id: str) -> str:
        """Replace ``skill_id: old_id`` in YAML frontmatter with new_id."""
        pattern = re.compile(
            r"(^|\n)(\s*skill_id\s*:\s*)" + re.escape(old_id) + r"(\s*(?:\n|$))"
        )
        return pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{new_id}{m.group(3)}", md, count=1)

    def _append_pf_code(self, candidate: CandidateSkill, epoch: int,
                        stored_skill_id: str) -> None:
        """Append PF code to dynamic_program_functions.py.

        Rewrites ``@register_pf("old_id")`` to use ``stored_skill_id`` so
        the runtime registry keys by the versioned id.
        """
        code = candidate.pf_code.strip()
        # Normalize: ensure decorator exists and targets stored_skill_id
        code = re.sub(
            r'@register_pf\(\s*["\']' + re.escape(candidate.skill_id) + r'["\']\s*\)',
            f'@register_pf("{stored_skill_id}")',
            code,
        )
        if f'@register_pf("{stored_skill_id}")' not in code:
            code = f'@register_pf("{stored_skill_id}")\n{code}'

        # Rewrite skill_id="..." kwargs inside Intervention() etc. so
        # runtime PFRecord.skill_id matches the versioned id.
        code = re.sub(
            r'(skill_id\s*=\s*)["\']' + re.escape(candidate.skill_id) + r'["\']',
            r'\1"' + stored_skill_id + '"',
            code,
        )

        separator = f"\n\n# === Skill: {stored_skill_id} (epoch {epoch}) ===\n\n"
        with open(self.dynamic_pf_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write(code)
            f.write("\n")

    def _store_rejected(
        self,
        candidate: CandidateSkill,
        review: Optional[ReviewResult],
        validation: Optional["ValidationResult"],
        epoch: int,
    ) -> None:
        """Store rejected skills to trash/ for audit (never silently discarded)."""
        trash_dir = self.generated_dir.parent / "trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        path = trash_dir / f"{candidate.skill_id}_epoch{epoch}.json"
        path.write_text(json.dumps({
            "decision": "reject",
            "epoch": epoch,
            "candidate": candidate.to_dict(),
            "review": review.to_dict() if review else None,
            "validation": validation.to_dict() if validation else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Rejected skill %s stored in trash/ (epoch %d)", candidate.skill_id, epoch)

    def _generate_default_md(self, candidate: CandidateSkill) -> str:
        """Generate a default SKILL.md if student didn't provide one."""
        return f"""\
---
skill_id: {candidate.skill_id}
name: "{candidate.name}"
version: 1
priority: 0.7
error_category: "{candidate.category}"
system_summary: "{candidate.intervention_description[:100]}"
detection_triggers:
  - {candidate.target_failure_pattern[:50]}
avoidance_strategies:
  - {candidate.intervention_description[:100]}
applicable_modes: ["clean"]
phases:
  pre_final:
    conditions: ["always"]
    action: "verify_{candidate.skill_id}"
---
# {candidate.name}

{candidate.failure_description}

## Trigger
{chr(10).join('- ' + c for c in candidate.trigger_conditions[:5])}

## Intervention
{candidate.intervention_description}
"""

    # ------------------------------------------------------------------
    # Snapshots and history
    # ------------------------------------------------------------------

    def snapshot(self, epoch: int) -> Path:
        """Create a snapshot of the full library (seed + generated) for this epoch."""
        snapshot_dir = self.snapshots_dir / f"epoch_{epoch}"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy seed and generated into snapshot
        seed_snap = snapshot_dir / "seed"
        gen_snap = snapshot_dir / "generated"
        shutil.copytree(self.seed_dir, seed_snap)
        if self.generated_dir.exists() and any(self.generated_dir.iterdir()):
            shutil.copytree(self.generated_dir, gen_snap)
        else:
            gen_snap.mkdir()

        # Copy dynamic PF file
        if self.dynamic_pf_file.exists():
            shutil.copy2(self.dynamic_pf_file, snapshot_dir / "dynamic_program_functions.py")

        logger.info("Snapshot saved: epoch %d, %d seed + %d generated skills",
                     epoch, len(self._seed_ids), len(self._generated_ids))
        return snapshot_dir

    def _get_existing_categories(self) -> Set[str]:
        """Get categories from all skills (seed + generated)."""
        categories = set()

        # Scan metadata.json in generated skills
        for skill_id in self._generated_ids:
            meta_path = self.generated_dir / skill_id / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    categories.add(meta.get("category", ""))
                except Exception:
                    pass

        # For seed skills, infer from known groups
        for skill_id in self._seed_ids:
            for group, ids in {
                "exploration_control": ["insufficient_exploration", "retrieval_failure", "search_depth_controller"],
                "reasoning_guard": ["hallucination", "reasoning_error", "multi_hop_reasoning_failure"],
                "entity_verification": ["wrong_entity_confusion", "temporal_confusion", "numerical_reasoning_error"],
                "information_synthesis": ["claim_triangulation", "evidence_synthesis", "misinformation_detector"],
                "query_strategy": ["query_decomposition", "constraint_search", "iterative_refinement"],
                "format_output": ["format_extraction_error", "answer_completeness", "citation_mismatch"],
            }.items():
                if skill_id in ids:
                    categories.add(group)

        return categories

    def _save_history(self) -> None:
        self._history_file.write_text(
            json.dumps(self._history, indent=2), encoding="utf-8"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get current library state summary."""
        return {
            "total_skills": len(self.skill_ids),
            "seed_skills": len(self._seed_ids),
            "generated_skills": len(self._generated_ids),
            "skill_ids": self.skill_ids,
            "seed_dir": str(self.seed_dir),
            "generated_dir": str(self.generated_dir),
            "categories": sorted(self._get_existing_categories()),
            "history_entries": len(self._history),
        }
