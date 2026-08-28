"""The contract a generated PF must satisfy, and the structural gate applied
BEFORE any generated code is executed.

A PF is (anchor, evidence, action) — see `skills/executable/math/evidence_pfs.py::_EvidencePF`.
`PFSpec` is the serializable form of that triple plus the checker source, so a
proposal can be reviewed, screened and diffed as data before it becomes code.

The structural gate here is cheap and runs first. It encodes the rules that
every PF which actually produced rescues obeyed, and that every falsified PF
broke:

  * the checker may only read what the MODEL ITSELF WROTE (its reasoning, its
    self-written compute[] Observations, its committed answer, the spec's own
    examples). A checker that reads the gold answer scores perfectly offline
    and is useless at inference — this is the single most important rule.
  * evidence must be `deterministic` or `executed`. `helper` evidence cannot be
    screened offline against the correct-set control, so it cannot be forged
    here; hand-write it if you want it.
  * the checker must be pure text/math: no filesystem, network, or process.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Signatures the runtime calls. `_EvidencePF.deterministic_evidence` dispatches
# on which one is present.
STEP_CHECKER_ARGS = ("step_text", "full_response", "step_start")
ANSWER_CHECKER_ARGS = ("text", "arg", "ctx")

# Names that would leak supervision into an inference-time checker.
_GOLD_LEAK = re.compile(
    r"\b(gold|gold_answer|ground_truth|label|target_answer|expected_answer|is_correct)\b"
)
# Modules a text/math checker has no business touching.
_FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "pathlib", "importlib", "pickle", "multiprocessing", "ctypes", "threading",
}
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "vars"}


@dataclass
class PFSpec:
    """One proposed program function."""

    skill_id: str
    domain: str                      # math | code | web
    family_scope: str                # the one-sentence "only flag X" scope
    anchor: Dict[str, str]           # {level: step|final, trigger: str, evidence: deterministic|executed}
    checker_kind: str                # step | answer
    checker_src: str                 # python source defining `def check(...)`
    can_repair: bool = False
    rationale: str = ""              # why this family is checkable
    source_uids: List[str] = field(default_factory=list)  # wrong cases it came from
    # filled in by screen.py
    screen: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "PFSpec":
        return PFSpec(**{k: v for k, v in d.items() if k in PFSpec.__dataclass_fields__})


def validate_spec(spec: PFSpec, known_ids: Optional[set] = None) -> List[str]:
    """Return a list of violations; empty list means the spec may be screened.

    Purely structural — parses the checker, never runs it.
    """
    bad: List[str] = []

    if not re.fullmatch(r"[a-z][a-z0-9_]{4,48}", spec.skill_id or ""):
        bad.append(f"skill_id {spec.skill_id!r} is not snake_case (5..49 chars)")
    if known_ids and spec.skill_id in known_ids:
        bad.append(f"skill_id {spec.skill_id!r} collides with a registered PF")
    if spec.domain not in ("math", "code", "web"):
        bad.append(f"domain {spec.domain!r} not in math|code|web")
    if spec.checker_kind not in ("step", "answer"):
        bad.append(f"checker_kind {spec.checker_kind!r} not in step|answer")

    lvl, ev = spec.anchor.get("level"), spec.anchor.get("evidence")
    if lvl not in ("step", "final"):
        bad.append(f"anchor.level {lvl!r} not in step|final")
    if ev not in ("deterministic", "executed"):
        bad.append(
            f"anchor.evidence {ev!r} not screenable offline "
            "(only deterministic|executed can be forged; hand-write helper PFs)"
        )
    if not (spec.anchor.get("trigger") or "").strip():
        bad.append("anchor.trigger is empty — a PF must say what makes it look at a step")
    if len((spec.family_scope or "").split()) < 6:
        bad.append("family_scope must be a real 'only flag X' sentence")

    # ── the checker source ──
    try:
        tree = ast.parse(spec.checker_src or "")
    except SyntaxError as e:
        return bad + [f"checker_src does not parse: {e}"]

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    entry = next((n for n in fns if n.name == "check"), None)
    if entry is None:
        bad.append("checker_src must define `def check(...)` at module level")
    else:
        want = STEP_CHECKER_ARGS if spec.checker_kind == "step" else ANSWER_CHECKER_ARGS
        got = tuple(a.arg for a in entry.args.args)
        if got != want:
            bad.append(f"check() args {got} != required {want}")

    if _GOLD_LEAK.search(spec.checker_src or ""):
        bad.append(
            "checker reads gold/label — a checker may only read what the model "
            "itself wrote; it must work at inference time with no supervision"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    bad.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                bad.append(f"forbidden import from: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                bad.append(f"forbidden call: {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            bad.append(f"dunder attribute access: .{node.attr}")

    return bad


def _yaml_folded(text: str, indent: str = "  ") -> str:
    """A YAML folded block (`>`), which needs no escaping of quotes or colons."""
    body = " ".join((text or "").split())
    out, line = [], ""
    for w in body.split(" "):
        if len(line) + len(w) + 1 > 78:
            out.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out) or f"{indent}(none)"


def skill_md_frontmatter(spec: "PFSpec", *, priority: float = 0.6,
                         extra: Optional[Dict[str, str]] = None) -> str:
    """The YAML frontmatter a generated SKILL.md must carry.

    `MarkdownSkillLoader` REJECTS any SKILL.md that does not start with `---`
    — the PF code would still register through the chain-load, but the skill
    would be invisible to the pf_select menu, so the model could never select
    it. Only `skill_id` is strictly required; `system_summary` is the line the
    model actually reads when choosing a PF, so it is always written.
    """
    name = " ".join(w.capitalize() for w in spec.skill_id.split("_"))
    lines = [
        "---",
        f"skill_id: {spec.skill_id}",
        f"name: {name}",
        "version: 1",
        f"priority: {priority}",
        f"error_category: {(extra or {}).get('error_category', spec.skill_id)}",
        "applicable_modes: [all]",
        "applicable_phases: [think]",
        "system_summary: >",
        _yaml_folded(spec.family_scope),
        "anchor:",
        f"  level: {spec.anchor.get('level', 'step')}",
        f"  trigger: {json.dumps(str(spec.anchor.get('trigger', '')))}",
        f"  evidence: {json.dumps(str(spec.anchor.get('evidence', 'deterministic')))}",
        f"  action: {json.dumps('inject the verdict and redo the work from the anchored step')}",
    ]
    for k, v in (extra or {}).items():
        if k != "error_category":
            lines.append(f"{k}: {json.dumps(str(v))}")
    lines.append("---")
    return "\n".join(lines)


def registered_pf_ids(hasp_root) -> set:
    """Every PF id already registered anywhere in skills/ (collision check)."""
    ids = set()
    pat = re.compile(r'(?:register_pf|_family)\(\s*["\']([a-z0-9_]+)["\']')
    for p in (hasp_root / "skills").rglob("*.py"):
        try:
            ids.update(pat.findall(p.read_text()))
        except OSError:
            pass
    return ids
