"""The run-scoped, growing library.

Training never writes into `skills/`. At generation 0 the domain library is
copied to `{output_dir}/library/`, the training config's
`rollout.skill_library_dir` is pointed at the copy, and every later generation
appends there. Three reasons:

  * the hand-written library is what every measured number in this workspace
    was produced with — a training run must not silently change it;
  * each experiment's evolved library is self-contained, diffable, and can be
    replayed or rolled back to any generation;
  * provenance is unambiguous: each PF block records the generation and the
    training step that produced it, so a result can be attributed to the
    library that was actually live at the time.

When new PFs take effect: `SkillRolloutRunner.setup()` loads the library ONCE.
RS builds a fresh Rollouter per iteration, so admissions take effect at the
next rollout phase; online-rollout training (GRPO) picks them up when its
runner is next set up. `evolved_pfs.py` is chain-loaded from the copy's
`dynamic_program_functions.py`, the same mechanism `evidence_pfs.py` uses.
"""
from __future__ import annotations

import json
import shutil
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

from skills_construct.forge.spec import PFSpec, skill_md_frontmatter

_HASP = Path(__file__).resolve().parents[1]

_HEADER = '''"""PFs distilled during training by `evolving/` — DO NOT EDIT BY HAND.

Each block records the generation and training step that produced it. Every PF
here passed the structural gate and the offline precision screen against the
correct-set control of the same in-training eval; NONE has passed the
end-to-end probe or the n=64 accuracy test. Treat them as provisional until
`forge` has measured them — see evolving/README.md.
"""
from __future__ import annotations

import importlib.util as _iu
import re  # noqa: F401
from math import gcd  # noqa: F401
from pathlib import Path as _P
from typing import Optional  # noqa: F401


def _load_evidence_module():
    p = _P(__file__).resolve().parent / "evidence_pfs.py"
    spec = _iu.spec_from_file_location("_hasp_evidence_for_evolved", str(p))
    mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EV = _load_evidence_module()
_family = _EV._family
'''

_CHAIN_LOAD = '''

# ── evolving: PFs distilled during training (appended by evolving/library.py) ──
def _load_evolved_pfs():
    import importlib.util as _iu
    from pathlib import Path as _P
    p = _P(__file__).resolve().parent / "evolved_pfs.py"
    if p.exists():
        spec = _iu.spec_from_file_location("_hasp_evolved_pfs", str(p))
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)


_load_evolved_pfs()
'''


class EvolvingLibrary:
    """A copy of one domain library that grows across generations."""

    def __init__(self, root: Path, domain: str):
        self.root = Path(root)
        self.domain = domain
        self.ledger_path = self.root / "evolution.jsonl"

    # ── setup ──
    @classmethod
    def create(cls, root: Path, domain: str) -> "EvolvingLibrary":
        root = Path(root)
        lib = cls(root, domain)
        if not root.exists():
            # The shipped library is split into skills/{textual,executable}/<domain>;
            # a run-scoped copy is flat, so both halves are merged into one dir.
            # The shipped library is three trees: the textual half (cards and
            # code together), the executable code, and the executable cards.
            # A run-scoped copy is flat, so all three are merged into one dir.
            srcs = [_HASP / "skills" / "textual" / domain,
                    _HASP / "skills" / "executable" / domain,
                    _HASP / "skills" / "executable" / "docs" / domain]
            present = [s for s in srcs if s.is_dir()]
            if not present:
                raise SystemExit(f"no such domain library to seed from: {srcs[0]}")
            root.parent.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(present):
                shutil.copytree(src, root, dirs_exist_ok=bool(i),
                                ignore=shutil.ignore_patterns("__pycache__"))
            src = srcs[0]
            (root / "evolved_pfs.py").write_text(_HEADER)
            dpf = root / "dynamic_program_functions.py"
            txt = dpf.read_text()
            if "_load_evolved_pfs" not in txt:
                dpf.write_text(txt + textwrap.dedent(_CHAIN_LOAD))
            lib._log(dict(event="seeded", generation=0, source=str(src)))
        return lib

    # ── growth ──
    def admit(self, specs: List[PFSpec], generation: int, step: int,
              pass1: Optional[float] = None, review: Optional[Dict] = None) -> int:
        """Append accepted PFs and record the generation. Returns how many."""
        if not specs:
            self._log(dict(event="generation", generation=generation, step=step,
                           pass1=pass1, admitted=[], n=0, review=review or {}))
            return 0

        f = self.root / "evolved_pfs.py"
        body = f.read_text() if f.exists() else _HEADER
        for s in specs:
            body += self._render(s, generation, step)
        f.write_text(body)

        for s in specs:
            d = self.root / s.skill_id
            d.mkdir(exist_ok=True)
            (d / "SKILL.md").write_text(self._skill_md(s, generation, step))

        self._log(dict(event="generation", generation=generation, step=step, pass1=pass1,
                       admitted=[dict(skill_id=s.skill_id,
                                      lift=(s.screen or {}).get("lift"),
                                      fire_wrong=(s.screen or {}).get("fire_wrong"),
                                      fire_correct=(s.screen or {}).get("fire_correct"))
                                 for s in specs],
                       n=len(specs), review=review or {}))
        return len(specs)

    def _render(self, s: PFSpec, generation: int, step: int) -> str:
        src = s.checker_src.strip().replace("def check(", f"def _check_{s.skill_id}(", 1)
        kind = "step_checker" if s.checker_kind == "step" else "answer_checker"
        anchor = json.dumps({"level": s.anchor.get("level", "step"),
                             "trigger": s.anchor.get("trigger", ""),
                             "evidence": s.anchor.get("evidence", "deterministic")})
        sc = s.screen or {}
        return (f"\n\n# ── {s.skill_id}  [generation {generation}, step {step}] "
                f"screened wrong {sc.get('fire_wrong', 0):.1%} / "
                f"correct {sc.get('fire_correct', 0):.1%}, lift {sc.get('lift', 0)} ──\n"
                f"{src}\n\n"
                f'_family("{s.skill_id}",\n'
                f'        "{s.family_scope.replace(chr(34), chr(39))}",\n'
                f"        {kind}=_check_{s.skill_id},\n"
                f"        can_repair={bool(s.can_repair)},\n"
                f"        anchor={anchor})\n")

    def _skill_md(self, s: PFSpec, generation: int, step: int) -> str:
        sc = s.screen or {}
        samples = "\n".join(f"- `{x['label']}` {x['verdict'][:150]}"
                            for x in sc.get("samples", [])[:3]) or "- (none recorded)"
        # frontmatter is mandatory: MarkdownSkillLoader rejects a SKILL.md
        # without it, which would leave the PF registered but invisible to the
        # pf_select menu
        fm = skill_md_frontmatter(s, priority=0.55,
                                  extra={"provenance": f"evolving gen {generation} step {step}"})
        return (fm + "\n\n"
                f"# {s.skill_id}\n\n"
                f"Distilled during training at **generation {generation}, step {step}**, "
                f"from failures of the checkpoint at that step.\n\n"
                f"## Anchor\n- **level**: {s.anchor.get('level','')}\n"
                f"- **trigger**: {s.anchor.get('trigger','')}\n"
                f"- **evidence**: {s.anchor.get('evidence','')}\n\n"
                f"## Scope\n{s.family_scope}\n\n"
                f"## Why this is checkable\n{s.rationale or '(not stated)'}\n\n"
                f"## In-training screening\n"
                f"| | fires on |\n|---|---|\n"
                f"| failures of this checkpoint | {sc.get('fire_wrong', 0):.1%} |\n"
                f"| its successes (control) | {sc.get('fire_correct', 0):.1%} |\n"
                f"| lift | {sc.get('lift', 0)} |\n\n"
                f"{samples}\n\n"
                f"## Status\n**Provisional.** Screened only — no end-to-end probe, no "
                f"measured accuracy. Run it through `forge` before treating it as "
                f"established.\n")

    # ── bookkeeping ──
    def _log(self, rec: Dict) -> None:
        rec["at"] = time.strftime("%Y-%m-%d %H:%M")
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def history(self) -> List[Dict]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(l) for l in self.ledger_path.open() if l.strip()]

    def admitted_ids(self) -> List[str]:
        return [a["skill_id"] for r in self.history()
                for a in r.get("admitted", [])]

    def falsified_note(self) -> str:
        """What this run already admitted — so later cycles do not re-propose it."""
        ids = self.admitted_ids()
        if not ids:
            return ""
        return "\n".join(f"  - {i} (already in this run's library)" for i in ids)

    def summary(self) -> str:
        h = [r for r in self.history() if r.get("event") == "generation"]
        n = sum(r.get("n", 0) for r in h)
        curve = " → ".join(f"{r['pass1']:.3f}" for r in h if r.get("pass1") is not None)
        return (f"library[{self.root.name}] {len(h)} generations, {n} PFs admitted"
                + (f"; pass@1 {curve}" if curve else ""))
