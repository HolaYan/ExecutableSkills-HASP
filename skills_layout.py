"""Where the skill library lives, and how to load it.

`skills/` is split two ways — by what a skill *is*, then by domain:

    skills/
    ├── textual/<domain>/          prose-only PFs: they fire on a trigger and
    │   ├── <skill_id>/SKILL.md    inject a fixed reminder. The document IS the
    │   └── dynamic_program_functions.py                              skill.
    └── executable/
        ├── <domain>/evidence_pfs.py     the skill: a checker that re-verifies
        │                                a claim the model wrote
        └── docs/<domain>/<skill_id>/SKILL.md    its card — what the policy
                                                 reads when it picks a PF

Under `executable/` the code and the documents are kept apart on purpose: the
checker is the skill, and a tree of SKILL.md sitting next to it reads as though
the prose were.

Both halves of a domain are one library at runtime: `dynamic_program_functions.py`
chain-loads its sibling `evidence_pfs.py`, and where both register the same
`skill_id` the executable one wins (it is loaded last).

Callers should not join these paths by hand. `resolve()` takes whatever a config
or a CLI flag carries and returns the directories to scan and the modules to
exec-import, so a layout change lands here rather than in six call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Union

KINDS = ("textual", "executable")
DOMAINS = ("math", "web", "code")

DYNAMIC_PF = "dynamic_program_functions.py"
EXECUTABLE_PF = "skills.py"   # the single registration entry point
DOCS = "docs"          # executable/<DOCS>/<domain>/<skill_id>/SKILL.md


@dataclass
class Library:
    """What to load for one request."""
    skill_dirs: List[Path] = field(default_factory=list)   # each holds <skill_id>/SKILL.md
    pf_modules: List[Path] = field(default_factory=list)   # exec-import these, in order
    domains: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.skill_dirs or self.pf_modules)


def _is_flat_library(p: Path) -> bool:
    """A directory that is itself one library — the run-scoped copies that
    training writes, which have no textual/executable split."""
    if (p / DYNAMIC_PF).exists() or (p / EXECUTABLE_PF).exists():
        return True
    return any(c.is_dir() and (c / "SKILL.md").exists() for c in p.iterdir()) if p.is_dir() else False


def _domain_of(p: Path) -> str:
    return p.name if p.name in DOMAINS else ""


def resolve(spec: Union[str, Path], domains: Sequence[str] = ()) -> Library:
    """Resolve a library specification.

    Accepts, in order of preference:

    * the `skills/` root — every domain, both kinds;
    * `skills/<domain>` — one domain, both kinds. This form no longer exists on
      disk but is what training configs carry, so it is honoured;
    * `skills/<kind>/<domain>` — exactly that half;
    * any other directory that looks like a library — returned as-is, which is
      how the run-scoped copies under `{output_dir}/library/` keep working.

    `domains` narrows the root form; ignored for the others.
    """
    p = Path(spec)
    want = [d for d in (domains or DOMAINS) if d in DOMAINS]

    # skills/<kind>/<domain>
    if p.parent.name in KINDS and _domain_of(p):
        lib = Library(domains=[p.name])
        docs = p.parent / DOCS / p.name          # executable keeps its cards here
        if docs.is_dir():
            lib.skill_dirs.append(docs)
        elif p.is_dir():
            lib.skill_dirs.append(p)
        if p.is_dir():
            for f in (DYNAMIC_PF, EXECUTABLE_PF):
                if (p / f).exists():
                    lib.pf_modules.append(p / f)
        return lib

    # the root: <root>/{textual,executable}/<domain>
    if (p / "textual").is_dir() or (p / "executable").is_dir():
        lib = Library(domains=list(want))
        for dom in want:
            for kind in KINDS:                     # textual first: executable overrides
                d = p / kind / dom
                docs = p / kind / DOCS / dom       # executable/docs/<domain>
                if docs.is_dir():
                    lib.skill_dirs.append(docs)
                elif d.is_dir():
                    lib.skill_dirs.append(d)
                if not d.is_dir():
                    continue
                mod = d / (DYNAMIC_PF if kind == "textual" else EXECUTABLE_PF)
                if mod.exists():
                    lib.pf_modules.append(mod)     # skills.py loads programs.py itself
        return lib

    # skills/<domain> — the pre-split form still written in training configs
    if _domain_of(p) and (p.parent / "textual" / p.name).is_dir():
        return resolve(p.parent, domains=[p.name])

    # anything else that is already a library
    if p.is_dir() and _is_flat_library(p):
        lib = Library(skill_dirs=[p], domains=[_domain_of(p)] if _domain_of(p) else [])
        for f in (DYNAMIC_PF, EXECUTABLE_PF):
            if (p / f).exists():
                lib.pf_modules.append(p / f)
                break                              # DYNAMIC_PF chain-loads the other
        return lib

    return Library()
