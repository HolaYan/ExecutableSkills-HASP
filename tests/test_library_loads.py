"""Every skill module must import, and the library must be the size it claims.

`_load_pf_system` catches whatever a skill module raises so one bad file cannot
take down a rollout. The cost is that a module which fails to import leaves the
*previous* registration holding the skill id: the library still answers, still
reports skills, and quietly serves the pre-template implementations. That is
not a hypothetical — a one-line scoping bug in `pf_skill` took out all three
skill modules while 66 tests stayed green, because nothing asserted that the
modules import at all.

So import each one directly, outside the swallowing loader, and let the
traceback through.
"""
from __future__ import annotations

import importlib.util as _iu
import sys
from pathlib import Path

import pytest

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

SKILL_MODULES = sorted((_HASP / "skills" / "executable").glob("*/skills.py"))

#: The library ships this many registered skills. A drop means a module failed
#: to load; a rise means one was added without updating this number.
EXPECTED_SKILLS = 72


@pytest.mark.parametrize("path", SKILL_MODULES, ids=lambda p: p.parent.name)
def test_skill_module_imports(path: Path):
    """No try/except: if the module raises, the test shows exactly where."""
    spec = _iu.spec_from_file_location(f"_probe_{path.parent.name}", str(path))
    spec.loader.exec_module(_iu.module_from_spec(spec))


def test_library_registers_every_skill():
    from pf_select.pf_select_eval import _load_pf_system
    cards = _load_pf_system("skills")[1]
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore

    assert len(_PF_REGISTRY) == EXPECTED_SKILLS, (
        f"{len(_PF_REGISTRY)} PFs registered, expected {EXPECTED_SKILLS}. A "
        f"module that fails to import leaves its ids registered to whatever "
        f"claimed them earlier, so the count is the symptom, not the cause — "
        f"run the import test above for the traceback."
    )
    assert len(cards) == EXPECTED_SKILLS, f"{len(cards)} cards, expected {EXPECTED_SKILLS}"


def test_every_skill_came_from_the_template():
    """A template skill carries `pf_impl`; a raw survivor does not.

    This is what separates "the library loaded" from "the library loaded the
    files we think it did".
    """
    from pf_select.pf_select_eval import _load_pf_system
    _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore

    raw = sorted(sid for sid, pf in _PF_REGISTRY.items()
                 if not hasattr(pf, "pf_impl") and not hasattr(pf, "pf_adapted_from"))
    assert not raw, (
        f"{len(raw)} skills are neither template-declared nor adapted: {raw}. "
        f"They are pre-template implementations still holding their ids."
    )
