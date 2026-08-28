"""Every `ctx.<field>` a skill reads must be a field `Ctx` actually declares.

This is the guard that makes attribute access safer than the dict it replaced.
`ctx.raed_count` raises at runtime — but a Detect that raises declines, and a
skill that always declines looks exactly like a skill that never matches. Three
skills were lost that way during the template migration, none with a traceback.

So the check runs by AST over the source, without importing or dispatching
anything: it needs no rollout to reach the typo, and it names the file and the
attribute rather than leaving a silent abstention to be noticed months later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills.pf_template import Ctx, ctx_fields_used  # noqa: E402

#: Not rollout state — the skill's own identity, the raw dict, and the two
#: escape hatches. Readable on `ctx`, but deliberately not in `FIELDS`.
CTX_API = {"raw", "skill_id", "anchor", "reasoning", "pf_helper", "get", "FIELDS"}

SKILL_FILES = sorted((_HASP / "skills" / "executable").glob("*/skills.py"))


def test_skill_files_are_where_the_test_thinks():
    assert len(SKILL_FILES) == 3, [str(p) for p in SKILL_FILES]


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_every_ctx_field_is_declared(path: Path):
    unknown = ctx_fields_used(path.read_text()) - set(Ctx.FIELDS) - CTX_API
    assert not unknown, (
        f"{path.relative_to(_HASP)} reads {sorted(unknown)} off ctx, which "
        f"Ctx.FIELDS does not declare. Either it is a typo, or the field is "
        f"real and belongs in Ctx.FIELDS with a default."
    )


def test_unknown_field_raises_rather_than_defaulting():
    ctx = Ctx("t", {"read_count": 3}, None)
    assert ctx.read_count == 3
    assert ctx.search_count == 0          # declared, absent -> its default
    with pytest.raises(AttributeError, match="raed_count"):
        ctx.raed_count                    # undeclared -> not a silent 0


def test_raw_stays_reachable_for_the_moved_implementations():
    raw = {"question": "q", "_private_cross_step": [1]}
    ctx = Ctx("t", raw, None)
    assert ctx.raw is raw                 # implementations.py takes step_context
    assert ctx.get("_private_cross_step") == [1]


def test_both_halves_take_ctx_action_arg():
    """The signature the contract is stated in, on every registered skill."""
    import inspect

    from pf_select.pf_select_eval import _load_pf_system
    _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore

    wrong = []
    for sid, pf in sorted(_PF_REGISTRY.items()):
        impl = getattr(pf, "pf_impl", None)
        if impl is None:                  # adapt_skill wraps a raw runtime PF
            continue
        for half in ("should_activate", "intervene"):
            names = [p.name for p in
                     inspect.signature(getattr(impl, half)).parameters.values()]
            if names != ["ctx", "action", "arg"]:
                wrong.append(f"{sid}.{half}{tuple(names)}")
    assert not wrong, wrong
