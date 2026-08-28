"""A helper-backed PF must actually receive the helper.

This channel fails silently by construction. Every helper-backed PF has a
deterministic fallback, so when the helper never arrives the PF still fires,
still returns an intervention, and still looks healthy — it just quietly stops
using the evidence it was written around. Nothing raises, no count changes, and
the only symptom is a smaller effect.

`call_base_intervene` decides whether to pass the helper by *inspecting the
parameter name*, which makes the channel sensitive to a rename that no other
test would notice. So: assert the helper arrives, under both spellings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills.pf_template import (  # noqa: E402
    HELPER_PARAM_NAMES, call_base_intervene,
)


class _Spy:
    """Stands in for the PF helper; records that it was consulted."""

    def __init__(self):
        self.calls = 0

    def generate(self, **kw):
        self.calls += 1
        return "parasite director bong joon-ho"

    def locate(self, **kw):
        self.calls += 1
        return "OK"


@pytest.mark.parametrize("param", HELPER_PARAM_NAMES)
def test_helper_reaches_a_pf_that_declares_it(param):
    ns = {}
    exec(f"def intervene(self, ctx, action_type, arg, {param}=None):\n"
         f"    return {param}\n", ns)

    class Base:
        pass
    Base.intervene = ns["intervene"]

    spy = _Spy()
    assert call_base_intervene(Base, Base(), {}, "SEARCH", "q", spy) is spy


def test_a_pf_without_the_parameter_is_not_handed_one():
    class Base:
        def intervene(self, ctx, action_type, arg):
            return "four-arg"

    assert call_base_intervene(Base, Base(), {}, "SEARCH", "q", _Spy()) == "four-arg"


def _registry():
    from pf_select.pf_select_eval import _load_pf_system
    _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore
    return _PF_REGISTRY


def test_some_skill_still_declares_it_needs_a_helper():
    declared = [sid for sid, pf in _registry().items()
                if getattr(pf, "needs_helper", False)]
    assert declared, (
        "no skill declares needs_helper. Either the attribute was renamed on "
        "one side of the dispatch only, or the helper-backed skills are gone."
    )


def test_the_wrapper_puts_the_helper_where_repair_looks_for_it():
    """The two halves of the channel, checked separately.

    A skill's Repair reads the helper off `ctx`, while the dispatch hands it in
    as an argument; `_PF.intervene` is the joint. Probing whole skills tests
    this only by luck — most Repair bodies reach a deterministic verdict or a
    continuation and return before they ever ask for a helper. So check the
    joint itself: the wrapper stores it, and `helper_verdict` finds it.
    """
    from skills.pf_template import Anchor, Ctx, helper_verdict

    pf = next(p for p in _registry().values() if getattr(p, "needs_helper", False))
    step_context, spy = {"question": "q", "raw_reasoning": "r"}, _Spy()
    try:
        pf.intervene(step_context, "FINAL", "a", spy)
    except Exception:
        pass                                  # the verdict is not what is under test
    assert step_context.get("_pf_helper") is spy, (
        "the dispatch did not store the helper on the step context, so every "
        "Repair body that asks for one will find None and fall back silently"
    )

    spy2 = _Spy()
    ctx = Ctx("t", {"_pf_helper": spy2, "question": "q", "raw_reasoning": "r"},
              Anchor(level="final"))
    assert ctx.pf_helper is spy2
    helper_verdict(ctx, "arithmetic")
    assert spy2.calls == 1, "helper_verdict did not consult the helper it was given"
