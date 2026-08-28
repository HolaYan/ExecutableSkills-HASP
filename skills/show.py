"""Browse and demonstrate one PF skill by name.

This repository exists to show what the skills are, so a skill has to be
inspectable on its own — not only reachable through a model that happened to
select it. Naming one on the command line shows its card, its anchor, the two
modules it is made of, and, given a rollout, exactly what it does with it.

    python -m skills.show --list                       every skill, one line each
    python -m skills.show --list --domain math         one domain
    python -m skills.show compute_observation_verify   the skill itself
    python -m skills.show compute_observation_verify --demo
    python -m skills.show arithmetic_slip --on rollout.txt

`--demo` runs the skill against a built-in rollout chosen for its domain, so
the output shows Detect firing and Repair speaking without needing a GPU or a
dataset. `--on FILE` does the same with your own rollout text.

To force a skill through the real evaluation rather than inspect it:

    python pf_select/eval_models.py --skills compute_observation_verify ...
"""
from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

# One rollout per domain, wrong on purpose, so a demo has something to find.
DEMO = {
    "math": dict(
        question="How many ways are there to choose 2 of 4 and 2 of 6?",
        arg="36",
        response=("Thought: I need C(4,2)*C(6,2).\n"
                  "Action: compute[binom(4,2)*binom(6,2)]\n"
                  "Observation: 36\n"
                  "Thought: The only arrangement is a = b = c, so the answer is 36.\n"
                  "Action: finish[36]")),
    "code": dict(
        question=('def add(a, b):\n    """Add two numbers.\n\n'
                  '    >>> add(1, 2)\n    3\n    """\n'),
        arg="    return a - b",
        response="", entry_point="add", public_test_code=""),
    "web": dict(
        question="Who directed Parasite?",
        arg="Christopher Nolan",
        response="",
        all_read_contents="Bong Joon-ho directed Parasite, released in 2019. " * 6,
        last_search_results_text="Parasite (2019) — directed by Bong Joon-ho"),
}


def _load():
    from pf_select.pf_select_eval import _load_pf_system
    _, library = _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore
    return dict(_PF_REGISTRY), library


def _wrap(text: str, indent: str = "    ", width: int = 76) -> str:
    return "\n".join(textwrap.fill(" ".join(str(text).split()), width=width,
                                   initial_indent=indent, subsequent_indent=indent)
                     .splitlines())


def cmd_list(reg: Dict, library: Dict, domain: Optional[str]) -> None:
    rows = []
    for sid in sorted(reg):
        pf = reg[sid]
        dom = getattr(type(pf), "pf_domain", "?")
        if domain and dom != domain:
            continue
        a = getattr(type(pf), "pf_anchor", None)
        card = library.get(sid)
        summ = (getattr(type(pf), "pf_summary", "")
                or getattr(card, "system_summary", "") or "")
        rows.append((dom, sid, f"{a.level}/{a.evidence}" if a else "—",
                     " ".join(summ.split())[:64]))
    for dom, sid, anch, summ in sorted(rows):
        print(f"  [{dom:<4}] {sid:<34} {anch:<20} {summ}")
    print(f"\n  {len(rows)} skill(s)" + (f" in {domain}" if domain else ""))


def cmd_show(reg: Dict, library: Dict, sid: str, demo: bool, on: Optional[str]) -> int:
    pf = reg.get(sid)
    if pf is None:
        near = [k for k in reg if sid in k or k in sid][:5]
        print(f"no skill named {sid!r}" + (f"; did you mean {near}?" if near else ""))
        return 1
    cls = type(pf)
    a = getattr(cls, "pf_anchor", None)
    dom = getattr(cls, "pf_domain", "?")

    print(f"\n{'=' * 78}\n  {sid}   [{dom}]\n{'=' * 78}")
    print("\nWHAT IT CHECKS")
    print(_wrap(getattr(cls, "pf_summary", "") or "(no summary)"))

    if a is not None:
        print("\nANCHOR")
        print(f"    level     {a.level}     — "
              + ("one reasoning step, reported as @step k/N" if a.level == "step"
                 else "the committed answer"))
        print(f"    evidence  {a.evidence}")
        print("    trigger")
        print(_wrap(a.trigger or "(none)", indent="      "))

    impl = getattr(cls, "pf_impl", None)
    if impl is not None:
        print("\nDETECT — should_activate(ctx, action, arg)")
        try:
            print(textwrap.indent(textwrap.dedent(
                inspect.getsource(type(impl).should_activate)).rstrip(), "    "))
        except (OSError, TypeError):
            print("    (source unavailable)")
        print("\nREPAIR — intervene(ctx, action, arg)")
        try:
            print(textwrap.indent(textwrap.dedent(
                inspect.getsource(type(impl).intervene)).rstrip(), "    "))
        except (OSError, TypeError):
            print("    (source unavailable)")

    card = library.get(sid)
    if card is not None and getattr(card, "description", ""):
        print("\nCARD")
        print(_wrap(card.description))

    if demo or on:
        _run_demo(pf, sid, dom, on)
    else:
        print("\n  (add --demo to run it on a sample rollout)")
    print()
    return 0


def _run_demo(pf, sid: str, dom: str, on: Optional[str]) -> None:
    case = dict(DEMO.get(dom, DEMO["math"]))
    if on:
        case["response"] = Path(on).read_text()
    ctx = {"question": case["question"], "domain": dom,
           "raw_reasoning": case.get("response", ""),
           "thought": case.get("response", ""),
           "all_read_contents": case.get("all_read_contents", ""),
           "last_search_results_text": case.get("last_search_results_text", ""),
           "entry_point": case.get("entry_point", ""),
           "public_test_code": case.get("public_test_code", ""),
           "step_count": 3, "max_steps": 10, "search_count": 1, "read_count": 1,
           "action_history": [], "_pf_fire_counts": {}}
    arg = case.get("arg", "")

    print(f"\n{'-' * 78}\nDEMO — a rollout this skill is meant to catch\n{'-' * 78}")
    body = case.get("response") or case["question"]
    print(textwrap.indent(body.strip()[:600], "    "))
    print(f"\n    committed answer: {arg!r}")

    fired = False
    try:
        fired = bool(pf.should_activate(dict(ctx), "FINAL", arg))
    except Exception as e:
        print(f"\n  Detect raised {type(e).__name__}: {e}")
        return
    print(f"\n  Detect  -> {'FIRED' if fired else 'silent'}")
    if not fired:
        print("            this rollout does not match the failure pattern, so the "
              "skill\n            never runs — which is the common case and the point.")
        return
    try:
        iv = pf.intervene(dict(ctx), "FINAL", arg)
    except Exception as e:
        print(f"  Repair raised {type(e).__name__}: {e}")
        return
    kind = iv.type.value if iv is not None else "none"
    print(f"  Repair  -> {kind}")
    text = (getattr(iv, "context_text", "") or getattr(iv, "new_action_arg", "") or "")
    if text:
        print()
        print(_wrap(text, indent="            "))
    elif kind == "noop":
        print("            nothing concrete to say, so it abstains and the answer stands.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="skills.show",
                                 description="inspect and demonstrate one PF skill")
    ap.add_argument("skill", nargs="?", help="skill id")
    ap.add_argument("--list", action="store_true", help="list every skill")
    ap.add_argument("--domain", choices=["math", "web", "code"])
    ap.add_argument("--demo", action="store_true", help="run it on a built-in rollout")
    ap.add_argument("--on", metavar="FILE", help="run it on a rollout of your own")
    a = ap.parse_args()

    reg, library = _load()
    if a.list or not a.skill:
        cmd_list(reg, library, a.domain)
        if not a.skill:
            return
    sys.exit(cmd_show(reg, library, a.skill, a.demo, a.on))


if __name__ == "__main__":
    main()
