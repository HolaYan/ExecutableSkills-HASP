"""Behavioural parity harness for the PF library.

Rewriting a skill's Detect and Repair into template form is transcription, and
transcription is where this refactor has already introduced three silent
failures — a load cycle swallowed by `except: pass`, an `intervene` signature
mismatch turned into a NOOP, and a strip regex that deleted a class. None of
them raised; all three were found by counting things afterwards.

So: snapshot what every skill does across a battery of constructed rollouts
BEFORE touching it, and diff after. A migration is correct when the diff is
empty, or when every difference is one you can name and defend.

    python tests/pf_parity.py --snapshot .baseline.json   # before you change anything
    ... change a Detect, a Repair, a checker ...
    python tests/pf_parity.py --check    .baseline.json

No baseline ships with the repository. It records how the library behaves on
one machine with one set of installed packages, so it is only meaningful
against itself — take your own before you start.

What is compared, per (skill, case):
  * whether Detect fired;
  * the intervention type Repair returned (inject / modify / noop);
  * the injected text, normalised — the skill id tag and anchor tag are
    stripped, since those are exactly what the migration is meant to add.

CPU-only. Some code skills run the sandbox, so a full pass takes a couple of
minutes; that is cheap next to a silently broken library.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

# ── the battery ──────────────────────────────────────────────────────────
# Each case is (name, action_type, arg, ctx). They are constructed rather than
# sampled so the harness needs no data files and stays deterministic.

_MATH_WRONG_COMPUTE = (
    "Thought: I need C(4,2)*C(6,2).\n"
    "Action: compute[binom(4,2)*binom(6,2)]\n"
    "Observation: 36\n"
    "Thought: So the answer is 36.\nAction: finish[36]"
)
_MATH_STALLED = "Thought: I need the total.\nAction: compute[12*4]"
_MATH_CASES = ("Thought: Consider the case x < 0, and the case x >= 0. By WLOG assume "
               "a <= b. Let u = x + 1. Clearly the only solution is a = b = c.\n"
               "Action: finish[7]")
_MATH_GUESS = ("Thought: This is hard. It is a known result that the answer is 12, so I "
               "will go with that.\nAction: finish[12]")
_MATH_CLEAN = ("Thought: We verify by substituting back: 3*4 = 12, which checks out.\n"
               "Action: finish[12]")

_SPEC_DOCTEST = ('def add(a, b):\n    """Add two numbers.\n\n    >>> add(1, 2)\n    3\n    """\n')
_SPEC_PLAIN = 'def add(a, b):\n    """Add two numbers."""\n'
_SPEC_RAISE = ('def div(a, b):\n    """Divide. Raises ValueError for b == 0.\n\n'
               '    >>> div(4, 2)\n    2.0\n    """\n')

_WEB_READ = ("Bong Joon-ho directed Parasite, released in 2019. " * 8)

# The runtime appends dicts, not action names — see agent_runner.py:1167. A list
# of bare strings makes every skill that reads the history raise AttributeError,
# which the dispatch turns into a permanent quiet abstention.
def _SEARCHED(q): return {"action_type": "SEARCH", "arg": q}
def _READ(u): return {"action_type": "READ", "arg": u}

CASES = [
    ("math/wrong-compute", "FINAL", "36",
     {"question": "How many ways are there to choose 2 of 4 and 2 of 6?",
      "raw_reasoning": _MATH_WRONG_COMPUTE, "domain": "math"}),
    ("math/stalled", "FINAL", "",
     {"question": "What is 12 times 4?", "raw_reasoning": _MATH_STALLED, "domain": "math"}),
    ("math/cases-and-substitution", "FINAL", "7",
     {"question": "Find the positive integer n.", "raw_reasoning": _MATH_CASES,
      "domain": "math"}),
    ("math/guess", "FINAL", "12",
     {"question": "AIME 2024: find m+n.", "raw_reasoning": _MATH_GUESS, "domain": "math"}),
    ("math/clean", "FINAL", "12",
     {"question": "What is the probability?", "raw_reasoning": _MATH_CLEAN, "domain": "math"}),
    ("math/out-of-range", "FINAL", "1200",
     {"question": "AIME: the answer is an integer between 0 and 999.",
      "raw_reasoning": _MATH_CLEAN, "domain": "math"}),

    ("code/doctest-failing", "FINAL", "    return a - b",
     {"question": _SPEC_DOCTEST, "entry_point": "add", "public_test_code": "",
      "domain": "code", "raw_reasoning": ""}),
    ("code/doctest-passing", "FINAL", "    return a + b",
     {"question": _SPEC_DOCTEST, "entry_point": "add", "public_test_code": "",
      "domain": "code", "raw_reasoning": ""}),
    ("code/no-examples", "FINAL", "    return a + b",
     {"question": _SPEC_PLAIN, "entry_point": "add", "public_test_code": "",
      "domain": "code", "raw_reasoning": ""}),
    ("code/raise-contract", "FINAL", "def div(a, b):\n    return a / b",
     {"question": _SPEC_RAISE, "entry_point": "div", "public_test_code": "",
      "domain": "code", "raw_reasoning": ""}),
    ("code/split-and-import", "FINAL",
     "def words(s):\n    return np.array(s.split(' '))",
     {"question": "Return the whitespace-separated words as a list.",
      "entry_point": "words", "public_test_code": "", "domain": "code",
      "raw_reasoning": ""}),

    ("web/final-grounded", "FINAL", "Bong Joon-ho",
     {"question": "Who directed Parasite?", "all_read_contents": _WEB_READ,
      "last_search_results_text": "Parasite (2019) — Bong Joon-ho", "thought": "",
      "step_count": 4, "max_steps": 10, "search_count": 2, "read_count": 1,
      "action_history": [_SEARCHED("parasite director"), _READ("https://example.org/parasite")], "domain": "web"}),
    ("web/final-ungrounded", "FINAL", "Christopher Nolan in 1998",
     {"question": "Who directed Parasite and when?", "all_read_contents": _WEB_READ,
      "last_search_results_text": "Parasite (2019) — Bong Joon-ho", "thought": "",
      "step_count": 4, "max_steps": 10, "search_count": 2, "read_count": 1,
      "action_history": [_SEARCHED("parasite director"), _READ("https://example.org/parasite")], "domain": "web"}),
    ("web/final-no-evidence", "FINAL", "Bong Joon-ho",
     {"question": "Who directed Parasite?", "all_read_contents": "", "thought": "",
      "last_search_results_text": "", "step_count": 1, "max_steps": 10,
      "search_count": 0, "read_count": 0, "action_history": [], "domain": "web"}),
    ("web/search-long-query", "SEARCH",
     "who was the director of the south korean film parasite which won best picture in 2020",
     {"question": "Who directed Parasite?", "all_read_contents": "",
      "last_search_results_text": "some results", "thought": "", "step_count": 2,
      "max_steps": 10, "search_count": 1, "read_count": 0,
      "action_history": [_SEARCHED("parasite director")], "domain": "web"}),
    ("web/read-dense", "READ", "https://example.org/parasite",
     {"question": "Who directed Parasite?", "all_read_contents": _WEB_READ,
      "last_search_results_text": "r", "thought": "", "step_count": 3, "max_steps": 10,
      "search_count": 1, "read_count": 1, "action_history": [_SEARCHED("parasite director"), _READ("https://example.org/parasite")],
      "domain": "web"}),
]

_TAG = re.compile(r"^\[[a-z0-9_]+(?:\s+@[^\]]*)?\]\s*")


def _norm_text(t: str) -> str:
    """Drop the skill-id/anchor tag — adding it is the point of the migration."""
    return " ".join(_TAG.sub("", (t or "").strip()).split())[:300]


def _probe():
    from pf_select.pf_select_eval import _load_pf_system
    _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore

    out = {}
    for sid in sorted(_PF_REGISTRY):
        pf = _PF_REGISTRY[sid]
        per = {}
        for name, at, arg, ctx in CASES:
            rec = {"fired": False, "type": None, "text": ""}
            try:
                fired = bool(pf.should_activate(dict(ctx), at, arg))
                rec["fired"] = fired
                if fired:
                    iv = pf.intervene(dict(ctx), at, arg)
                    rec["type"] = iv.type.value if iv is not None else None
                    rec["text"] = _norm_text(getattr(iv, "context_text", "")
                                             or getattr(iv, "new_action_arg", "") or "")
            except Exception as e:
                rec["type"] = f"RAISED:{type(e).__name__}"
            per[name] = rec
        out[sid] = per
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", metavar="PATH")
    ap.add_argument("--check", metavar="PATH")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not (a.snapshot or a.check):
        ap.error("give --snapshot PATH or --check PATH")

    cur = _probe()
    if a.snapshot:
        p = Path(a.snapshot)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur, indent=1, sort_keys=True))
        n_fire = sum(1 for s in cur.values() for c in s.values() if c["fired"])
        print(f"[snapshot] {len(cur)} skills x {len(CASES)} cases, {n_fire} fires -> {p}")
        return

    old = json.loads(Path(a.check).read_text())
    added = sorted(set(cur) - set(old))
    removed = sorted(set(old) - set(cur))
    diffs = []
    for sid in sorted(set(cur) & set(old)):
        for case in sorted(set(cur[sid]) | set(old[sid])):
            o, n = old[sid].get(case, {}), cur[sid].get(case, {})
            for key in ("fired", "type", "text"):
                if o.get(key) != n.get(key):
                    diffs.append((sid, case, key, o.get(key), n.get(key)))

    if removed:
        print(f"❌ skills gone: {removed}")
    if added:
        print(f"⚠️  skills new: {added}")
    if not diffs:
        print(f"✅ behavioural parity: {len(cur)} skills identical across {len(CASES)} cases")
    else:
        by_skill = {}
        for sid, case, key, o, n in diffs:
            by_skill.setdefault(sid, []).append((case, key, o, n))
        print(f"❌ {len(diffs)} differences across {len(by_skill)} skills")
        for sid, rows in sorted(by_skill.items()):
            print(f"\n  {sid}")
            for case, key, o, n in rows[:6] if a.quiet else rows:
                print(f"    {case} · {key}\n      before: {o!r}\n      after:  {n!r}")
    sys.exit(1 if (diffs or removed) else 0)


if __name__ == "__main__":
    main()
