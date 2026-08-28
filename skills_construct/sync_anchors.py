"""Keep every SKILL.md's `anchor:` block in step with the code.

The anchor is declared once, in the skill's `@pf_skill(...)` head, and that
declaration is what the runtime uses. The card is what the *policy* reads when
it picks a PF, and what a person reads first — so an anchor that lives only in
code is invisible to both.

    python -m skills_construct.sync_anchors --check    # CI / tests: drift is an error
    python -m skills_construct.sync_anchors --write    # write the code's anchors into the cards

`--check` is the one to wire into tests. It fails on three things: a card whose
anchor block disagrees with the code, a registered skill with no card, and a
card for a skill nothing registers.

Only the `anchor:` block is touched. Everything else in the frontmatter and the
whole body are left exactly as they are, so a hand-written card stays
hand-written.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

_ANCHOR_BLOCK = re.compile(r"^anchor:\s*\n(?:[ \t]+\S.*\n)+", re.M)
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
# `anchor:` goes after system_summary and before phases, which is where the
# hand-written cards already put it.
_AFTER = re.compile(r"^system_summary:\s*>\s*\n(?:[ \t]+\S.*\n)+", re.M)
_BEFORE = re.compile(r"^phases:\s*$", re.M)


def _yaml_str(v: str) -> str:
    """Quote for YAML — the trigger text routinely contains `:`, `[`, quotes."""
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_anchor(anchor, action: str = "") -> str:
    act = action or ("inject the verdict and redo the work from the anchored step"
                     if anchor.level == "step" else
                     "inject the verdict and redo the committed answer")
    return ("anchor:\n"
            f"  level: {anchor.level}\n"
            f"  trigger: {_yaml_str(anchor.trigger)}\n"
            f"  evidence: {_yaml_str(anchor.evidence)}\n"
            f"  action: {_yaml_str(act)}\n")


def load_registry() -> Dict[str, object]:
    from pf_select.pf_select_eval import _load_pf_system
    _load_pf_system("skills")
    try:
        from skills_agent.skills.program_functions import _PF_REGISTRY
    except ImportError:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY  # type: ignore
    return dict(_PF_REGISTRY)


def find_cards() -> Dict[str, Path]:
    """skill_id -> SKILL.md. A skill documented in both trees keeps the
    executable card, which is the one describing what actually runs."""
    out: Dict[str, Path] = {}
    for pat in ("skills/textual/*/*/SKILL.md", "skills/executable/docs/*/*/SKILL.md"):
        for p in sorted(glob.glob(str(_HASP / pat))):
            t = Path(p).read_text()
            m = re.search(r"^skill_id:\s*(\S+)", t, re.M)
            if m:
                out[m.group(1)] = Path(p)
    return out


def card_anchor(text: str) -> Optional[Tuple[str, str]]:
    m = _ANCHOR_BLOCK.search(text)
    if not m:
        return None
    lvl = re.search(r"level:\s*\"?([a-z]+)", m.group(0))
    ev = re.search(r"evidence:\s*\"?([a-z]+)", m.group(0))
    return (lvl.group(1) if lvl else "", ev.group(1) if ev else "")


def sync_one(path: Path, anchor) -> Tuple[bool, str]:
    """-> (changed, note). Rewrites only the anchor block."""
    text = path.read_text()
    fm = _FRONT.search(text)
    if not fm:
        return False, "no frontmatter"
    head, rest = text[:fm.end()], text[fm.end():]
    block = render_anchor(anchor)

    if _ANCHOR_BLOCK.search(head):
        # A lambda replacement: re.sub parses backslash escapes in a string
        # replacement, and triggers contain things like \boxed{} — a string
        # replacement mangles them, differently on each run.
        new_head = _ANCHOR_BLOCK.sub(lambda _m: block, head, count=1)
        note = "updated"
    else:
        m = _AFTER.search(head) or None
        if m:
            new_head = head[:m.end()] + block + head[m.end():]
        else:
            b = _BEFORE.search(head)
            pos = b.start() if b else head.rfind("\n---")
            new_head = head[:pos] + block + head[pos:]
        note = "inserted"
    if new_head == head:
        return False, "unchanged"
    path.write_text(new_head + rest)
    return True, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not (a.write or a.check):
        ap.error("give --write or --check")

    reg, cards = load_registry(), find_cards()
    problems, changed = [], 0

    for sid, pf in sorted(reg.items()):
        anchor = getattr(type(pf), "pf_anchor", None)
        p = cards.get(sid)
        if p is None:
            problems.append(f"{sid}: registered but has no SKILL.md — unselectable")
            continue
        if anchor is None:
            problems.append(f"{sid}: registered with no anchor in code")
            continue
        have = card_anchor(p.read_text())
        if a.write:
            did, note = sync_one(p, anchor)
            changed += did
            if did:
                print(f"  {note:<8} {sid:<36} {anchor.level}/{anchor.evidence}")
        elif have is None:
            problems.append(f"{sid}: card has no anchor block "
                            f"(code says {anchor.level}/{anchor.evidence})")
        elif have != (anchor.level, anchor.evidence):
            problems.append(f"{sid}: card says {have[0]}/{have[1]}, "
                            f"code says {anchor.level}/{anchor.evidence}")

    for sid in sorted(set(cards) - set(reg)):
        problems.append(f"{sid}: card exists but nothing registers it")

    if a.write:
        print(f"[sync] {changed} card(s) updated, {len(reg)} skills checked")
        if problems:
            print("[sync] unresolved:")
            for x in problems:
                print("   ", x)
        return

    if problems:
        print(f"❌ {len(problems)} anchor problem(s)")
        for x in problems:
            print("   ", x)
        sys.exit(1)
    print(f"✅ anchors in step: {len(reg)} skills, code and cards agree")


if __name__ == "__main__":
    main()
