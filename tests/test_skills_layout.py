"""The skill library's on-disk layout, pinned.

`skills/` is split by what a skill is, then by domain:

    textual/<domain>/<skill_id>/SKILL.md      cards only — no code
    executable/<domain>/skills.py             every skill declared once
    executable/docs/<domain>/<skill_id>/      their cards, kept apart from the code

Several consumers depend on that shape — the pf_select menu, training's rollout
runner, forge's emit stage, and the run-scoped copies evolving makes. These
tests fail if a move breaks any of them, which is the point: the last
reorganisation broke three consumers silently and each was found by counting
afterwards.

Run with: python -m pytest tests/test_skills_layout.py
"""

import pytest

from skills_layout import DOMAINS, KINDS, resolve

COUNTS = {("textual", "math"): 4, ("textual", "web"): 28, ("textual", "code"): 14,
          ("executable", "math"): 17, ("executable", "web"): 4, ("executable", "code"): 5}
TOTAL = sum(COUNTS.values())


def _skills_in(lib):
    return {d.name for sd in lib.skill_dirs for d in sd.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()}


@pytest.mark.parametrize("kind,domain", sorted(COUNTS))
def test_each_half_has_its_documented_size(kind, domain):
    lib = resolve(f"skills/{kind}/{domain}")
    assert len(_skills_in(lib)) == COUNTS[(kind, domain)]


def test_root_resolves_to_every_skill():
    assert len(_skills_in(resolve("skills"))) == TOTAL


def test_legacy_domain_path_still_resolves():
    """Training configs carry `./skills/<domain>`, which no longer exists on
    disk. Both halves must still come back."""
    for domain in DOMAINS:
        lib = resolve(f"skills/{domain}")
        expected = COUNTS[("textual", domain)] + COUNTS[("executable", domain)]
        assert len(_skills_in(lib)) == expected, domain


def test_executable_code_and_cards_are_separate_trees():
    """The point of the split: no SKILL.md next to a checker."""
    for domain in DOMAINS:
        lib = resolve(f"skills/executable/{domain}")
        assert lib.pf_modules and lib.pf_modules[0].name == "skills.py"
        assert lib.skill_dirs and lib.skill_dirs[0].parent.name == "docs"
        code_dir = lib.pf_modules[0].parent
        assert not any((c / "SKILL.md").exists() for c in code_dir.iterdir() if c.is_dir())


def test_skills_py_is_the_only_registration_entry_point():
    """One file per domain registers every skill in it.

    Registration used to be spread over three files whose order decided which
    definition won. `skills.py` loads whatever else it needs itself, so the
    library has one entry point and no order to get wrong.
    """
    import pathlib
    for domain in DOMAINS:
        d = pathlib.Path("skills/executable") / domain
        assert (d / "skills.py").exists(), domain
        assert not (d / "evidence_pfs.py").exists(), f"{domain}: superseded by skills.py"
        assert not (d / "inject_pfs.py").exists(), f"{domain}: folded into skills.py"


def test_textual_half_holds_cards_and_nothing_else():
    """"Textual" has to mean textual.

    The implementations that used to live here moved under `executable/`, so
    every consumer finds code in one place. A `.py` reappearing here means the
    split has started to leak.
    """
    import pathlib
    for domain in DOMAINS:
        d = pathlib.Path("skills/textual") / domain
        assert not list(d.glob("*.py")), f"{domain}: code is supposed to be in executable/"
        lib = resolve(f"skills/textual/{domain}")
        assert lib.pf_modules == [], domain


def test_every_skill_dir_has_a_parsable_card():
    for d in _iter_skill_dirs():
        md = (d / "SKILL.md").read_text()
        assert md.startswith("---"), f"{d} has no YAML frontmatter — the loader would reject it"


def _iter_skill_dirs():
    for sd in resolve("skills").skill_dirs:
        for d in sd.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                yield d


def test_kinds_and_domains_are_the_only_top_level_split():
    import pathlib
    top = {p.name for p in pathlib.Path("skills").iterdir()
           if p.is_dir() and not p.name.startswith((".", "__"))}
    assert top == set(KINDS)
