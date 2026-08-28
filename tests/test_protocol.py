"""The protocol file and the code must not drift apart.

Every value in configs/protocol.yaml was, before it moved there, a literal in a
harness. These tests pin the two things that matter: the shipped file still
carries the values the reported results were produced with, and the loader
actually feeds them to the code that consumes them.

Run with: python -m pytest tests/test_protocol.py
"""

import pytest

from hasp_config import Protocol, load, protocol

# The measured protocol. Changing a number here means the reported results were
# produced under a different setting — update the paper, not just the test.
REPORTED = {
    ("accuracy", "n"): 64,
    ("accuracy", "temperature"): 0.7,
    ("accuracy", "max_tokens"): 8192,
    ("accuracy", "max_model_len"): 16384,
    ("serving", "tensor_parallel"): 1,
    ("serving", "gpu_memory_utilization"): 0.88,
    ("screen", "min_fire_wrong"): 0.03,
    ("screen", "max_fire_correct"): 0.06,
    ("screen", "min_lift"): 2.0,
    ("screen", "max_error_rate"): 0.02,
    ("segmentation", "base_min_len"): 40,
    ("segmentation", "base_max_steps"): 200,
    ("segmentation", "union_min_len"): 60,
    ("segmentation", "production_min_len"): 350,
    ("checkers", "compute_tolerance_rel"): 0.01,
    ("checkers", "compute_tolerance_abs"): 1e-6,
    ("locator", "locate_temperature"): 0.0,
    ("locator", "vote_temperature"): 0.6,
}


@pytest.mark.parametrize("key,expected", sorted(REPORTED.items()))
def test_shipped_file_carries_the_measured_value(key, expected):
    section, name = key
    assert getattr(getattr(protocol(), section), name) == expected


def test_dataclass_defaults_match_the_shipped_file():
    """The defaults in hasp_config.py are the fallback when the yaml is absent.

    If they drift from the file, a checkout without configs/ silently runs a
    different protocol — the exact failure this module exists to prevent.
    """
    fallback, shipped = Protocol(), protocol()
    for (section, name), _ in REPORTED.items():
        assert getattr(getattr(fallback, section), name) == \
               getattr(getattr(shipped, section), name), f"{section}.{name} drifted"


def test_base_pfs_matches_both_consumers():
    from anchor.eval_polished_pfs import PFS
    from skills_construct.forge.measure import BASE_PFS
    assert BASE_PFS == protocol().library.base_pfs
    assert PFS == protocol().library.base_pfs


def test_screen_thresholds_are_read_from_the_protocol():
    import skills_construct.forge.screen as screen
    s = protocol().screen
    assert (screen.MIN_FIRE_WRONG, screen.MAX_FIRE_CORRECT,
            screen.MIN_LIFT, screen.MAX_ERROR_RATE) == \
           (s.min_fire_wrong, s.max_fire_correct, s.min_lift, s.max_error_rate)


def test_missing_file_falls_back_instead_of_raising():
    p = load("/nonexistent/protocol.yaml")
    assert p.accuracy.n == Protocol().accuracy.n
    assert p.source == "<defaults>"


def test_unknown_keys_are_ignored(tmp_path):
    """A file written for a newer checkout must not break an older one."""
    f = tmp_path / "p.yaml"
    f.write_text("accuracy:\n  n: 7\n  future_knob: 1\nnot_a_section:\n  x: 1\n")
    p = load(f)
    assert p.accuracy.n == 7
    assert p.accuracy.max_tokens == Protocol().accuracy.max_tokens
