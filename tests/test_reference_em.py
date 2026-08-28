"""Lock the reference scorer's semantics.

`verifiers/reference_em.py` reproduces the upstream evaluation harness's
scorer. Every measured number in this repository depends on it behaving
*exactly* as it does today — including the two counter-intuitive cases below.
These are not bugs to fix; changing them invalidates the results.

Run with: python -m pytest tests/test_reference_em.py
"""

from verifiers.reference_em import (
    em_match, em_match_multi, extract_answer_math, extract_answer_web, normalize,
)


class TestNormalize:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize("  The   ANSWER  ") == "answer"

    def test_drops_articles(self):
        assert normalize("a cat an owl the dog") == "cat owl dog"

    def test_strips_ascii_punctuation(self):
        assert normalize("1,234.5") == "12345"

    def test_decimal_point_is_stripped(self):
        """The hazard behind the AMC23 gold-format bug: "27" != "27.0"."""
        assert normalize("27") == "27"
        assert normalize("27.0") == "270"
        assert not em_match("27", "27.0")


class TestExtractAnswerMath:
    def test_prefers_last_finish_action(self):
        assert extract_answer_math("finish[1]\nfinish[ 294 ]") == "294"

    def test_falls_back_to_final_answer_recorded(self):
        assert extract_answer_math("Final answer recorded: 42\nmore") == "42"

    def test_falls_back_to_last_boxed(self):
        assert extract_answer_math(r"\boxed{1} then \boxed{7}") == "7"

    def test_boxed_does_not_handle_nested_braces(self):
        """Preserved on purpose: `[^}]+` stops at the first closing brace.

        Unreachable on ReAct rollouts, where the finish[...] branch fires first.
        """
        assert extract_answer_math(r"\boxed{\frac{1}{2}}") == r"\frac{1"

    def test_falls_back_to_answer_colon(self):
        assert extract_answer_math("Answer: 13\ntrailing") == "13"

    def test_falls_back_to_answer_tags(self):
        assert extract_answer_math("<answer>x</answer><answer>y</answer>") == "y"

    def test_falls_back_to_last_non_empty_line(self):
        assert extract_answer_math("alpha\n\n  beta  \n\n") == "beta"

    def test_empty_input(self):
        assert extract_answer_math("") == ""

    def test_extraction_order_finish_beats_boxed(self):
        assert extract_answer_math(r"\boxed{9} ... finish[5]") == "5"


class TestExtractAnswerWeb:
    def test_prefers_answer_tags(self):
        assert extract_answer_web("<answer>Paris</answer>") == "Paris"

    def test_ignores_finish_actions(self):
        """The web extractor has no finish[...] branch."""
        assert extract_answer_web("finish[Paris]") == "finish[Paris]"


class TestEmMatchMulti:
    def test_accepts_any_semicolon_separated_alternative(self):
        assert em_match_multi("dog", "cat; dog ; owl")

    def test_rejects_a_non_alternative(self):
        assert not em_match_multi("fox", "cat; dog")

    def test_normalises_both_sides(self):
        assert em_match_multi("The Dog.", "dog")

    def test_single_gold_behaves_like_em_match(self):
        for pred, gold in [("5", "5"), ("5", "6"), ("", ""), ("the", "a")]:
            assert em_match_multi(pred, gold) == em_match(pred, gold)
