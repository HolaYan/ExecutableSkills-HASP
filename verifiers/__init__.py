"""Answer verification.

Two independent checkers live here, and they are not interchangeable:

- `verify_answer` — the lenient cascading pipeline (exact → numeric →
  structural → sympy). Use it when you want to know whether two answers mean
  the same thing.
- `reference_em` — the strict normalised exact match that every measured
  number in this repository was produced with. Use it when reproducing or
  extending a measured result; swapping in `verify_answer` would silently
  change every reported figure.
"""

from .base import VerifyResult, verify_answer
from .extractors import extract_boxed_answer, extract_final_answer
from .normalizers import normalize_answer, normalize_latex
from .checkers import (
    exact_match,
    numeric_match,
    sympy_equivalence,
    expression_match,
)
from .reference_em import (
    em_match,
    em_match_multi,
    extract_answer_math,
    extract_answer_web,
)

__all__ = [
    "VerifyResult",
    "verify_answer",
    "extract_boxed_answer",
    "extract_final_answer",
    "normalize_answer",
    "normalize_latex",
    "exact_match",
    "numeric_match",
    "sympy_equivalence",
    "expression_match",
    "em_match",
    "em_match_multi",
    "extract_answer_math",
    "extract_answer_web",
]
