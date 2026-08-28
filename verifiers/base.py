"""Base verification logic — cascading verifier pipeline."""

from dataclasses import dataclass
from typing import Optional

from .extractors import extract_boxed_answer, extract_final_answer
from .normalizers import normalize_answer
from .checkers import exact_match, numeric_match, sympy_equivalence, expression_match


@dataclass
class VerifyResult:
    is_correct: bool
    candidate_normalized: str
    ground_truth_normalized: str
    method: str  # which checker matched


def verify_answer(
    candidate: str,
    ground_truth: str,
    answer_type: Optional[str] = None,
) -> VerifyResult:
    """Cascading verification: try each method from cheapest to most expensive.

    Order: exact_match → numeric_match → expression_match → sympy_equivalence.
    """
    cand_norm = normalize_answer(candidate)
    gt_norm = normalize_answer(ground_truth)

    if not cand_norm:
        return VerifyResult(False, cand_norm, gt_norm, "empty_candidate")

    # 1. Exact string match (after normalization)
    if exact_match(cand_norm, gt_norm):
        return VerifyResult(True, cand_norm, gt_norm, "exact_match")

    # 2. Numeric match (handles floats, fractions, percentages)
    if numeric_match(cand_norm, gt_norm):
        return VerifyResult(True, cand_norm, gt_norm, "numeric_match")

    # 3. Expression match (structural LaTeX comparison)
    if expression_match(cand_norm, gt_norm):
        return VerifyResult(True, cand_norm, gt_norm, "expression_match")

    # 4. Sympy equivalence (expensive but most general)
    if sympy_equivalence(cand_norm, gt_norm):
        return VerifyResult(True, cand_norm, gt_norm, "sympy_equivalence")

    return VerifyResult(False, cand_norm, gt_norm, "no_match")
