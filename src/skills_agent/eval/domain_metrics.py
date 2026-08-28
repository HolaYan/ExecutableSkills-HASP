"""
Domain-specific evaluation metrics.

Currently only web_search is supported.  The dispatch functions are kept
so that call-sites do not need to change.

Metrics follow the web-search benchmark definitions:
    EM, F1, Cover-EM, search/read counts.

Dispatch:
    metrics = compute_domain_metrics(episode, domain="web_search")
    agg     = aggregate_domain_metrics(metrics_list, domain="web_search")
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import math

from .episode import Episode
from .metrics import EpisodeMetrics, AggregatedMetrics

logger = logging.getLogger(__name__)


# ===================================================================
# Combinatorics helper for pass@k
# ===================================================================

def _comb(n: int, k: int) -> int:
    """Compute C(n, k) safely."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: probability that at least 1 of k trials succeeds.

    pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return float(c > 0)
    denom = _comb(n, k)
    if denom == 0:
        return 0.0
    return 1.0 - _comb(n - c, k) / denom


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k: probability that ALL k trials succeed.

    pass^k = C(c, k) / C(n, k)
    """
    if n < k:
        return float(c == n and n > 0)
    denom = _comb(n, k)
    if denom == 0:
        return 0.0
    return _comb(c, k) / denom


# ===================================================================
# Unified dispatch (web_search only)
# ===================================================================

def compute_domain_metrics(
    episode: Episode,
    domain: str = "web_search",
    sample: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> EpisodeMetrics:
    """Dispatch to domain-specific metric computation."""
    from .metrics import compute_metrics, MathAnswerEvaluator, CodeAnswerEvaluator

    if domain == "math":
        # Math uses \boxed{} extraction + numerical equivalence.
        # Build metrics from scratch since compute_metrics hardcodes text eval.
        prediction = episode.get_answer()
        gold = episode.gold_answers or []
        if isinstance(gold, str):
            gold = [gold]
        ev = MathAnswerEvaluator
        em = ev.exact_match(prediction, gold)
        f1 = ev.f1_score(prediction, gold)
        # cover_exact_match unused for math; reuse em.
        return EpisodeMetrics(
            exact_match=em,
            f1_score=f1,
            cover_exact_match=em,
            has_read=False,
            step_count=episode.get_step_count(),
            search_count=0,
            read_count=0,
            valid_structure=bool(prediction),
        )

    if domain == "code":
        # Code: pass@1 from sandbox execution. Without this branch the path
        # falls through to text-EM in compute_metrics (which always yields 0
        # on Python source vs gold answer string). The dataset row carries
        # `eval_test_code` for HumanEval+/MBPP+/BCB or `public_tests` /
        # `private_tests` for legacy LCB; we prefer the combined driver when
        # present.
        prediction = episode.get_answer()
        s = sample or {}
        eval_test_code = s.get("eval_test_code")
        entry_point = s.get("entry_point")
        gold_tests = s.get("private_tests") or s.get("public_tests") or []
        func_name = (s.get("metadata") or {}).get("func_name") if isinstance(s.get("metadata"), dict) else None
        # Generous timeouts — BCB imports alone take ~5s.
        sandbox_kwargs = {"cpu_seconds": 20, "wall_timeout_s": 30.0}
        try:
            res = CodeAnswerEvaluator.evaluate(
                prediction or "",
                gold_tests=gold_tests,
                func_name=func_name,
                sandbox_kwargs=sandbox_kwargs,
                eval_test_code=eval_test_code,
                entry_point=entry_point,
            )
            em = bool(res.pass_at_1)
            f1 = float(res.pass_rate)
        except Exception as e:
            logger.warning("CodeAnswerEvaluator failed for episode %s: %s",
                           getattr(episode, "sample_id", "?"), e)
            em, f1 = False, 0.0
        return EpisodeMetrics(
            exact_match=em,
            f1_score=f1,
            cover_exact_match=em,
            has_read=False,
            step_count=episode.get_step_count(),
            search_count=0,
            read_count=0,
            valid_structure=bool(prediction),
        )

    return compute_metrics(episode)


def aggregate_domain_metrics(
    metrics_list: list,
    domain: str = "web_search",
) -> AggregatedMetrics:
    """Dispatch to domain-specific aggregation."""
    from .metrics import aggregate_metrics
    return aggregate_metrics(metrics_list)
