"""Answer comparison checkers — from cheap to expensive."""

import re
import math
from fractions import Fraction
from typing import Optional


def exact_match(candidate: str, ground_truth: str) -> bool:
    """Exact string match after normalization."""
    return candidate == ground_truth


def numeric_match(candidate: str, ground_truth: str, tol: float = 1e-6) -> bool:
    """Compare numeric values (int, float, fraction, percentage)."""
    c_val = _parse_numeric(candidate)
    g_val = _parse_numeric(ground_truth)

    if c_val is None or g_val is None:
        return False

    if g_val == 0:
        return abs(c_val) < tol

    return abs(c_val - g_val) / max(abs(g_val), 1e-10) < tol


def expression_match(candidate: str, ground_truth: str) -> bool:
    """Structural comparison of mathematical expressions.

    Handles reordering of commutative operations, set elements, etc.
    """
    # Set comparison: {a, b, c} vs {c, a, b}
    c_set = _parse_set(candidate)
    g_set = _parse_set(ground_truth)
    if c_set is not None and g_set is not None:
        return c_set == g_set

    # Tuple/pair comparison: (a, b) — order matters
    c_tuple = _parse_tuple(candidate)
    g_tuple = _parse_tuple(ground_truth)
    if c_tuple is not None and g_tuple is not None:
        if len(c_tuple) != len(g_tuple):
            return False
        return all(
            exact_match(c, g) or numeric_match(c, g)
            for c, g in zip(c_tuple, g_tuple)
        )

    # Commutative addition: a + b == b + a
    c_terms = _split_addition(candidate)
    g_terms = _split_addition(ground_truth)
    if c_terms and g_terms and len(c_terms) > 1:
        return sorted(c_terms) == sorted(g_terms)

    return False


def sympy_equivalence(candidate: str, ground_truth: str) -> bool:
    """Use sympy to check mathematical equivalence. Expensive but general."""
    try:
        from sympy import simplify, sympify, Eq
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return False

    # Try direct sympify first
    c_expr = _safe_sympify(candidate)
    g_expr = _safe_sympify(ground_truth)

    if c_expr is None or g_expr is None:
        return False

    try:
        diff = simplify(c_expr - g_expr)
        return diff == 0 or diff.is_zero
    except Exception:
        pass

    try:
        return bool(Eq(c_expr, g_expr).simplify())
    except Exception:
        return False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_numeric(s: str) -> Optional[float]:
    """Parse a string as a numeric value."""
    s = s.strip()

    # Percentage: 25% → 0.25
    if s.endswith('%'):
        inner = s[:-1].strip()
        val = _parse_numeric(inner)
        return val / 100 if val is not None else None

    # Fraction: 3/4
    if '/' in s and not re.search(r'[a-zA-Z]', s):
        parts = s.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    return num / den
            except ValueError:
                pass

    # Direct float/int
    try:
        return float(s)
    except ValueError:
        pass

    # Scientific notation: 1.5e3, 2 × 10^3
    sci_match = re.match(r'^(-?\d+\.?\d*)\s*[×x*]\s*10\^?\{?(\d+)\}?$', s)
    if sci_match:
        try:
            return float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))
        except ValueError:
            pass

    return None


def _parse_set(s: str) -> Optional[set]:
    """Parse {a, b, c} into a frozenset of normalized elements."""
    s = s.strip()
    if not (s.startswith('{') and s.endswith('}')):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return set()
    elements = [e.strip() for e in inner.split(',')]
    return set(elements)


def _parse_tuple(s: str) -> Optional[list]:
    """Parse (a, b, c) into a list of elements."""
    s = s.strip()
    if not (s.startswith('(') and s.endswith(')')):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [e.strip() for e in inner.split(',')]


def _split_addition(s: str) -> Optional[list]:
    """Split expression by top-level + signs."""
    # Only split if no nested parens contain +
    if '(' in s or '{' in s:
        return None
    terms = [t.strip() for t in re.split(r'\s*\+\s*', s)]
    if len(terms) > 1:
        return terms
    return None


def _safe_sympify(s: str):
    """Attempt to parse string into sympy expression."""
    try:
        from sympy import sympify, pi, E, I, oo
        from sympy.parsing.latex import parse_latex

        # Try LaTeX parsing if it looks like LaTeX
        if '\\' in s:
            try:
                return parse_latex(s)
            except Exception:
                pass

        # Clean up for sympify
        clean = s.replace('^', '**')
        clean = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', clean)

        return sympify(clean, locals={'pi': pi, 'e': E, 'i': I, 'inf': oo})
    except Exception:
        return None
