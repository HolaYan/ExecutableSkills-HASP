"""Answer normalization — canonicalize math expressions for comparison."""

import re
from typing import Optional


def normalize_latex(text: str) -> str:
    """Normalize LaTeX formatting without changing mathematical meaning."""
    if not text:
        return ""

    s = text.strip()

    # Remove display math delimiters
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\[(\[]|\\[)\]]$', '', s)

    # Normalize common LaTeX commands
    replacements = [
        (r'\\text\{([^}]*)\}', r'\1'),
        (r'\\mathrm\{([^}]*)\}', r'\1'),
        (r'\\mathbf\{([^}]*)\}', r'\1'),
        (r'\\textbf\{([^}]*)\}', r'\1'),
        (r'\\left\s*', ''),
        (r'\\right\s*', ''),
        (r'\\,', ' '),
        (r'\\;', ' '),
        (r'\\!', ''),
        (r'\\quad', ' '),
        (r'\\qquad', ' '),
        (r'\\cdot', '*'),
        (r'\\times', '*'),
        (r'\\div', '/'),
        (r'\\infty', 'inf'),
        (r'\\pi', 'pi'),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    # Normalize fractions: \frac{a}{b} → (a)/(b)
    def replace_frac(match):
        # Simple case: \frac{X}{Y}
        return f"({match.group(1)})/({match.group(2)})"

    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', replace_frac, s)

    # Normalize sqrt: \sqrt{X} → sqrt(X), \sqrt[n]{X} → X^(1/n)
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^{}]+)\}', r'(\2)^(1/\1)', s)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)

    # Remove unnecessary braces around single characters: {x} → x
    s = re.sub(r'\{([a-zA-Z0-9])\}', r'\1', s)

    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


_BOXED_UNWRAP_RE = re.compile(r"^\s*\\boxed\s*\{(.+)\}\s*$", re.DOTALL)


def _unwrap_boxed(text: str) -> str:
    """If `text` is exactly `\\boxed{X}` (one outer wrapper, balanced braces),
    return X; otherwise return text unchanged. Iterates a few times in case
    the model wrote `\\boxed{\\boxed{X}}`."""
    s = text
    for _ in range(3):
        m = _BOXED_UNWRAP_RE.match(s)
        if not m:
            break
        inner = m.group(1)
        # Verify outer braces actually balanced — count from the start.
        # `\boxed{X{Y}Z}` should unwrap to `X{Y}Z`.
        depth = 0
        end = -1
        start = s.find("{", s.find("\\boxed"))
        if start < 0:
            break
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != len(s.rstrip()) - 1:
            break  # Outer brace doesn't end at string end → not a clean wrap.
        s = s[start + 1:end].strip()
    return s


def normalize_answer(text: str) -> str:
    """Full answer normalization pipeline."""
    if not text:
        return ""

    s = text.strip()

    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Unwrap a single outer `\boxed{...}` so wrapped and bare answers compare equal.
    s = _unwrap_boxed(s)

    # Normalize LaTeX
    s = normalize_latex(s)

    # Lowercase for text answers (but not pure math)
    if re.search(r'[a-zA-Z]{3,}', s) and not re.search(r'[\\^_{}]', s):
        # Looks like a text answer, not math
        s = s.lower()

    # Remove trailing periods/commas
    s = re.sub(r'[.,;:]+$', '', s).strip()

    # Normalize common number formats
    # Remove leading zeros: 007 → 7 (but keep 0.7)
    s = re.sub(r'^0+(\d)', r'\1', s)

    # Remove trailing zeros after decimal: 3.140 → 3.14, but keep "3.0" as "3.0"
    if '.' in s and re.match(r'^-?\d+\.\d+$', s):
        s = s.rstrip('0').rstrip('.')

    # Normalize negative signs
    s = s.replace('−', '-').replace('–', '-')

    # Normalize comma as thousands separator: 1,000 → 1000
    if re.match(r'^-?\d{1,3}(,\d{3})+$', s):
        s = s.replace(',', '')

    return s


def classify_answer_type(answer: str) -> str:
    """Classify the type of a math answer.

    Returns one of: integer, float, fraction, expression, set, tuple,
                    interval, text, matrix, multiple_choice
    """
    if not answer:
        return "empty"

    a = answer.strip()

    # Multiple choice
    if re.match(r'^[A-E]$', a):
        return "multiple_choice"

    # Integer
    if re.match(r'^-?\d+$', a):
        return "integer"

    # Float
    if re.match(r'^-?\d+\.\d+$', a):
        return "float"

    # Fraction (a/b or \frac{a}{b})
    if re.match(r'^-?\d+/\d+$', a) or '\\frac' in a:
        return "fraction"

    # Set {a, b, c}
    if a.startswith('{') and a.endswith('}'):
        return "set"

    # Tuple (a, b) or ordered pair
    if a.startswith('(') and a.endswith(')') and ',' in a:
        return "tuple"

    # Interval [a, b] or (a, b)
    if re.match(r'^[\[(]-?\d.*,.*-?\d.*[\])]$', a):
        return "interval"

    # Matrix
    if '\\begin{' in a and ('matrix' in a or 'pmatrix' in a):
        return "matrix"

    # General expression (contains operators or variables)
    if re.search(r'[+\-*/^=<>]|[a-z]', a):
        return "expression"

    return "text"
