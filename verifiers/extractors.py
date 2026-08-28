"""Extract final answers from model outputs."""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract content from \boxed{...}, handling nested braces."""
    # Find all \boxed occurrences, take the last one (final answer)
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()

    # Walk through characters to find matching closing brace
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer from various formats.

    Supports:
      - \\boxed{...}
      - "Answer: ..."  /  "The answer is ..."
      - "FINAL ANSWER: ..."
      - Last line after "####"
    """
    # Try boxed first
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    # "Answer: X" or "The answer is X" pattern
    # Use greedy match to end of line (avoids splitting on decimal points)
    patterns = [
        r'(?:The\s+)?(?:final\s+)?[Aa]nswer\s*(?:is|:)\s*(.+)$',
        r'FINAL\s+ANSWER\s*:\s*(.+)$',
        r'####\s*(.+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Strip trailing sentence punctuation (but not decimal points in numbers)
            answer = re.sub(r'(?<!\d)[.;,]+$', '', answer).strip()
            answer = re.sub(r'\.$', '', answer).strip()
            if answer:
                return answer

    return None


def extract_answer_from_solution(solution: str) -> Optional[str]:
    """Extract from a full R1-style solution trace."""
    # R1 traces often end with boxed answer
    boxed = extract_boxed_answer(solution)
    if boxed:
        return boxed

    # Or "Answer: X" at the very end
    lines = solution.strip().split('\n')
    for line in reversed(lines[-5:]):
        answer = extract_final_answer(line)
        if answer:
            return answer

    return None
