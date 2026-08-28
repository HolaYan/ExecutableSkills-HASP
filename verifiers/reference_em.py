"""The reference exact-match scorer, inlined.

Every number in this repository was measured with the upstream evaluation
harness's own scorer, so HASP used to import it directly. That made the math
path unusable without a checkout of that repository. The functions are
reproduced here **verbatim in semantics** — same normalisation, same extraction
order, same fallbacks — so results stay on exactly the same footing while the
dependency goes away.

This is deliberately NOT `verifiers.verify_answer`. That one cascades through
numeric, structural and sympy equivalence and is far more permissive; swapping
it in would silently change every reported number. Use this module when
reproducing or extending a measured result, and `verify_answer` when you want
the lenient checker.

Semantics, for the record:

- `normalize` lowercases, strips ASCII punctuation, drops the articles
  a/an/the, and collapses whitespace.
- `em_match_multi` splits gold on `;` and accepts any alternative whose
  normalised form equals the normalised prediction.
- `extract_answer_math` prefers the LAST `finish[...]` action — the canonical
  extraction under the ReAct prompt — then `Final answer recorded:`, then the
  last `\\boxed{...}`, then `Answer: ...`, then the last `<answer>...</answer>`,
  and finally the last non-empty line.

Note the two known sharp edges, both of which have bitten this project:

- Normalisation strips punctuation, so `"27"` and `"27.0"` do NOT match — the
  decimal point is removed and `270` remains. Gold answers stored as float
  strings must be normalised at load time (`pf_select.eval_models.norm_gold`).
- `\\boxed{...}` is matched with `[^}]+`, so a nested brace such as
  `\\boxed{\\frac{1}{2}}` extracts as `\\frac{1`. The `finish[...]` branch
  fires first on ReAct rollouts, so this is only reachable on legacy traces.

Both behaviours are preserved on purpose: they are part of how the reference
numbers were produced.
"""

from __future__ import annotations

import re
import string

__all__ = [
    "normalize",
    "em_match",
    "em_match_multi",
    "extract_answer_math",
    "extract_answer_web",
]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_answer_math(text: str) -> str:
    # ReAct: prefer the LAST `finish[<answer>]` action (the trajectory's
    # final answer). This is the canonical extraction under the ReAct prompt.
    m = re.findall(r"finish\s*\[\s*(.+?)\s*\]", text, re.DOTALL)
    if m:
        return m[-1].strip()
    # Engine-format `Final answer recorded: X` is also a strong signal.
    m = re.search(r"Final answer recorded\s*:\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    # Try \boxed{} (legacy / non-ReAct rollouts)
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m[-1].strip()
    # Try "Answer: X" pattern
    m = re.search(r"(?:answer|Answer)\s*[:=]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    # Try <answer> tags
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m[-1].strip()
    # Last line as fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def extract_answer_web(text: str) -> str:
    # Try <answer> tags
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m[-1].strip()
    # Try "Answer: X"
    m = re.search(r"(?:answer|Answer)\s*[:=]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    # Last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def em_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def em_match_multi(prediction: str, gold: str) -> bool:
    """Gold may hold several acceptable answers separated by `;`."""
    targets = [t.strip() for t in gold.split(";")]
    norm_pred = normalize(prediction)
    return any(normalize(t) == norm_pred for t in targets)
