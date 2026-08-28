"""
Evaluation metrics for ASearcher.

Core metrics:
- answer_em: Exact Match
- answer_f1: Token-level F1 score
- has_read: Whether READ was used (>=1)
- avg_steps: Average number of steps
- avg_search_calls: Average SEARCH count
- avg_read_calls: Average READ count
- valid_structure: SEARCH -> READ -> FINAL pattern
- robustness_drop: score_clean - score_adv
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import string
import re

from .episode import Episode


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    # Answer quality
    exact_match: bool = False
    f1_score: float = 0.0
    cover_exact_match: bool = False

    # Tool usage
    has_read: bool = False
    step_count: int = 0
    search_count: int = 0
    read_count: int = 0

    # Structure validity
    valid_structure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "cover_exact_match": self.cover_exact_match,
            "has_read": self.has_read,
            "step_count": self.step_count,
            "search_count": self.search_count,
            "read_count": self.read_count,
            "valid_structure": self.valid_structure,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple episodes."""
    # Answer quality
    answer_em: float = 0.0
    answer_f1: float = 0.0
    answer_cem: float = 0.0

    # Tool usage
    has_read_rate: float = 0.0
    avg_steps: float = 0.0
    avg_search_calls: float = 0.0
    avg_read_calls: float = 0.0

    # Structure
    valid_structure_rate: float = 0.0

    # Count
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_em": self.answer_em,
            "answer_f1": self.answer_f1,
            "answer_cem": self.answer_cem,
            "has_read_rate": self.has_read_rate,
            "avg_steps": self.avg_steps,
            "avg_search_calls": self.avg_search_calls,
            "avg_read_calls": self.avg_read_calls,
            "valid_structure_rate": self.valid_structure_rate,
            "num_samples": self.num_samples,
        }


class AnswerEvaluator:
    """Evaluator for answer quality."""

    ARTICLES = {"a", "an", "the"}

    def __init__(
        self,
        normalize: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_articles: bool = True,
    ):
        self.normalize = normalize
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_articles = remove_articles

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        if not self.normalize:
            return text

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove articles
        if self.remove_articles:
            words = text.split()
            words = [w for w in words if w not in self.ARTICLES]
            text = " ".join(words)

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return self._normalize_text(text).split()

    def exact_match(self, prediction: str, gold_answers: List[str]) -> bool:
        """Check exact match after normalization."""
        if not prediction or not gold_answers:
            return False

        pred_norm = self._normalize_text(prediction)

        for gold in gold_answers:
            if self._normalize_text(gold) == pred_norm:
                return True

        return False

    def f1_score(self, prediction: str, gold_answers: List[str]) -> float:
        """Compute max token-level F1 score."""
        if not prediction or not gold_answers:
            return 0.0

        pred_tokens = set(self._tokenize(prediction))
        if not pred_tokens:
            return 0.0

        max_f1 = 0.0

        for gold in gold_answers:
            gold_tokens = set(self._tokenize(gold))
            if not gold_tokens:
                continue

            common = pred_tokens & gold_tokens
            if not common:
                continue

            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)

            max_f1 = max(max_f1, f1)

        return max_f1

    @staticmethod
    def _bool_mapping(text: str) -> str:
        """Map boolean strings to yes/no (following ASearcher convention)."""
        if text == "True":
            return "yes"
        elif text == "False":
            return "no"
        return text

    def cover_exact_match(self, prediction: str, gold_answers: List[str]) -> bool:
        """
        Cover Exact Match: all gold tokens must appear in prediction tokens.

        For any gold answer, if every word in the normalized gold appears
        in the normalized prediction word list, return True.
        """
        if not prediction or not gold_answers:
            return False

        pred_normalized = self._normalize_text(self._bool_mapping(prediction))
        pred_list = pred_normalized.split()

        for gold in gold_answers:
            gold_normalized = self._normalize_text(self._bool_mapping(gold))
            gold_list = gold_normalized.split()
            if gold_list and all(word in pred_list for word in gold_list):
                return True

        return False


class MathAnswerEvaluator:
    """Evaluator for math answers. Extracts `\\boxed{...}` content and compares
    for numerical / symbolic equivalence rather than raw string EM.

    Used for the MATH domain (AIME24 / AIME25 / MATH-500). Handles:
      - `\\boxed{42}`, `\\boxed{\\frac{1}{2}}`, `\\boxed{2\\sqrt{3}}`
      - Bare integer answers (AIME: 0-999)
      - Fraction equivalence (1/2 == 0.5)
      - Whitespace / LaTeX normalization
    """

    _BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    _TRAILING_RE = re.compile(r"[\s\.,;:]+$")
    _TEX_CLEANUP = [
        (r"\\left", ""), (r"\\right", ""),
        (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\:", ""),
        (r"\\ ", " "), (r"\\textrm", ""), (r"\\text", ""),
    ]

    @classmethod
    def extract(cls, text: str) -> str:
        """Extract the canonical answer from a model output string.
        Priority: last \\boxed{} > 'Answer:' prefix > last number-like token > stripped text."""
        if text is None:
            return ""
        s = str(text).strip()
        # 1) Last \boxed{...}
        boxes = cls._BOX_RE.findall(s)
        if boxes:
            return boxes[-1].strip()
        # 2) "Answer: ..." prefix
        m = re.search(r"(?:final\s*answer|answer)\s*[:=]\s*([^\n]+)", s, re.IGNORECASE)
        if m:
            return cls._TRAILING_RE.sub("", m.group(1).strip())
        # 3) Last number-like token (integer/decimal/fraction)
        tokens = re.findall(r"-?\d+(?:[./]\d+)?", s)
        if tokens:
            return tokens[-1]
        return cls._TRAILING_RE.sub("", s)

    @classmethod
    def _normalize(cls, expr: str) -> str:
        """LaTeX cleanup + whitespace normalization."""
        if not expr:
            return ""
        e = str(expr).strip()
        e = cls._TRAILING_RE.sub("", e)
        for pat, rep in cls._TEX_CLEANUP:
            e = re.sub(pat, rep, e)
        # Collapse spaces
        e = re.sub(r"\s+", "", e)
        # Trim outer braces: {42} → 42
        while e.startswith("{") and e.endswith("}"):
            e = e[1:-1]
        return e

    @classmethod
    def _numeric(cls, expr: str):
        """Try to evaluate expr to a number via sympy; return (value, ok)."""
        if not expr:
            return None, False
        try:
            from sympy import sympify, Rational, Integer, Float
            from sympy.parsing.latex import parse_latex  # may fail silently
        except Exception:
            return None, False
        # Try LaTeX parse first, then plain sympify
        for parser in (lambda s: parse_latex(s), lambda s: sympify(s, evaluate=True)):
            try:
                val = parser(expr)
                return val, True
            except Exception:
                continue
        return None, False

    @classmethod
    def exact_match(cls, prediction: str, gold_answers: List[str]) -> bool:
        if not prediction or not gold_answers:
            return False
        pred_raw = cls.extract(prediction)
        pred_n = cls._normalize(pred_raw)
        for gold in gold_answers:
            gold_raw = cls.extract(gold)
            gold_n = cls._normalize(gold_raw)
            # (a) string-equal after normalization
            if pred_n == gold_n and pred_n != "":
                return True
            # (b) numeric equivalence via sympy
            p_val, p_ok = cls._numeric(pred_n)
            g_val, g_ok = cls._numeric(gold_n)
            if p_ok and g_ok:
                try:
                    from sympy import simplify, Eq
                    if bool(simplify(p_val - g_val) == 0):
                        return True
                except Exception:
                    try:
                        if float(p_val) == float(g_val):
                            return True
                    except Exception:
                        pass
        return False

    @classmethod
    def f1_score(cls, prediction: str, gold_answers: List[str]) -> float:
        """Token-level F1 over the EXTRACTED answers (not raw outputs)."""
        if not prediction or not gold_answers:
            return 0.0
        pred_raw = cls.extract(prediction)
        pred_n = cls._normalize(pred_raw)
        if not pred_n:
            return 0.0

        def _toks(s: str) -> List[str]:
            # simple symbol tokenization: split on whitespace + punctuation
            return [t for t in re.split(r"[\s,;\\\(\)\{\}]+", s) if t]

        p_set = set(_toks(pred_n))
        if not p_set:
            return 0.0

        max_f1 = 0.0
        for gold in gold_answers:
            g_n = cls._normalize(cls.extract(gold))
            g_set = set(_toks(g_n))
            if not g_set:
                continue
            common = p_set & g_set
            if not common:
                continue
            prec = len(common) / len(p_set)
            rec = len(common) / len(g_set)
            f1 = 2 * prec * rec / (prec + rec)
            max_f1 = max(max_f1, f1)
        return max_f1


class CodeAnswerEvaluator:
    """Evaluator for LiveCodeBench-style code generation. Extracts a Python
    code block from the model output, executes it in a subprocess sandbox
    against (public+private) test cases, and reports pass@1 / pass_rate.

    Unlike Math/Answer evaluators this is NOT a string comparison — it
    requires the test cases. Callers must pass them in via `gold_tests`
    (list of {input, output, testtype} dicts) and optionally `func_name`
    (for testtype="functional").

    Imports the sandbox lazily so the metrics module stays importable on
    machines without `resource` (Windows) — math/web evaluators don't need it.
    """

    _CODE_FENCE_RE = re.compile(
        r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
    )
    _ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    _SOLUTION_CLASS_RE = re.compile(
        r"^(?:class\s+Solution\b|def\s+\w+\s*\()", re.MULTILINE
    )

    @classmethod
    def extract(cls, text: str) -> str:
        """Pull the most plausible Python code body out of model output."""
        if text is None:
            return ""
        s = str(text)

        # 1) ```python ... ``` fenced block — last fence wins (model may have
        #    drafts before the final answer).
        fences = cls._CODE_FENCE_RE.findall(s)
        if fences:
            return cls._sanitize_leading_junk(fences[-1].strip())

        # 2) <answer>...</answer> wrapper (matches our generic agent contract)
        m = cls._ANSWER_TAG_RE.search(s)
        if m:
            inner = m.group(1).strip()
            inner_fences = cls._CODE_FENCE_RE.findall(inner)
            if inner_fences:
                return cls._sanitize_leading_junk(inner_fences[-1].strip())
            return cls._sanitize_leading_junk(inner)

        # 3) FINAL("""...""") strips its triple-quoted arg straight into
        #    `final.answer`, so the typical case is bare-code starting with
        #    `from`/`import`/`class`/`def`. Returning s.strip() preserves
        #    leading imports that the previous SOLUTION_CLASS_RE-based
        #    anchor was incorrectly slicing away (it would drop
        #    "from typing import List" before "class Solution:").
        return cls._sanitize_leading_junk(s.strip())

    @staticmethod
    def _sanitize_leading_junk(code: str) -> str:
        """Strip model-format artifacts the agent harness sometimes leaks into
        the FINAL body:
          • XML-like wrappers — `<python>...</python>`, `<code>...</code>`,
            `<answer>...</answer>` — that the model invents because it sees
            `<think>` in the prompt.
          • Nested `<think>...</think>` blocks inside the code body —
            Qwen2.5-7B sometimes interprets "think inside <think>" as
            "first line of the python block must be <think>" → invalid
            Python (verified: 8/200 HE+ episodes hit SyntaxError this way).
          • Bare leading `<` from copying the `<your code>` placeholder
            (e.g. `<from typing import List`).
          • Markdown fences (```python ... ```).
          • A stray ``python`` language tag on its own line before the body.
        Idempotent — clean code passes through untouched.
        """
        if not code:
            return code
        s = code.strip()

        # 0) Strip nested <think>...</think> blocks first — they're never
        #    valid Python and only ever appear here through model confusion.
        import re as _re
        # First, balanced <think>...</think>.
        s = _re.sub(
            r"<\s*think\s*>.*?</\s*think\s*>",
            "",
            s, flags=_re.DOTALL | _re.IGNORECASE,
        ).strip()
        # Unbalanced: model often writes `<think>` then reasoning bullets,
        # then `"""` as a stray closer (no `</think>`), then resumes code.
        # Strip from `<think>` to the first `"""` OR the next blank line +
        # python-statement, whichever comes first.
        s = _re.sub(
            r'<\s*think\s*>.*?"""',
            "",
            s, flags=_re.DOTALL | _re.IGNORECASE,
        ).strip()
        # Drop any leftover unbalanced <think> / </think> tag.
        s = _re.sub(r"</?\s*think\s*>", "", s, flags=_re.IGNORECASE).strip()

        # 1) XML-style wrapper tags: <python>...</python>, <code>, <answer>.
        m = _re.match(r"\s*<\s*(python|code|answer|solution)\s*>\s*\n?", s, _re.IGNORECASE)
        if m:
            tag = m.group(1)
            s = s[m.end():]
            close = _re.search(r"\n?\s*</\s*" + _re.escape(tag) + r"\s*>\s*$", s, _re.IGNORECASE)
            if close:
                s = s[:close.start()]
            s = s.strip()

        # 2) Bare leading `<` immediately followed by a Python keyword
        #    (`<from`, `<import`, `<def`, `<class`, `<async`, `<@`).
        if s.startswith("<") and not s.startswith("<<"):
            nxt = s[1:].lstrip()
            if nxt.startswith(("from ", "import ", "def ", "class ",
                                "async ", "@", "#")):
                s = nxt

        # 3) Markdown fence (```python ... ``` or ``` ... ```).
        if s.startswith("```"):
            nl = s.find("\n")
            if nl != -1:
                s = s[nl + 1:]
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3].rstrip()

        # 4) Bare `python` language tag on its own first line.
        if s.startswith("python\n"):
            s = s[len("python\n"):]

        return s

    @classmethod
    def evaluate(
        cls,
        prediction: str,
        gold_tests: List[Dict[str, Any]],
        func_name: Optional[str] = None,
        sandbox_kwargs: Optional[Dict[str, Any]] = None,
        eval_test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ):
        """Returns the full CodeEvalResult (tests passed/total + per-test info).

        Two scoring paths, picked by which arg the caller supplied:

          * ``eval_test_code`` set → run the candidate concatenated with the
            EvalPlus/BCB driver script and treat exit-0 as pass@1 (used by
            humaneval_plus, mbpp_plus, bigcodebench).
          * Else → loop over LCB-style ``gold_tests`` (legacy LCB path).
        """
        from .code_sandbox import CodeSandbox, CodeEvalResult
        code = cls.extract(prediction or "")
        if eval_test_code:
            if not code.strip():
                return CodeEvalResult(passed=0, total=1)
            sb = CodeSandbox(**(sandbox_kwargs or {}))
            return sb.evaluate_with_test_script(code, eval_test_code, entry_point)
        if not gold_tests:
            return CodeEvalResult(passed=0, total=0)
        if not code.strip():
            return CodeEvalResult(passed=0, total=len(gold_tests))
        sb = CodeSandbox(**(sandbox_kwargs or {}))
        return sb.evaluate(code, gold_tests, func_name=func_name)

    @classmethod
    def exact_match(
        cls,
        prediction: str,
        gold_tests: List[Dict[str, Any]],
        func_name: Optional[str] = None,
        sandbox_kwargs: Optional[Dict[str, Any]] = None,
        eval_test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> bool:
        """pass@1: candidate passes the test driver (ALL tests / exit-0)."""
        return cls.evaluate(
            prediction, gold_tests, func_name, sandbox_kwargs,
            eval_test_code=eval_test_code, entry_point=entry_point,
        ).pass_at_1

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        gold_tests: List[Dict[str, Any]],
        func_name: Optional[str] = None,
        sandbox_kwargs: Optional[Dict[str, Any]] = None,
        eval_test_code: Optional[str] = None,
        entry_point: Optional[str] = None,
    ) -> float:
        """Partial credit = pass_rate (passed / total). For the EvalPlus/BCB
        path total=1, so this collapses to 0.0 / 1.0."""
        return cls.evaluate(
            prediction, gold_tests, func_name, sandbox_kwargs,
            eval_test_code=eval_test_code, entry_point=entry_point,
        ).pass_rate


def compute_metrics(
    episode: Episode,
    gold_answers: Optional[List[str]] = None,
) -> EpisodeMetrics:
    """
    Compute metrics for a single episode.

    Args:
        episode: The episode to evaluate
        gold_answers: Gold standard answers (uses episode.gold_answers if not provided)

    Returns:
        EpisodeMetrics object
    """
    gold = gold_answers or episode.gold_answers
    if isinstance(gold, str):
        gold = [gold]

    prediction = episode.get_answer()

    # Answer metrics
    evaluator = AnswerEvaluator()
    exact_match = evaluator.exact_match(prediction, gold)
    f1 = evaluator.f1_score(prediction, gold)
    cem = evaluator.cover_exact_match(prediction, gold)

    # Tool usage metrics
    search_count = episode.get_search_count()
    read_count = episode.get_read_count()
    step_count = episode.get_step_count()
    has_read = read_count > 0

    # Structure validity
    valid_structure = episode.has_valid_structure()

    return EpisodeMetrics(
        exact_match=exact_match,
        f1_score=f1,
        cover_exact_match=cem,
        has_read=has_read,
        step_count=step_count,
        search_count=search_count,
        read_count=read_count,
        valid_structure=valid_structure,
    )


def aggregate_metrics(
    metrics_list: List[EpisodeMetrics],
) -> AggregatedMetrics:
    """
    Aggregate metrics across multiple episodes.

    Args:
        metrics_list: List of EpisodeMetrics

    Returns:
        AggregatedMetrics object
    """
    if not metrics_list:
        return AggregatedMetrics()

    n = len(metrics_list)

    return AggregatedMetrics(
        answer_em=sum(m.exact_match for m in metrics_list) / n,
        answer_f1=sum(m.f1_score for m in metrics_list) / n,
        answer_cem=sum(m.cover_exact_match for m in metrics_list) / n,
        has_read_rate=sum(m.has_read for m in metrics_list) / n,
        avg_steps=sum(m.step_count for m in metrics_list) / n,
        avg_search_calls=sum(m.search_count for m in metrics_list) / n,
        avg_read_calls=sum(m.read_count for m in metrics_list) / n,
        valid_structure_rate=sum(m.valid_structure for m in metrics_list) / n,
        num_samples=n,
    )


def compute_robustness_drop(
    clean_metrics: AggregatedMetrics,
    adv_metrics: AggregatedMetrics,
) -> Dict[str, float]:
    """
    Compute robustness drop: clean - adv.

    Args:
        clean_metrics: Metrics on clean data
        adv_metrics: Metrics on adversarial data

    Returns:
        Dictionary with robustness drop values
    """
    return {
        "em_drop": clean_metrics.answer_em - adv_metrics.answer_em,
        "f1_drop": clean_metrics.answer_f1 - adv_metrics.answer_f1,
        "cem_drop": clean_metrics.answer_cem - adv_metrics.answer_cem,
        "has_read_drop": clean_metrics.has_read_rate - adv_metrics.has_read_rate,
        "valid_structure_drop": clean_metrics.valid_structure_rate - adv_metrics.valid_structure_rate,
    }


def metrics_to_csv_row(
    model_name: str,
    mode: str,
    metrics: AggregatedMetrics,
) -> Dict[str, Any]:
    """Convert metrics to CSV row format."""
    return {
        "model": model_name,
        "mode": mode,
        "answer_em": f"{metrics.answer_em:.4f}",
        "answer_f1": f"{metrics.answer_f1:.4f}",
        "answer_cem": f"{metrics.answer_cem:.4f}",
        "has_read_rate": f"{metrics.has_read_rate:.4f}",
        "avg_steps": f"{metrics.avg_steps:.2f}",
        "avg_search_calls": f"{metrics.avg_search_calls:.2f}",
        "avg_read_calls": f"{metrics.avg_read_calls:.2f}",
        "valid_structure_rate": f"{metrics.valid_structure_rate:.4f}",
        "num_samples": metrics.num_samples,
    }


def create_comparison_table(
    results: Dict[str, Dict[str, AggregatedMetrics]],
) -> List[Dict[str, Any]]:
    """
    Create comparison table across models and modes.

    Args:
        results: {model_name: {mode: AggregatedMetrics}}

    Returns:
        List of rows for CSV/table output
    """
    rows = []

    for model_name, mode_metrics in results.items():
        clean = mode_metrics.get("clean")
        adv = mode_metrics.get("adv")

        row = {
            "model": model_name,
            "clean_em": f"{clean.answer_em:.4f}" if clean else "-",
            "clean_f1": f"{clean.answer_f1:.4f}" if clean else "-",
            "clean_cem": f"{clean.answer_cem:.4f}" if clean else "-",
            "adv_em": f"{adv.answer_em:.4f}" if adv else "-",
            "adv_f1": f"{adv.answer_f1:.4f}" if adv else "-",
            "adv_cem": f"{adv.answer_cem:.4f}" if adv else "-",
        }

        # Robustness drop
        if clean and adv:
            drop = compute_robustness_drop(clean, adv)
            row["em_drop"] = f"{drop['em_drop']:.4f}"
            row["f1_drop"] = f"{drop['f1_drop']:.4f}"
            row["cem_drop"] = f"{drop['cem_drop']:.4f}"
        else:
            row["em_drop"] = "-"
            row["f1_drop"] = "-"
            row["cem_drop"] = "-"

        # Tool usage from clean mode
        if clean:
            row["has_read_rate"] = f"{clean.has_read_rate:.4f}"
            row["avg_steps"] = f"{clean.avg_steps:.2f}"
        else:
            row["has_read_rate"] = "-"
            row["avg_steps"] = "-"

        rows.append(row)

    return rows


def aggregate_pass_at_k(
    per_seed_metrics: Dict[int, List[EpisodeMetrics]],
) -> AggregatedMetrics:
    """
    Aggregate metrics across multiple seeds using Max@k strategy.

    For each sample index, takes the max EM, max F1, and max CEM across seeds,
    then averages these max values.

    Args:
        per_seed_metrics: {seed: [EpisodeMetrics per sample]}

    Returns:
        AggregatedMetrics with Max@k aggregated values
    """
    if not per_seed_metrics:
        return AggregatedMetrics()

    seeds = list(per_seed_metrics.keys())
    num_samples = len(per_seed_metrics[seeds[0]])

    # For each sample, take max across seeds
    max_em = []
    max_f1 = []
    max_cem = []
    sum_has_read = []
    sum_steps = []
    sum_search = []
    sum_read = []
    sum_valid = []

    for i in range(num_samples):
        sample_ems = [per_seed_metrics[s][i].exact_match for s in seeds]
        sample_f1s = [per_seed_metrics[s][i].f1_score for s in seeds]
        sample_cems = [per_seed_metrics[s][i].cover_exact_match for s in seeds]

        max_em.append(max(sample_ems))
        max_f1.append(max(sample_f1s))
        max_cem.append(max(sample_cems))

        # For non-answer metrics, average across seeds
        sum_has_read.append(any(per_seed_metrics[s][i].has_read for s in seeds))
        sum_steps.append(sum(per_seed_metrics[s][i].step_count for s in seeds) / len(seeds))
        sum_search.append(sum(per_seed_metrics[s][i].search_count for s in seeds) / len(seeds))
        sum_read.append(sum(per_seed_metrics[s][i].read_count for s in seeds) / len(seeds))
        sum_valid.append(any(per_seed_metrics[s][i].valid_structure for s in seeds))

    n = num_samples
    return AggregatedMetrics(
        answer_em=sum(max_em) / n,
        answer_f1=sum(max_f1) / n,
        answer_cem=sum(max_cem) / n,
        has_read_rate=sum(sum_has_read) / n,
        avg_steps=sum(sum_steps) / n,
        avg_search_calls=sum(sum_search) / n,
        avg_read_calls=sum(sum_read) / n,
        valid_structure_rate=sum(sum_valid) / n,
        num_samples=n,
    )
