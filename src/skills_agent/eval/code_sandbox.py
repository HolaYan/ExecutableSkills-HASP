"""Subprocess-based Python sandbox for evaluating LiveCodeBench solutions.

Two test_types LCB uses:

  * "stdin"      — codeforces-style: feed stdin, capture stdout, exact match.
  * "functional" — leetcode-style:    instantiate `Solution`, call a method,
                                      compare return value (json-equality).

Sandboxing approach (option A from the chat-pinned design): subprocess +
`resource` rlimits via preexec_fn. Practical for HPC research:

  * RLIMIT_CPU      → max CPU seconds (cgroups-style would be stronger but
                      requires root/cgroups; rlimit is good enough here)
  * RLIMIT_AS       → max virtual memory
  * RLIMIT_NPROC    → no fork bomb
  * RLIMIT_FSIZE    → no writing huge files
  * cwd =/tmp       → mild filesystem isolation
  * env stripped    → no leaked secrets

Not bulletproof against a determined attacker (no namespace isolation, can
still touch /home unless `unshare` is added later). Same posture as the
official LiveCodeBench evaluator. DO NOT run this against adversarial input
without an additional containment layer.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------

@dataclass
class TestResult:
    """One test case verdict."""
    status: str            # "pass" | "fail" | "timeout" | "runtime_error" | "compile_error"
    actual_output: str = ""
    error_msg: str = ""
    duration_s: float = 0.0


@dataclass
class CodeEvalResult:
    """Aggregate verdict across all tests for one solution."""
    passed: int = 0
    total: int = 0
    per_test: List[TestResult] = field(default_factory=list)
    # Useful for signal scoring: did it at least compile?
    syntax_ok: bool = True
    first_failure_msg: str = ""

    @property
    def pass_at_1(self) -> bool:
        """All tests pass — strict definition matching LCB's pass@1."""
        return self.total > 0 and self.passed == self.total

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


# ----------------------------------------------------------------------
# Sandbox
# ----------------------------------------------------------------------

class CodeSandbox:
    """Runs untrusted Python code against test cases under rlimit constraints."""

    def __init__(
        self,
        cpu_seconds: int = 6,                # RLIMIT_CPU hard cap per test
        memory_mb: int = 1024,               # RLIMIT_AS in MB
        wall_timeout_s: float = 10.0,        # subprocess wall-clock kill
        max_output_bytes: int = 256 * 1024,  # truncate captured stdout to this
        python_executable: Optional[str] = None,
    ):
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.wall_timeout_s = wall_timeout_s
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable

    # ------------------------------------------------------------------
    # Single test runners
    # ------------------------------------------------------------------

    def run_stdin_test(self, code: str, stdin: str, expected_stdout: str) -> TestResult:
        """Run `code` as a script, pipe `stdin`, compare stdout (rstripped)."""
        return self._run_subprocess(
            script=code,
            stdin=stdin,
            comparator=lambda out: out.rstrip() == expected_stdout.rstrip(),
            expected_repr=expected_stdout.rstrip()[:200],
        )

    def run_functional_test(
        self,
        code: str,
        func_name: str,
        input_args_repr: str,
        expected_output_json: str,
    ) -> TestResult:
        """LeetCode-style: instantiate Solution(), call method, compare json.

        `input_args_repr` is the LCB-encoded test args string (e.g. '"abc"\\n5'
        — newlines separate positional args). LCB stores expected output as a
        JSON-serialisable string (e.g. "true", "[1,2,3]", "\"foo\"").

        The wrapper is built by plain string concatenation rather than
        textwrap.dedent + f-string substitution: dedent picks the SMALLEST
        common-leading-whitespace count across all non-blank lines, and a
        user-supplied snippet whose internal indents (e.g. 4-space method
        bodies) are smaller than the template's source indent will leave
        residual leading whitespace on the prefix lines → IndentationError
        on `import json, sys, ast`.
        """
        prefix = "import json, sys, ast\n\n"
        suffix = (
            "\n"
            "_args_raw = sys.stdin.read()\n"
            "_arg_lines = _args_raw.split('\\n') if _args_raw else []\n"
            "_parsed = []\n"
            "for _ln in _arg_lines:\n"
            "    _ln = _ln.rstrip('\\r')\n"
            "    if not _ln:\n"
            "        continue\n"
            "    try:\n"
            "        _parsed.append(json.loads(_ln))\n"
            "    except Exception:\n"
            "        try:\n"
            "            _parsed.append(ast.literal_eval(_ln))\n"
            "        except Exception:\n"
            "            _parsed.append(_ln)\n"
            "\n"
            "_sol = Solution()\n"
            f"_result = getattr(_sol, {func_name!r})(*_parsed)\n"
            "print(json.dumps(_result, default=str, sort_keys=True))\n"
        )
        wrapper = prefix + code.rstrip() + "\n" + suffix

        def cmp(actual: str) -> bool:
            try:
                a = json.loads(actual.strip())
            except Exception:
                return actual.strip() == expected_output_json.strip()
            try:
                e = json.loads(expected_output_json)
            except Exception:
                return actual.strip() == expected_output_json.strip()
            return _semantic_equal(a, e)

        return self._run_subprocess(
            script=wrapper,
            stdin=input_args_repr if input_args_repr.endswith("\n") else input_args_repr + "\n",
            comparator=cmp,
            expected_repr=expected_output_json[:200],
        )

    # ------------------------------------------------------------------
    # eval_test_code path (HumanEval+, MBPP+, BigCodeBench)
    # ------------------------------------------------------------------

    def evaluate_with_test_script(
        self,
        candidate_code: str,
        test_script: str,
        entry_point: Optional[str] = None,
    ) -> CodeEvalResult:
        """Run ``<candidate_code>\\n\\n<test_script>`` as one Python script;
        exit-0 → pass@1, anything else → fail. Used by EvalPlus / BigCodeBench
        where each problem ships a single combined test driver instead of LCB's
        per-test list.

        Unlike :py:meth:`evaluate`, this path does NOT pass ``-I -S`` to
        Python — those benchmarks need third-party packages (numpy, pandas,
        unittest helpers, …) from site-packages. rlimits + stripped env still
        apply, so behaviour is "trusted-research code execution," same posture
        as the official EvalPlus runner.

        ``entry_point`` is unused here (the test scripts already call the
        candidate's symbol directly) but kept in the signature for symmetry.
        """
        try:
            compile(candidate_code, "<solution>", "exec")
        except SyntaxError as e:
            return CodeEvalResult(
                passed=0, total=1,
                per_test=[TestResult(status="compile_error", error_msg=str(e))],
                syntax_ok=False,
                first_failure_msg=f"SyntaxError: {e}",
            )
        try:
            compile(test_script, "<test>", "exec")
        except SyntaxError as e:
            return CodeEvalResult(
                passed=0, total=1,
                per_test=[TestResult(status="compile_error",
                                     error_msg=f"test script syntax error: {e}")],
                syntax_ok=True,
                first_failure_msg=f"test script SyntaxError: {e}",
            )

        # The candidate must be importable both as top-level statements AND as
        # `__main__` (BCB tests pass `if __name__ == '__main__'`). Direct
        # concatenation handles both cases — Python sets `__name__` to
        # `__main__` for the script itself.
        #
        # PREAMBLE: official EvalPlus / BCB runners always prepend the original
        # prompt (which contains the typing imports) before the candidate. We
        # reproduce that by injecting a small, idempotent stdlib preamble. This
        # rescues HumanEval candidates where the model emits
        # `def f(xs: List[int]) -> bool` without `from typing import List`.
        # Side effects: re-imports of these names by the candidate are no-ops;
        # candidate-level `from typing import *` still works.
        preamble = (
            "from typing import *  # auto-injected by sandbox preamble\n"
            "from collections import *\n"
            "import math, re, string, itertools, functools, heapq, bisect\n"
            "\n"
        )
        wrapper = preamble + candidate_code.rstrip() + "\n\n" + test_script

        tr = self._run_subprocess(
            script=wrapper,
            stdin="",
            comparator=lambda _: True,         # exit-code-only verdict
            expected_repr="",
            isolated=False,                     # need site-packages
        )
        passed = 1 if tr.status == "pass" else 0
        return CodeEvalResult(
            passed=passed, total=1, per_test=[tr],
            syntax_ok=True,
            first_failure_msg="" if passed else (
                f"{tr.status} — {tr.error_msg[:300]}" if tr.error_msg else tr.status
            ),
        )

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def evaluate(
        self,
        code: str,
        tests: List[Dict[str, Any]],
        func_name: Optional[str] = None,
    ) -> CodeEvalResult:
        """Run all `tests` against `code`. Stops at first non-pass for cost,
        but always reports total = len(tests). LCB pass@1 demands all-pass."""
        # Cheap syntax check first — a syntax error fails everything fast.
        try:
            compile(code, "<solution>", "exec")
        except SyntaxError as e:
            return CodeEvalResult(
                passed=0, total=len(tests),
                per_test=[TestResult(status="compile_error", error_msg=str(e))],
                syntax_ok=False,
                first_failure_msg=f"SyntaxError: {e}",
            )

        result = CodeEvalResult(total=len(tests))
        for i, t in enumerate(tests):
            test_type = (t.get("testtype") or "").lower()
            if test_type == "functional":
                if not func_name:
                    tr = TestResult(
                        status="runtime_error",
                        error_msg="functional test but func_name missing in metadata",
                    )
                else:
                    tr = self.run_functional_test(
                        code,
                        func_name=func_name,
                        input_args_repr=t.get("input", ""),
                        expected_output_json=t.get("output", ""),
                    )
            else:
                # default = stdin
                tr = self.run_stdin_test(
                    code,
                    stdin=t.get("input", ""),
                    expected_stdout=t.get("output", ""),
                )
            result.per_test.append(tr)
            if tr.status == "pass":
                result.passed += 1
            else:
                if not result.first_failure_msg:
                    result.first_failure_msg = (
                        f"test {i}: {tr.status} — {tr.error_msg[:200]}"
                        if tr.error_msg else f"test {i}: {tr.status}"
                    )
                # Optimisation: short-circuit on first failure (pass@1 needs all-pass).
                # Continue counting so per_test has length == total.
                # (cheaper alternative: break + extend with skipped — kept verbose for diagnostics)
        return result

    # ------------------------------------------------------------------
    # Internal: subprocess + rlimit
    # ------------------------------------------------------------------

    def _preexec_strict(self):
        """Locked-down rlimits for LCB-style problems (no third-party imports)."""
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            mem_bytes = self.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        except Exception:
            pass
        try:
            os.setsid()
        except Exception:
            pass

    def _preexec_relaxed(self):
        """Looser rlimits for HumanEval+/MBPP+/BigCodeBench. matplotlib/scipy/
        flask spawn many threads and jemalloc reserves a lot of address space,
        so RLIMIT_NPROC=64 / RLIMIT_AS=1GB are too tight. We keep CPU + wall
        timeout (the meaningful safety net) and drop the rest."""
        try:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (self.cpu_seconds, self.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE,
                               (256 * 1024 * 1024, 256 * 1024 * 1024))
        except Exception:
            pass
        try:
            os.setsid()
        except Exception:
            pass

    def _run_subprocess(
        self,
        script: str,
        stdin: str,
        comparator,
        expected_repr: str,
        isolated: bool = True,
    ) -> TestResult:
        import time as _time
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Empty HOME so user code can't reach ~/.cache etc.
            "HOME": "/tmp",
        }
        # `isolated=False` lets HumanEval+/MBPP+/BigCodeBench scripts import
        # numpy, pandas, unittest helpers etc. from site-packages. We still
        # keep rlimits and the stripped env, so it's a soft sandbox, not a
        # locked-down one — same posture as the official EvalPlus runner.
        if not isolated:
            for k in ("PYTHONPATH", "VIRTUAL_ENV"):
                if k in os.environ:
                    env[k] = os.environ[k]
        with tempfile.TemporaryDirectory(prefix="lcb_") as tmpdir:
            script_path = os.path.join(tmpdir, "sol.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            t0 = _time.time()
            py_args = [self.python_executable]
            if isolated:
                py_args += ["-I", "-S"]
            py_args.append(script_path)
            try:
                proc = subprocess.run(
                    py_args,
                    input=stdin.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir,
                    env=env,
                    timeout=self.wall_timeout_s,
                    preexec_fn=(self._preexec_strict if isolated
                                else self._preexec_relaxed),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return TestResult(
                    status="timeout",
                    error_msg=f"wall-clock > {self.wall_timeout_s}s",
                    duration_s=_time.time() - t0,
                )
            except Exception as e:
                return TestResult(
                    status="runtime_error",
                    error_msg=f"subprocess failed: {type(e).__name__}: {e}",
                    duration_s=_time.time() - t0,
                )
            duration = _time.time() - t0

        stdout = (proc.stdout or b"")[: self.max_output_bytes].decode("utf-8", errors="replace")
        stderr = (proc.stderr or b"")[: 8 * 1024].decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # SIGXCPU = -24 means RLIMIT_CPU triggered
            if proc.returncode in (-signal.SIGXCPU, 152):
                return TestResult(status="timeout", error_msg="cpu rlimit hit",
                                  duration_s=duration, actual_output=stdout)
            if proc.returncode in (-signal.SIGKILL, 137):
                return TestResult(status="timeout", error_msg="OOM or sigkill",
                                  duration_s=duration, actual_output=stdout)
            return TestResult(
                status="runtime_error",
                error_msg=stderr.strip()[-500:] or f"exit={proc.returncode}",
                actual_output=stdout,
                duration_s=duration,
            )

        ok = False
        try:
            ok = comparator(stdout)
        except Exception as e:
            return TestResult(
                status="runtime_error",
                error_msg=f"comparator failed: {e}",
                actual_output=stdout,
                duration_s=duration,
            )

        if ok:
            return TestResult(status="pass", actual_output=stdout, duration_s=duration)
        return TestResult(
            status="fail",
            actual_output=stdout[:500],
            error_msg=f"expected ≈ {expected_repr}",
            duration_s=duration,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _semantic_equal(a, b) -> bool:
    """Lenient JSON-value compare: ints == floats, lists order-sensitive, dicts
    by keys+values. Mirrors LCB's leniency on int/float and list-of-list output.
    """
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return False
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_semantic_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_semantic_equal(a[k], b[k]) for k in a)
    return a == b
