"""code PF skills — every skill writes out both modules.

    Detect   should_activate(state)  — does this state and proposed action
                                       match the failure pattern?
    Repair   intervene(state)        — redirect, inject corrective context, or
                                       abstain.

The five evidence skills mined from the code error cases. Each anchors on
something the submission or the spec itself contains, which is the property
every skill that ever produced a rescue shares.

`spec_example_check` is the strongest of these: the spec's own examples are an
executable statement of what the function must do, so a failure is proof rather
than suspicion. Its false fires are spec-side artifacts (HumanEval's own
`median` doctest is wrong; repr-vs-str quoting), not checker bugs.

Detect matters here more than anywhere: without it all five activate on every
FINAL, so a submission with no runnable examples still counts as activated, and
`edge_input_probe` — which fires on correct solutions by design of the specs —
runs on everything.
"""
from __future__ import annotations

import importlib.util as _iu
import re
import sys
from pathlib import Path

_HASP = Path(__file__).resolve().parents[3]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from skills.pf_template import (  # noqa: E402
    as_action, as_evidence, call_base_intervene, redirect, correction,
    Anchor, Finding, Ctx, abstain, inject, pf_skill, verdict,
)

# The checkers live next door; load them as a module rather than duplicating.
# `chain_load` in pf_template guards the resulting cycle (evidence_pfs loads
# this file, this file loads evidence_pfs).
_spec = _iu.spec_from_file_location(
    "_hasp_code_checkers", str(Path(__file__).resolve().parent / "checkers.py"))
_C = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_C)

_IMPL_SPEC = _iu.spec_from_file_location(
    "_hasp_code_impls", str(Path(__file__).resolve().parent / "implementations.py"))
_IMPL = _iu.module_from_spec(_IMPL_SPEC)
_IMPL_SPEC.loader.exec_module(_IMPL)

D = "code"



_DOCTEST = re.compile(r">>>")
_ASSERT = re.compile(r"\bassert\b")


def _submission(ctx: Ctx, arg: str) -> str:
    return arg or ctx.reasoning


def _run(ctx: Ctx, arg: str, checker) -> Finding | None:
    """Code checkers take `(step_context, answer)` and return a verdict string."""
    try:
        v = checker(ctx.raw, _submission(ctx, arg))
    except Exception:
        return None
    if isinstance(v, dict):        # {"verdict": ..., "fix": ...} -- see answer_finding
        return (Finding(verdict=v.get("verdict", ""), fix=v.get("fix"), data=v)
                if v.get("verdict") else None)
    return Finding(verdict=v) if v else None


# ── the measured one ─────────────────────────────────────────────────────

@pf_skill("spec_example_check", domain=D,
          anchor=Anchor(level="final", evidence="executed",
                        trigger="the spec carries its own `>>>` / `assert` examples, or "
                                "the runtime supplied public tests"),
          summary="Run the problem's own examples against the submitted solution and "
                  "report the first one it fails, with expected and actual values.")
class SpecExampleCheck:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(_DOCTEST.search(ctx.question) or _ASSERT.search(ctx.question)
                    or str(ctx.public_test_code).strip())

    #: single-edit mutations for the classic off-by-one / flipped-bound slips.
    #: Each candidate differs from the submission in ONE edit and ships only if
    #: the spec's own >>> examples all pass -- the gate carries the proof, so
    #: the mutation set can stay dumb.
    _MUTATIONS = (
        (re.compile(r"range\((\s*[^,()]+,\s*)([A-Za-z_]\w*|\d+)\s*\)"), r"range(\g<1>\g<2> + 1)"),
        (re.compile(r"(?<![<>=!])<(?!=)"), "<="),
        (re.compile(r"(?<![<>=!])>(?!=)"), ">="),
        (re.compile(r"<="), "<"),
        (re.compile(r">="), ">"),
    )

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.spec_example_evidence)
        if not f:
            return abstain()
        spec = ctx.question
        if _IMPL._passes_spec_examples(arg, spec) is False:
            for pat, rep in self._MUTATIONS:
                for m in pat.finditer(arg):
                    cand = arg[:m.start()] + pat.sub(rep, m.group(0), count=1) + arg[m.end():]
                    if cand != arg and _IMPL._passes_spec_examples(cand, spec) is True:
                        return redirect(action, cand,
                                        because=(f"{f.verdict[:120]} -- a one-edit repair "
                                                 f"passes every spec example"))
        return verdict(ctx, f)


# ── static contract checks ───────────────────────────────────────────────

@pf_skill("exception_contract_check", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the spec says an exception should be raised for some input"),
          summary="Flag a solution whose spec demands a specific exception that the code "
                  "never raises.")
class ExceptionContractCheck:
    def should_activate(self, ctx, action, arg) -> bool:
        return bool(re.search(r"\braise[sd]?\b|\bValueError\b|\bTypeError\b|\bexception\b",
                              ctx.question, re.I))

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.exception_contract)
        if not f:
            return abstain()
        # "Raises ValueError for b == 0" names both the exception and the
        # condition; when the condition is a simple comparison the guard
        # writes itself. The spec's ordinary examples must still pass, and a
        # candidate that fails to parse never ships.
        spec = ctx.question
        # both orders: "raises ValueError for b == 0" and "If divisor is 0,
        # raise a ValueError"; "is 0" normalises to "== 0"
        m = re.search(r"raises?\s+(?:an?\s+)?([A-Z]\w*(?:Error|Exception))"
                      r"[^\n]{0,40}?\b(?:for|when|if)\s+"
                      r"([A-Za-z_]\w*\s*(?:==|!=|<=|>=|<|>)\s*[\w.'\x22-]+)",
                      spec, re.I)
        cond = None
        if m:
            exc, cond = m.group(1), m.group(2).strip()
        else:
            m2 = re.search(r"\b(?:if|when)\s+(?:the\s+)?([A-Za-z_]\w*)\s+"
                           r"(?:is|equals|==)\s+([\w.'\x22-]+)\s*,?\s*"
                           r"raises?\s+(?:an?\s+)?([A-Z]\w*(?:Error|Exception))",
                           spec, re.I)
            if m2:
                exc = m2.group(3)
                # "is negative" / "is zero" are conditions, not values
                word = m2.group(2).lower()
                cond = {"negative": f"{m2.group(1)} < 0",
                        "zero": f"{m2.group(1)} == 0",
                        "positive": f"{m2.group(1)} > 0",
                        "empty": f"not {m2.group(1)}"}.get(
                            word, f"{m2.group(1)} == {m2.group(2)}")
        dm = re.search(r"^([ \t]*)def\s+\w+\([^)]*\):[ \t]*\n"
                      r"((?:\s*(?:'''|\x22\x22\x22)(?:.|\n)*?(?:'''|\x22\x22\x22)[ \t]*\n)?)",
                      arg, re.M)
        if cond and dm:
            indent = dm.group(1) + "    "
            guard = f"{indent}if {cond}:\n{indent}    raise {exc}({cond!r})\n"
            cand = arg[:dm.end()] + guard + arg[dm.end():]
            try:
                import ast
                ast.parse(cand)
                if _IMPL._passes_spec_examples(cand, spec) is not False:
                    return redirect(action, cand,
                                    because=f"the spec requires {exc} for {cond}; guard inserted")
            except SyntaxError:
                pass
        return verdict(ctx, f)


@pf_skill("signature_conformance_check", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the spec declares an entry point and the submission "
                                "defines a function"),
          summary="Check that the submitted function's name and arity match the "
                  "signature the spec declares.")
class SignatureConformanceCheck:
    """Precise by construction, and quiet on specs that define a helper before
    the target (`encode_cyclic` / `decode_cyclic`) once the spec's *last* def is
    taken as the entry point. Useful where signatures actually drift."""

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(ctx.entry_point) and "def " in _submission(ctx, arg)

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.signature_conformance)
        if f and f.fix:
            return redirect(action, f.fix, because=f.verdict)
        return verdict(ctx, f) if f else abstain()


@pf_skill("api_attribute_probe", domain=D,
          anchor=Anchor(level="final", evidence="executed",
                        trigger="the submission calls an attribute chain on a module it "
                                "imports"),
          summary="Check that every library attribute the code calls actually exists in "
                  "the installed library.")
class ApiAttributeProbe:
    """The sandbox runs under a 1 GB address-space rlimit with no network, so
    heavyweight imports (numpy) fail there for reasons that have nothing to do
    with the submission. A fire is a hint, not a proof."""

    def should_activate(self, ctx, action, arg) -> bool:
        code = _submission(ctx, arg)
        return bool(re.search(r"^\s*(?:import|from)\s+\w", code, re.M)
                    and re.search(r"\w+\.\w+\s*\(", code))

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.api_attribute_probe)
        if not f:
            return abstain()
        # The probe found the attribute missing AND dir() offered exactly one
        # close match ("arry" -> "array"): a single-token rename, shipped only
        # if the spec's own examples don't disprove it. The unfixed code raises
        # AttributeError on every call, so a parse-clean rename cannot be worse.
        bad_attr, sugg = f.data.get("bad_attr"), f.data.get("suggestion")
        if bad_attr and sugg:
            cand = re.sub(rf"\.{re.escape(bad_attr)}\b", f".{sugg}", arg)
            if cand != arg and _IMPL._passes_spec_examples(cand, ctx.question) is not False:
                try:
                    import ast
                    ast.parse(cand)
                    return redirect(action, cand, because=f.verdict[:160])
                except SyntaxError:
                    pass
        return verdict(ctx, f)


# ── opt-in only ──────────────────────────────────────────────────────────

@pf_skill("edge_input_probe", domain=D,
          anchor=Anchor(level="final", evidence="executed",
                        trigger="explicitly enabled via step_context['enable_edge_probe']"),
          summary="Run the entry point on empty and zero inputs and report an unhandled "
                  "exception.")
class EdgeInputProbe:
    """Inverted in practice: a passing solution often raises on empty input
    because the spec demands it, so this probe accuses correct code. The Detect
    below keeps it off by default — do not widen it without measuring the rate
    on solutions that already pass."""

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(ctx.enable_edge_probe)

    def intervene(self, ctx, action, arg):
        f = _run(ctx, arg, _C.edge_input_probe)
        return verdict(ctx, f) if f else abstain()


@pf_skill("code_split_whitespace", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger="the candidate calls split(' ') on a prompt asking for whitespace-separated words"),
          summary='This solution splits on a literal single space, which keeps '
                  'empty strings when separators repeat, but the problem is about '
                  'whitespace-separated words.')
class CodeSplitWhitespace:
    """Detect and Repair both delegate to `SplitWhitespacePF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.SplitWhitespacePF()
    NOTE = ('this solution splits on a literal single space, which keeps '
            'empty strings when separators repeat, but the problem is about '
            'whitespace-separated words.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_combinations_with_replacement", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the candidate uses itertools.combinations on a prompt whose count allows repeats'),
          summary='This solution uses `combinations`, which cannot repeat an '
                  'element, but the problem allows an element to be chosen more '
                  'than once.')
class CodeCombinationsWithReplacement:
    """Detect and Repair both delegate to `CombinationsReplacementPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.CombinationsReplacementPF()
    NOTE = ('this solution uses `combinations`, which cannot repeat an '
            'element, but the problem allows an element to be chosen more '
            'than once.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_specific_exception_type", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='an except or raise whose message keywords suggest a more specific exception type'),
          summary='This solution raises or catches a broader exception type than '
                  'the problem specifies.')
class CodeSpecificExceptionType:
    """Detect and Repair both delegate to `SpecificExceptionTypePF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.SpecificExceptionTypePF()
    NOTE = ('this solution raises or catches a broader exception type than '
            'the problem specifies.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_import_guard", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the code uses a name it never imports'),
          summary='This solution uses a name it never imports, so it raises '
                  'before producing any answer.')
class CodeImportGuard:
    """Detect and Repair both delegate to `ImportGuardPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.ImportGuardPF()
    NOTE = ('this solution uses a name it never imports, so it raises '
            'before producing any answer.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_numpy_return_pythonize", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the code returns a NumPy object where the examples compare '
                                'plain Python values'),
          summary="This solution returns a numpy object, while the problem's "
                  'examples compare against plain python values.')
class CodeNumpyReturnPythonize:
    """Detect and Repair both delegate to `NumpyReturnPythonizePF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.NumpyReturnPythonizePF()
    NOTE = ("this solution returns a NumPy object, while the problem's "
            'examples compare against plain Python values.')

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_empty_input_guard", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a function body that indexes its argument without an '
                                'empty-input guard'),
          summary="This solution does not handle the empty input the problem's "
                  'own examples include.')
class CodeEmptyInputGuard:
    """Detect and Repair both delegate to `EmptyInputGuardPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.EmptyInputGuardPF()
    NOTE = ("this solution does not handle the empty input the problem's "
            'own examples include.')

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_random_seed_inject", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='public test code is attached and the candidate draws random numbers unseeded'),
          summary="This solution's output depends on unseeded randomness, so it "
                  'cannot match a fixed expected value.')
class CodeRandomSeedInject:
    """Detect and Repair both delegate to `RandomSeedInjectPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.RandomSeedInjectPF()
    NOTE = ("this solution's output depends on unseeded randomness, so it "
            'cannot match a fixed expected value.')

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_helper_syntax_fix", domain=D,
          anchor=Anchor(level="final", evidence="helper",
                        trigger='the submission does not parse as Python'),
          summary='This solution does not parse as python, so it cannot be run at '
                  'all.')
class CodeHelperSyntaxFix:
    """Detect and Repair both delegate to `TeacherSyntaxFixPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.TeacherSyntaxFixPF()
    NOTE = ('this solution does not parse as Python, so it cannot be run at '
            'all.'
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_helper_logic_fix", domain=D,
          anchor=Anchor(level="final", evidence="helper",
                        trigger='public test code is available to verify a rewrite against'),
          summary="This solution does not satisfy the problem's own public "
                  'examples.')
class CodeHelperLogicFix:
    """Detect and Repair both delegate to `TeacherLogicFixPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.TeacherLogicFixPF()
    NOTE = ("this solution does not satisfy the problem's own public "
            'examples.')

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_sandbox_quick_check", domain=D,
          anchor=Anchor(level="final", evidence="executed",
                        trigger='public test code is attached to run the candidate against'),
          summary="Running this solution on the problem's own example fails.")
class CodeSandboxQuickCheck:
    """Detect and Repair both delegate to `SandboxQuickCheckPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.SandboxQuickCheckPF()
    NOTE = "running this solution on the problem's own example fails."

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("code_pick_format", domain=D,
          anchor=Anchor(level="final", evidence="helper",
                        trigger='any committed code answer'),
          summary="This solution's output shape does not match the format the "
                  'problem asks for.')
class CodePickFormat:
    """Detect and Repair both delegate to `PickFormatPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.PickFormatPF()
    NOTE = ("this solution's output shape does not match the format the "
            'problem asks for.')

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)


@pf_skill("decompose_question", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='a spec longer than 200 characters describing three or more '
                                'steps'),
          summary='A spec longer than 200 characters describing three or more '
                  'steps.')
class DecomposeQuestion:
    """Detect and Repair both delegate to `CodeDecomposeQuestionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.CodeDecomposeQuestionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("search_restructuring", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the spec matches one of the known task-category patterns'),
          summary='The spec matches one of the known task-category patterns.')
class SearchRestructuring:
    """Detect and Repair both delegate to `CodeSearchRestructuringPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.CodeSearchRestructuringPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_evidence(ctx, action, arg, iv, self.NOTE)


@pf_skill("entity_confusion_correction", domain=D,
          anchor=Anchor(level="final", evidence="deterministic",
                        trigger='the spec declares a return type'),
          summary='The spec declares a return type.')
class EntityConfusionCorrection:
    """Detect and Repair both delegate to `CodeEntityConfusionCorrectionPF` in implementations.py — the
    implementation that has been in service, moved rather than retyped."""
    _impl = _IMPL.CodeEntityConfusionCorrectionPF()
    NOTE = (''
            )

    def should_activate(self, ctx, action, arg) -> bool:
        return bool(self._impl.should_activate(ctx.raw, action, arg))

    def intervene(self, ctx, action, arg):
        iv = call_base_intervene(type(self._impl), self._impl, ctx.raw, action,
                                 arg, ctx.pf_helper)
        return as_action(ctx, action, arg, iv, self.NOTE)
