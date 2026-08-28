"""code — pure checker functions.

Extracted from the old `evidence_pfs.py`, which mixed three things: these
functions, a PF base class, and the registrations. Registration now lives in
`skills.py` (one declaration per skill) and this file holds only what a checker
needs to be: a function from (context, answer) to a verdict string, with no
knowledge of the PF runtime.

Imported by `skills.py` and by the offline harnesses in `anchor/` — the same
implementation the measured numbers came from.
"""


from __future__ import annotations


import re


import sys


from pathlib import Path


from typing import Any, Dict, Optional


_HASP = Path(__file__).resolve().parents[3]


if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))


try:
    from skills_agent.skills.program_functions import Intervention, InterventionType, ProgramFunction, register_pf
except ImportError:  # pragma: no cover
    from src.skills_agent.skills.program_functions import Intervention, InterventionType, ProgramFunction, register_pf


from anchor.sandbox import run_python  # noqa: E402


_DOCTEST = re.compile(r">>>\s*(.+)\n\s*([^\n>]+)")


_ASSERT = re.compile(r"^\s*assert\s+(.+)$", re.M)


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _code_of(arg: str) -> str:
    m = _FENCE.findall(str(arg))
    return m[-1] if m else str(arg)


_DEF = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)


_PREFIX_START = re.compile(r"^(?:from\s+\S+\s+import|import\s+\S+|def\s|class\s|@)", re.M)


def compose_program(spec: str, code: str, entry_point: str = "") -> str:
    """HumanEval-style outputs are often the function BODY only; the evaluator
    appends them to the prompt's signature. Mirror that: if the submitted code
    defines no entry function, prepend the spec's code prefix (imports +
    signature + docstring)."""
    defs = set(_DEF.findall(code))
    # the spec's import lines are part of the evaluation harness either way
    # (HumanEval: `from typing import List` precedes the signature); a full
    # solution that omits them would NameError on `List` — that is not a bug
    # in the solution.
    imports = "\n".join(l for l in spec.splitlines() if re.match(r"^(?:from\s+\S+\s+import|import\s+\S+)", l))
    if (entry_point and entry_point in defs) or (not entry_point and defs):
        return (imports + "\n" + code) if imports else code
    m = _PREFIX_START.search(spec)
    prefix = spec[m.start():] if m else ""
    if not prefix.strip():
        return code
    # completion body: indent it under the signature if it is not indented
    body = code if code.startswith((" ", "\t")) else "\n".join("    " + l if l.strip() else l for l in code.splitlines())
    return prefix.rstrip() + "\n" + body


def spec_example_evidence(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    spec = str(ctx.get("question") or "")
    code = compose_program(spec, _code_of(answer), str(ctx.get("entry_point") or ctx.get("func_name") or ""))
    if not code.strip():
        return None
    checks = []
    # 1) runtime-provided public tests (Skills_Agent code env)
    pub = ctx.get("public_test_code")
    if pub:
        checks.append(f"try:\n    exec({str(pub)!r}, globals())\n    print('PUB PASS')\nexcept AssertionError as e:\n    print('PUB FAIL', str(e)[:100])\nexcept Exception as e:\n    print('PUB ERROR', type(e).__name__, str(e)[:100])")
    # 2) examples embedded in the spec
    for call, out in _DOCTEST.findall(spec)[:6]:
        call, out = call.strip(), out.strip()
        # bigcodebench doctests are often illustrative (random values, DataFrame
        # dumps): 52 failing / 14 passing fires = no signal. Trust only expected
        # outputs that are plain literals (number / bool / quoted string / list of those).
        if not re.fullmatch(r"-?\d+(\.\d+)?|True|False|None|'[^']*'|\"[^\"]*\"|\[[^\[\]]{0,80}\]|\([^()]{0,80}\)", out):
            continue
        checks.append(f"try:\n    _r = {call}\n    _ok = repr(_r) == {out!r} or str(_r) == {out!r}\n    print('EX', {call!r}, 'PASS' if _ok else 'FAIL got ' + repr(_r)[:80] + ' expected ' + {out!r})\nexcept Exception as e:\n    print('EX', {call!r}, 'ERROR', type(e).__name__, str(e)[:80])")
    for a in _ASSERT.findall(spec)[:6]:
        a = a.strip()
        checks.append(f"try:\n    assert {a}\n    print('AS', {a!r}, 'PASS')\nexcept AssertionError:\n    print('AS', {a!r}, 'FAIL')\nexcept Exception as e:\n    print('AS', {a!r}, 'ERROR', type(e).__name__, str(e)[:80])")
    if not checks:
        return None
    ok, out = run_python(code + "\n\n" + "\n".join(checks), timeout_s=8)
    if not ok:
        tail = out.strip().splitlines()[-1][:160] if out.strip() else "timeout"
        return f"running the submitted code together with the problem's examples raises: {tail}"
    bad = [l for l in out.splitlines() if l.startswith(("EX", "AS", "PUB")) and (" FAIL" in l or " ERROR" in l)]
    if not bad:
        return None
    return "the submitted code fails the problem's own example(s): " + "; ".join(b[:140] for b in bad[:3])


_SIG = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.M)


def _arity(params: str) -> int:
    ps = [p.strip() for p in params.split(",") if p.strip() and not p.strip().startswith("*")]
    return len([p for p in ps if p != "self"])


def signature_conformance(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Anchor = the submission's def line for the entry point."""
    spec = str(ctx.get("question") or ""); code = _code_of(answer)
    entry = str(ctx.get("entry_point") or ctx.get("func_name") or "")
    spec_sigs = {m.group(1): _arity(m.group(2)) for m in _SIG.finditer(spec)}
    code_sigs = {m.group(1): _arity(m.group(2)) for m in _SIG.finditer(code)}
    if not entry:
        entry = list(spec_sigs)[-1] if spec_sigs else ""   # specs list helpers first, the target last
    if not entry or not code_sigs:
        return None                         # body-only completion: composed later, nothing to compare
    if entry not in code_sigs:
        verdict = (f"the specification's entry point is `{entry}(...)` but the submission defines "
                   f"{sorted(code_sigs)} and no `{entry}`; the grader calls `{entry}` by name")
        # The grader calls `entry` by name, so as submitted this scores zero
        # whatever the body computes: renaming cannot make it worse. Only when
        # there is exactly one function to rename and its arity matches, so the
        # rename is determined rather than chosen.
        want = spec_sigs.get(entry)
        cands = [n for n, a in code_sigs.items() if want is None or a == want]
        fix = None
        if len(cands) == 1:
            renamed = re.sub(rf"\b{re.escape(cands[0])}\b(?=\s*\()", entry, code)
            try:
                import ast; ast.parse(renamed)
                fix = renamed if renamed != code else None
            except SyntaxError:
                fix = None                  # a rename that breaks the parse is not a repair
        return dict(verdict=verdict, fix=fix)
    if entry in spec_sigs and code_sigs[entry] != spec_sigs[entry]:
        return (f"`{entry}` takes {spec_sigs[entry]} argument(s) in the specification but {code_sigs[entry]} "
                f"in the submission; calls with the specified arity will fail")
    return None


_EDGE = {"list": "[]", "List": "[]", "str": "''", "int": "0", "float": "0.0", "dict": "{}", "Dict": "{}",
         "tuple": "()", "Tuple": "()", "set": "set()", "bool": "False"}


def edge_input_probe(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Call the entry point on typed edge inputs ([] / '' / 0); flag only an EXCEPTION
    (a wrong value on an edge case is not decidable without the spec)."""
    spec = str(ctx.get("question") or "")
    # Opt-in only: even with the spec-constraint guard the probe fires on some of
    # passing solutions more often than failing ones (inverted), so it never runs unless a
    # caller sets step_context["enable_edge_probe"] = True.
    if not ctx.get("enable_edge_probe"):
        return None
    # Passing solutions raise on edge inputs MORE often than failing
    # ones (19/877 vs 3/323) because the spec itself demands it ("must be
    # positive", "non-empty"). Stay silent whenever the spec constrains the input.
    if re.search(r"\b(positive|non-?empty|at least one|non-?zero|greater than zero|not empty|assume)\b", spec, re.I):
        return None
    code = compose_program(spec, _code_of(answer), str(ctx.get("entry_point") or ctx.get("func_name") or ""))
    entry = str(ctx.get("entry_point") or ctx.get("func_name") or "")
    sigs = list(_SIG.finditer(spec))
    m = next((m for m in sigs if entry and m.group(1) == entry), (sigs[-1] if sigs and not entry else None))
    if m is None:
        return None
    entry = m.group(1)
    args = []
    for p in [p.strip() for p in m.group(2).split(",") if p.strip()]:
        if p.startswith("*") or "=" in p:
            break
        ann = p.split(":")[1].strip() if ":" in p else ""
        base = re.sub(r"\[.*", "", ann)
        if base in _EDGE:
            args.append(_EDGE[base]); continue
        if ann:
            return None                 # an explicit unknown type: no safe edge input
        # unannotated (which is most model output): infer the shape from the
        # spec's own example call -- a doctest arg opening with [ is a list,
        # a quote a string, a digit a number
        m2 = re.search(rf">>> {re.escape(entry)}\(\s*(.)", spec)
        ch = m2.group(1) if m2 else ""
        guess = {"[": "[]", "'": "''", '"': "''", "(": "()"}.get(ch, "0" if ch.isdigit() else None)
        if guess is None:
            return None
        args.append(guess)
    if not args:
        return None
    call = f"{entry}({', '.join(args)})"
    prog = code + f"\n\ntry:\n    {call}\n    print('EDGE OK')\nexcept Exception as e:\n    print('EDGE EXC', type(e).__name__, str(e)[:100])"
    ok, out = run_python(prog, timeout_s=8)
    if not ok or "EDGE EXC" not in out:
        return None
    line = next(l for l in out.splitlines() if l.startswith("EDGE EXC"))
    return f"the entry point raises on an edge input: `{call}` → {line[9:].strip()}; handle the empty/zero case explicitly"


# Any of the ways a docstring states the contract -- "Raises ValueError for
# b == 0", "should raise a TypeError when...", "Raises: ValueError ...". The
# old pattern only matched the literal phrase "should raise exception for:",
# giving the checker zero recall on its one target defect in the audit.
_EXC_SPEC = re.compile(
    r"(?:should\s+)?raises?\s+(?:the\s+|an?\s+)?"
    r"((?:[A-Z][A-Za-z]*(?:Error|Exception)[^\n]{0,80})|exception for:\s*[^\n]+)",
    re.I)


_EXC_NAMES = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception))\b")


def exception_contract(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Static, deterministic: the spec says 'should raise X for …' but the code
    never raises X. Static, so it is cheap and precise by construction."""
    spec = str(ctx.get("question") or ""); m = _EXC_SPEC.search(spec)
    if not m:
        return None
    wanted = set(_EXC_NAMES.findall(m.group(1)))
    if not wanted:
        return None
    code = _code_of(answer)
    raised = set(re.findall(r"raise\s+([A-Z][A-Za-z]*(?:Error|Exception))", code))
    if wanted & raised:
        return None
    w = sorted(wanted)[0]
    return (f"the specification requires the function to raise {w} ({m.group(1).strip()[:120]}), but the "
            f"submission never raises {w}; the grader asserts that exception is raised")


_IMPORT = re.compile(r"^\s*(?:import\s+([\w\.]+)(?:\s+as\s+(\w+))?|from\s+([\w\.]+)\s+import\s+([\w\*, ]+))", re.M)


_ATTR_CHAIN = re.compile(r"\b([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)\s*\(")


def api_attribute_probe(ctx: Dict[str, Any], answer: str) -> Optional[str]:
    """Executed, deterministic: every `alias.attr…(` call chain whose root is an
    imported module is resolved with getattr in a subprocess. Evidence only when
    the module IMPORTS FINE but lacks the attribute — an import failure is an
    environment fact (network stubs, missing package), not a wrong API, and is
    never reported. Dotted imports (`import urllib.request`) are imported in
    full. (First version flagged ftplib.FTP / urllib.request.urlopen: the
    sandbox's socket stub broke the import and `import urllib` alone does not
    load submodules — 51 'fires' that were all artifacts.)"""
    code = _code_of(answer)
    aliases = {}                       # alias -> (module to import, attribute path inside it)
    for m in _IMPORT.finditer(code):
        if m.group(1):                 # import a.b.c [as x]
            full, alias = m.group(1), m.group(2)
            if alias:
                aliases[alias] = (full, [])
            else:                      # `import urllib.request` binds `urllib`; resolve via the full name
                aliases[full.split(".")[0]] = (full, [])
        else:                          # from pkg import name [as x]
            for name in re.split(r"\s*,\s*", m.group(4)):
                nm = [x.strip() for x in name.split(" as ")]
                if nm[0] and nm[0] != "*":
                    aliases[nm[-1]] = (m.group(3), [nm[0]])
    chains = set()
    for m in _ATTR_CHAIN.finditer(code):
        root, rest = m.group(1), m.group(2)
        if root in aliases:
            full, pre = aliases[root]
            parts = pre + rest.lstrip(".").split(".")
            # `import urllib.request` + `urllib.request.urlopen`: the chain already names the submodule
            if not pre and full.count(".") and rest.lstrip(".").startswith(full.split(".", 1)[1]):
                parts = rest.lstrip(".").split(".")[full.count("."):]
            chains.add((full, tuple(parts), root + rest))
    if not chains:
        return None
    # BLAS-backed libraries spawn a thread pool at import; under the
    # sandbox's RLIMIT_NPROC that kills the probe before it probes anything
    # (the audit saw these OpenBLAS aborts misread as code errors).
    probe = ["import os",
             "for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):",
             "    os.environ[_v] = '1'",
             "import importlib", "bad = []"]
    for full, parts, shown in sorted(chains)[:25]:
        probe.append(
            f"try:\n    obj = importlib.import_module({full!r})\n"
            f"except Exception:\n    obj = None            # import failure is not evidence\n"
            f"if obj is not None:\n    try:\n        for p in {list(parts)!r}:\n            obj = getattr(obj, p)\n"
            f"    except AttributeError as e:\n"
            f"        import difflib\n"
            f"        sugg = difflib.get_close_matches(p, dir(obj), n=2, cutoff=0.75)\n"
            f"        bad.append(({shown!r}, str(e)[:80], p, sugg))\n"
            f"    except Exception:\n        pass")
    probe.append("print('APIBAD', repr(bad))")
    ok, out = run_python("\n".join(probe), timeout_s=10)
    m = re.search(r"APIBAD (\[.*\])", out or "")
    if not ok or not m or m.group(1) == "[]":
        return None
    try:
        bad = eval(m.group(1), {"__builtins__": {}}, {})
    except Exception:
        return None
    rec = bad[0]
    shown, err = rec[0], rec[1]
    # the probe also asked dir(obj) for close matches; exactly one at 0.75
    # similarity is a determined repair target ("arry" -> "array"), several or
    # none is a judgement the verdict states instead
    bad_attr = rec[2] if len(rec) > 2 else ""
    sugg = rec[3] if len(rec) > 3 else []
    out = dict(verdict=(f"the code calls `{shown}(…)` but that attribute does not exist "
                        f"({err}); use the correct API"))
    if bad_attr and len(sugg) == 1:
        out["bad_attr"], out["suggestion"] = bad_attr, sugg[0]
        out["verdict"] += f" — did you mean `{sugg[0]}`?"
    return out
