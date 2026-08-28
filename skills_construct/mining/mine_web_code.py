"""Mine web-search and coding wrong cases (from Skills_Agent's skills_analysis/errors)
into families, and measure two candidate deterministic anchors.

Records: data/error_cases_src/{web,code}/<dataset>__wrong.jsonl
  web : question, gold_answers, pred_answer, trace[{i, action, query, thought, obs}], signals, pf
  code: question (spec with >>> examples / asserts), pred_answer (code), first_failure, pass_rate, trace

Families
  web : evidence_present   gold string appears in some observation but pred != gold
        early_finish       <= 2 actions, never READ
        no_evidence        gold never retrieved (query/retrieval failure)
        giveup             pred is empty / "unknown" / "cannot determine"
  code: exception class parsed from first_failure (AssertionError = wrong output,
        TypeError/NameError/AttributeError/ImportError = API misuse, SyntaxError,
        timeout) ; plus whether the spec carries runnable examples

Candidate anchors (measured on the wrong set; precision needs a correct set,
see TODO at the bottom of the report)
  web  answer_grounding_check : the final answer must occur (fuzzy) in at least
                                one observation; else "unsupported answer"
  code spec_example_check     : execute the submitted code on the >>> / assert
                                examples in the spec (sandbox); a failing
                                example is provable evidence at a known line
"""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
_HASP = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(_HASP))
from anchor.sandbox import run_python  # noqa: E402
SRC = _HASP / "data" / "error_cases_src"

# ── web ──────────────────────────────────────────────────────────────────
def _norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
def _contains(hay, needle):
    h, n = " ".join(_norm(hay)), " ".join(_norm(needle))
    return bool(n) and n in h
_GIVEUP = re.compile(r"^\s*$|unknown|cannot (?:be )?determine|not (?:enough|sufficient)|unable to|no information|i don'?t know", re.I)

def web_family(r):
    obs = " ".join(str(s.get("obs", "")) for s in r["trace"])
    gold_in_obs = any(_contains(obs, g) for g in r["gold_answers"])
    n_read = sum(1 for s in r["trace"] if s["action"] == "READ")
    if _GIVEUP.search(str(r["pred_answer"])[:80]): return "giveup", gold_in_obs
    if gold_in_obs: return "evidence_present", gold_in_obs
    if len(r["trace"]) <= 2 and n_read == 0: return "early_finish", gold_in_obs
    return "no_evidence", gold_in_obs

def answer_grounding(r):
    """anchor: is the committed answer supported by any observation?"""
    obs = " ".join(str(s.get("obs", "")) for s in r["trace"])
    pred = str(r["pred_answer"]).strip()
    if not pred or len(_norm(pred)) > 12:       # long free-text answers: take the last clause
        pred = re.split(r"[.;]\s", pred)[-1] if pred else pred
    return _contains(obs, pred)

# ── code ─────────────────────────────────────────────────────────────────
_EXC = re.compile(r"([A-Za-z_]*(?:Error|Exception|Timeout|timeout))\b")
def code_family(r):
    ff = r.get("first_failure")
    s = json.dumps(ff) if not isinstance(ff, str) else ff
    if "timeout" in s.lower(): return "timeout"
    m = _EXC.findall(s)
    ex = m[-1] if m else "unknown"
    if ex == "AssertionError": return "wrong_output"
    if ex in ("TypeError", "NameError", "AttributeError", "ImportError", "ModuleNotFoundError", "KeyError", "ValueError", "IndexError"): return f"runtime:{ex}"
    if ex == "SyntaxError": return "syntax"
    return f"other:{ex}"

_DOCTEST = re.compile(r">>>\s*(.+)\n\s*([^\n>]+)")
_ASSERT = re.compile(r"^\s*assert\s+(.+)$", re.M)
def spec_examples(spec: str):
    ex = [(c.strip(), o.strip()) for c, o in _DOCTEST.findall(spec)]
    asserts = [a.strip() for a in _ASSERT.findall(spec)]
    return ex, asserts

def spec_example_check(r):
    """Run the submitted code on the spec's own examples. Returns
    ('no_examples'|'pass'|'fail'|'error', detail)."""
    ex, asserts = spec_examples(r["question"])
    code = str(r["pred_answer"])
    if not ex and not asserts: return "no_examples", ""
    checks = []
    for call, out in ex[:6]:
        checks.append(f"try:\n    _r = {call}\n    _ok = repr(_r) == {out!r} or str(_r) == {out!r}\n    print('EX', {call!r}, 'PASS' if _ok else 'FAIL got ' + repr(_r)[:80])\nexcept Exception as e:\n    print('EX', {call!r}, 'ERROR', type(e).__name__, str(e)[:80])")
    for a in asserts[:6]:
        checks.append(f"try:\n    assert {a}\n    print('AS', {a!r}, 'PASS')\nexcept AssertionError:\n    print('AS', {a!r}, 'FAIL')\nexcept Exception as e:\n    print('AS', {a!r}, 'ERROR', type(e).__name__, str(e)[:80])")
    prog = code + "\n\n" + "\n".join(checks)
    ok, out = run_python(prog, timeout_s=8)
    if not ok: return "error", out[-200:]
    lines = [l for l in out.splitlines() if l.startswith(("EX", "AS"))]
    fails = [l for l in lines if " FAIL" in l or " ERROR" in l]
    if not lines: return "error", out[-200:]
    return ("fail", fails[0][:200]) if fails else ("pass", "")

def main():
    out = ["# Web & code wrong-case review (Skills_Agent eval, skills_analysis/errors)\n"]
    # web
    fam = Counter(); fam_ds = defaultdict(Counter); ex = defaultdict(list); ground = Counter(); pfs = Counter()
    for f in sorted((SRC / "web").glob("*__wrong.jsonl")):
        for r in map(json.loads, f.open()):
            name, gio = web_family(r); fam[name] += 1; fam_ds[r["dataset"]][name] += 1
            g = answer_grounding(r); ground[(name, "grounded" if g else "UNSUPPORTED")] += 1
            for p in (r.get("pf") or {}).get("activated", []): pfs[p.get("skill_id", p) if isinstance(p, dict) else p] += 1
            if len(ex[name]) < 3: ex[name].append(r)
    n = sum(fam.values())
    out += [f"## Web ({n} wrong rollouts; 3 datasets x 3 settings)\n", "| family | n | share | answer UNSUPPORTED by any observation |", "|---|---|---|---|"]
    for k, v in fam.most_common():
        out.append(f"| {k} | {v} | {v/n:.0%} | {ground[(k,'UNSUPPORTED')]} ({ground[(k,'UNSUPPORTED')]/max(1,v):.0%}) |")
    out.append(f"\nPFs activated on these wrong rollouts (skills settings): {dict(pfs.most_common(8))}\n")
    for k in fam:
        out.append(f"\n### web family `{k}`")
        for r in ex[k]:
            acts = " → ".join(f"{s['action']}[{str(s.get('query'))[:40]}]" for s in r["trace"][:6])
            out += [f"- **{r['dataset']}/{r['sample_id']}** ({r['setting']}) gold={r['gold_answers'][:2]} pred=`{str(r['pred_answer'])[:80]}`", f"  - actions: {acts}", f"  - grounded: {answer_grounding(r)}"]
    # code
    cfam = Counter(); cex = defaultdict(list); spec = Counter(); spec_by = defaultdict(Counter)
    for f in sorted((SRC / "code").glob("*__wrong.jsonl")):
        for r in map(json.loads, f.open()):
            k = code_family(r); cfam[k] += 1
            v, d = spec_example_check(r); spec[v] += 1; spec_by[k][v] += 1
            if len(cex[k]) < 2: cex[k].append((r, v, d))
    nc = sum(cfam.values())
    out += [f"\n## Code ({nc} wrong rollouts; humaneval+ / mbpp+ / bigcodebench)\n", "| family | n | spec_example_check: fail / pass / no_examples / error |", "|---|---|---|"]
    for k, v in cfam.most_common():
        sb = spec_by[k]; out.append(f"| {k} | {v} | {sb['fail']} / {sb['pass']} / {sb['no_examples']} / {sb['error']} |")
    out.append(f"\n**spec_example_check over all code wrong cases:** {dict(spec)} — 'fail' = provable evidence from the spec's own examples.\n")
    for k in list(cfam)[:8]:
        out.append(f"\n### code family `{k}`")
        for r, v, d in cex[k]:
            ff = r.get("first_failure"); ffs = (json.dumps(ff) if not isinstance(ff, str) else ff)[:160]
            out += [f"- **{r['dataset']}/{r['sample_id']}** ({r['setting']}) first_failure=`{ffs}`", f"  - spec_example_check={v} {d[:120]}", f"  - code head: `{str(r['pred_answer'])[:140].replace(chr(10),' ⏎ ')}`"]
    out.append("\n## TODO\n- precision of both anchors needs a CORRECT set (outputs/skill_eval_* episodes); this report measures the wrong set only.")
    (_HASP / "WRONG_CASES_WEB_CODE.md").write_text("\n".join(out))
    print("web families:", dict(fam)); print("web answer UNSUPPORTED:", {k: v for k, v in ground.items() if k[1] == 'UNSUPPORTED'})
    print("code families:", dict(cfam.most_common())); print("code spec_example_check:", dict(spec))
    print("-> WRONG_CASES_WEB_CODE.md")

if __name__ == "__main__":
    main()
