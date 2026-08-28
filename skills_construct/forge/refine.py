"""Optimise a candidate that was close but did not pay off.

Two failure modes are worth a second pass, and they need opposite fixes:

  precision (screened_out: fires on correct solutions too)
      -> NARROW the trigger. The check is looking at text it should ignore.

  efficacy (refine: fires on wrong solutions, changes nothing)
      -> SHARPEN the verdict. The anchor was right and the evidence was too
         vague to act on. This is the more common failure and the more
         valuable fix: the difference between "check your arithmetic in this
         step" and "the Observation for compute[7*13] is 81, but 7*13 = 91"
         is the difference between no rescue and a rescue.

Everything else is retired rather than refined. A candidate that never fires,
or that breaks a correct solution, is not a tuning problem.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .propose import _CONTRACT, _parse
from .spec import PFSpec, validate_spec, registered_pf_ids
from pathlib import Path

_HASP = Path(__file__).resolve().parents[2]

_REFINE_EFFICACY = '''You proposed this program function. It was screened and then run end to end.
It ANCHORS CORRECTLY — it fires on real failures — but the agent read its verdict and
did not change its answer. The evidence was not decisive.

CURRENT PF
  skill_id:      {skill_id}
  scope:         {scope}
  anchor:        {anchor}
  checker:
{checker}

WHAT IT MEASURED
  fired on {fired_wrong} wrong solutions and {fired_correct} correct ones
  rescued {rescue}, broke {broke}, changed nothing on {no_change}

WHAT IT SAID, AND WHAT HAPPENED ANYWAY
{misses}

Look at those verdicts. The usual cause is that the verdict names a PROBLEM without
naming a VALUE: it says something is wrong but not what the right thing is, so the
agent re-derives the same answer. The fix is to make the checker compute the corrected
value and put it in the verdict.

Return ONE revised PF as a JSON array of one object, same schema as before, with
skill_id "{skill_id}_v{v}". Requirements:
  - the verdict must state the specific wrong value AND the correct value whenever the
    check can compute it; set can_repair accordingly;
  - do not widen the trigger — precision was not the problem here;
  - if this family genuinely cannot produce a decisive verdict, return [] and say
    nothing else. That is a legitimate answer.
'''

_REFINE_PRECISION = '''You proposed this program function. It was screened against a control set of
solutions that were ALREADY CORRECT, and it fired on too many of them. Every one of
those is working reasoning it would have interrupted.

CURRENT PF
  skill_id:      {skill_id}
  scope:         {scope}
  anchor:        {anchor}
  checker:
{checker}

SCREENING
  fires on {fw:.1%} of wrong solutions   (population: fine)
  fires on {fc:.1%} of correct solutions (false-positive floor: too high, limit {maxfc:.0%})
  lift {lift} (needs >= {minlift})

Verdicts it produced on CORRECT solutions — these are the false fires to eliminate:
{false_fires}

Return ONE revised PF as a JSON array of one object, skill_id "{skill_id}_v{v}".
Requirements:
  - narrow the trigger so the false fires above stop, without abandoning the family;
  - recompute rather than pattern-match wherever the check currently guesses;
  - losing some true fires to gain precision is the right trade. A check that fires on
    5% of wrong solutions and 0% of correct ones is more valuable than one that fires
    on 30% of wrong and 8% of correct.
'''


def _version(skill_id: str) -> int:
    m = re.search(r"_v(\d+)$", skill_id)
    return int(m.group(1)) + 1 if m else 2


def _base_id(skill_id: str) -> str:
    return re.sub(r"_v\d+$", "", skill_id)


def build_refine_prompts(specs: List[PFSpec], probe: Dict[str, Dict]) -> List[Dict]:
    """One prompt per refinable candidate."""
    prompts = []
    for s in specs:
        sc = s.screen or {}
        pr = probe.get(s.skill_id) or {}
        checker = "\n".join("    " + l for l in s.checker_src.splitlines()[:60])
        anchor = json.dumps(s.anchor)
        v = _version(s.skill_id)

        if pr and pr.get("fired_wrong", 0) > 0 and pr.get("rescue", 0) == 0:
            misses = "\n".join(
                f"  - it said: {m['verdict'][:260]}\n"
                f"    the agent still answered {m['still_answered']} (correct was {m['gold']})"
                for m in pr.get("misses", [])[:4]) or "  (no samples recorded)"
            prompts.append(dict(
                base=_base_id(s.skill_id), kind="efficacy",
                text=_CONTRACT + "\n\n" + _REFINE_EFFICACY.format(
                    skill_id=_base_id(s.skill_id), scope=s.family_scope, anchor=anchor,
                    checker=checker, fired_wrong=pr.get("fired_wrong", 0),
                    fired_correct=pr.get("fired_correct", 0), rescue=pr.get("rescue", 0),
                    broke=pr.get("broke", 0), no_change=pr.get("no_change", 0),
                    misses=misses, v=v)))
            continue

        if sc.get("verdict") == "reject" and sc.get("fire_correct", 0) > 0:
            from .screen import MAX_FIRE_CORRECT, MIN_LIFT
            ff = "\n".join(f"  - {x['verdict'][:220]}"
                           for x in sc.get("samples", []) if x.get("label") == "correct") \
                 or "  (none recorded)"
            prompts.append(dict(
                base=_base_id(s.skill_id), kind="precision",
                text=_CONTRACT + "\n\n" + _REFINE_PRECISION.format(
                    skill_id=_base_id(s.skill_id), scope=s.family_scope, anchor=anchor,
                    checker=checker, fw=sc.get("fire_wrong", 0), fc=sc.get("fire_correct", 0),
                    lift=sc.get("lift", 0), maxfc=MAX_FIRE_CORRECT, minlift=MIN_LIFT,
                    false_fires=ff, v=v)))
    return prompts


def refine(specs: List[PFSpec], probe: Dict[str, Dict], domain: str, llm, tok,
           max_tokens: int = 4096, temperature: float = 0.7, n: int = 2) -> List[PFSpec]:
    """Generate v2 candidates. `llm`/`tok` are an already-loaded vLLM engine so a
    round can propose and refine in one GPU allocation."""
    from vllm import SamplingParams

    prompts = build_refine_prompts(specs, probe)
    if not prompts:
        return []
    chat = [tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts]
    outs = llm.generate(chat, SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n),
                        use_tqdm=True)

    known = registered_pf_ids(_HASP) | {s.skill_id for s in specs}
    by_id = {s.skill_id: s for s in specs}
    out, seen = [], set()
    for p, o in zip(prompts, outs):
        parent = next((s for s in specs if _base_id(s.skill_id) == p["base"]), None)
        for cand in o.outputs:
            for s in _parse(cand.text, domain, p["base"]):
                if s.skill_id in seen or validate_spec(s, known):
                    continue
                if parent is not None:
                    s.source_uids = parent.source_uids
                    s.rationale = (s.rationale or parent.rationale)
                s.rationale = f"[refined for {p['kind']} from {p['base']}] " + s.rationale
                seen.add(s.skill_id)
                out.append(s)
                break   # one revision per prompt
    print(f"[refine] {len(prompts)} candidates refined -> {len(out)} v2 specs", flush=True)
    return out
