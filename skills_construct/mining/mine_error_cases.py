"""Mine the error population that PF-select currently fails to fix.

Data source: the paired base-model eval in the Agentic_RL repo
(results/v1/base_qwen3_4b_inst_2507_react). pf_select reused the skills_off
Turn-1s verbatim, so rollout i under skills_off and rollout i under pf_select
are the SAME trajectory — every difference is attributable to the PF layer.

Mechanism recap (measured over all 23,680 paired rollouts):
  - the rescue / broke split of the run being mined.
  - 99.9% of rescues are "Turn-1 stalled at Action: ... -> the PF turn
    intervened there and the rollout continued to finish[]". That channel
    works; it is not where further gains are.
  - Only 3 rollouts with a *committed wrong* answer were flipped to correct.

So the population where PF-library quality and anchoring can still add value is:

    COMMITTED-WRONG: Turn-1 committed an answer (finish[]/\boxed{}/Answer:)
                     and that answer is wrong, and pf_select did NOT fix it.

For each such rollout we record everything needed for (a) error taxonomy,
(b) PF-blindspot analysis (which PFs fired and did nothing), and (c) the
anchor-recoverability signal:

    gold_in_reasoning: the gold answer literally appears in the Turn-1
    reasoning text before the (wrong) final answer. These are "the model had
    it and lost it" cases — the prime target for an anchored intervention.

Output: <HASP>/data/error_cases/<dataset>.jsonl  (one record per rollout,
capped per question) + a printed summary table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_HASP = Path(__file__).resolve().parents[2]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

from hasp_paths import rollouts_dir  # noqa: E402
# Resolved lazily: only the corpus-building stage reads the raw rollouts,
# so scoring an existing corpus needs no upstream checkout at all.

# Same commitment test used in the mechanism analysis (mirrors pre-E5 v2's
# Turn-1 filter): the rollout produced a recognisable final-answer marker.
_COMMIT_RE = re.compile(
    r"finish\s*\[|\\boxed\s*\{|(?:^|\n)\s*(?:Final answer|Answer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# PF intervention markers in the pf_select response tail.
_PF_FINAL_RE = re.compile(r"\[PF Final Answer\]", re.IGNORECASE)
_SYS_FB_RE = re.compile(r"\[System Feedback\]", re.IGNORECASE)
_FB_SKILL_RE = re.compile(r"\[([a-z0-9_]+)\]", re.IGNORECASE)

DATASETS = ["aime24", "amc23", "math500", "gsm8k", "olympiadbench"]


# The reference EM scorer, so correctness here is identical to the published
# pass@1 numbers. See verifiers/reference_em.py.
from verifiers.reference_em import em_match_multi as _em_match_multi  # noqa: E402


def _gold_variants(gold: str) -> list[str]:
    """Literal spellings of the gold answer to search for inside reasoning.

    Deliberately conservative: only exact-ish spellings (raw, stripped, with
    \\boxed, int/float collapse). A hit means the correct value was literally
    written down during reasoning.
    """
    g = str(gold).strip()
    out = {g}
    try:
        f = float(g)
        if f == int(f):
            out.add(str(int(f)))
        out.add(str(f))
    except ValueError:
        pass
    return [v for v in out if v]


def _gold_in_reasoning(response: str, wrong_pred: str, gold: str) -> bool:
    """True iff a literal spelling of gold appears in the reasoning body.

    Guards against trivial hits: very short numeric golds (1-2 digits) match
    accidentally all the time (step numbers, indices), so those require a
    math-ish context: adjacent to '=', '\\boxed', 'answer', or end-of-line.
    """
    variants = _gold_variants(gold)
    body = response or ""
    for v in variants:
        if len(v) >= 3:
            if v in body:
                return True
        else:
            for m in re.finditer(re.escape(v), body):
                s, e = m.start(), m.end()
                # digit-boundary check
                if s > 0 and (body[s - 1].isdigit() or body[s - 1] == "."):
                    continue
                if e < len(body) and (body[e].isdigit() or body[e] == "."):
                    continue
                ctx = body[max(0, s - 30):s].lower()
                if "=" in ctx or "boxed" in ctx or "answer" in ctx or ctx.rstrip().endswith(("is", ":")):
                    return True
    return False


def _fired_pfs(sel_response: str, off_response: str) -> tuple[str, list[str]]:
    """Classify the PF intervention visible in the pf_select tail and pull
    the skill ids named inside a [System Feedback] block."""
    tail = sel_response[len(off_response):] if sel_response.startswith(off_response[:200]) else sel_response
    if _PF_FINAL_RE.search(tail):
        return "B", []
    m = _SYS_FB_RE.search(tail)
    if m:
        fb_block = tail[m.end():].split("Revised answer:")[0]
        ids = [s.lower() for s in _FB_SKILL_RE.findall(fb_block)
               if s.lower() not in ("system feedback",)]
        return "C", sorted(set(ids))
    return "A", []


def mine(dataset: str, per_question_cap: int = 3) -> tuple[list[dict], Counter]:
    off = json.loads((rollouts_dir() / "skills_off" / f"{dataset}_results.json").read_text())["results"]
    sel = json.loads((rollouts_dir() / "pf_select" / f"{dataset}_results.json").read_text())["results"]

    records: list[dict] = []
    stats: Counter = Counter()
    for qo, qs in zip(off, sel):
        assert qo["id"] == qs["id"]
        gold = str(qo["gold"])
        kept_for_q = 0
        seen_preds: set[str] = set()
        for i, (po, ps, ro, rs) in enumerate(zip(
            qo["all_predictions"], qs["all_predictions"],
            qo["all_responses"], qs["all_responses"],
        )):
            stats["rollouts"] += 1
            committed = bool(_COMMIT_RE.search(ro))
            ok_off = _em_match_multi(po, gold)
            ok_sel = _em_match_multi(ps, gold)
            if not committed:
                stats["uncommitted"] += 1
                continue
            if ok_off:
                stats["committed_correct"] += 1
                continue
            stats["committed_wrong"] += 1
            if ok_sel:
                stats["committed_wrong_but_sel_fixed"] += 1
                continue  # already rescued — not our target
            case, fb_ids = _fired_pfs(rs, ro)
            stats[f"cw_case_{case}"] += 1
            gir = _gold_in_reasoning(ro, po, gold)
            if gir:
                stats["cw_gold_in_reasoning"] += 1
            # cap: a few *distinct-wrong-answer* rollouts per question
            if kept_for_q >= per_question_cap or po in seen_preds:
                continue
            seen_preds.add(po)
            kept_for_q += 1
            records.append({
                "dataset": dataset,
                "qid": qo["id"],
                "rollout_idx": i,
                "question": qo["question"],
                "gold": gold,
                "wrong_pred": po,
                "sel_pred": ps,
                "pf_case": case,
                "pf_feedback_ids": fb_ids,
                "gold_in_reasoning": gir,
                "response_len": len(ro),
                "orig_response": ro,
            })
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HASP / "data" / "error_cases"))
    ap.add_argument("--cap", type=int, default=3, help="max kept rollouts per question")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grand = Counter()
    rows = []
    for ds in args.datasets.split(","):
        recs, stats = mine(ds, args.cap)
        with open(out_dir / f"{ds}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        grand.update(stats)
        cw = stats["committed_wrong"]
        rows.append((ds, stats["rollouts"], stats["uncommitted"],
                     stats["committed_correct"], cw,
                     stats["committed_wrong_but_sel_fixed"],
                     stats["cw_case_A"], stats["cw_case_B"], stats["cw_case_C"],
                     stats["cw_gold_in_reasoning"], len(recs)))
        print(f"[mine] {ds}: {len(recs)} error cases -> {out_dir / (ds + '.jsonl')}")

    print("\n=== committed-wrong population (per rollout) ===")
    hdr = ["dataset", "rollouts", "uncommit", "cw_ok", "cw_wrong", "sel_fixed",
           "case_A", "case_B", "case_C", "gold_in_reason", "kept"]
    print("  ".join(f"{h:>13}" for h in hdr))
    for r in rows:
        print("  ".join(f"{v:>13}" for v in r))
    t = grand
    print(f"\nTOTAL committed-wrong: {t['committed_wrong']}  "
          f"(sel fixed {t['committed_wrong_but_sel_fixed']}, "
          f"gold literally in reasoning: {t['cw_gold_in_reasoning']} "
          f"= {t['cw_gold_in_reasoning'] / max(1, t['committed_wrong']):.1%})")


if __name__ == "__main__":
    main()
