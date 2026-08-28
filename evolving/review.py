"""Score the skills already in the library, with the four credit signals.

An evolve cycle without this only ever adds. It distils new skills from the
current checkpoint's failures and appends them, and nothing ever asks whether
the skills already there are earning their place. That is how a library grows
into something nobody can account for.

The four signals answer four different questions about one intervention, and a
skill can fail any of them independently:

| family | asks | a skill fails it by |
|---|---|---|
| **timing** (S1) | did it fire on the steps that were actually risky? | firing everywhere, or never on the steps that mattered |
| **modality** (S2) | did it intervene at the right point of the ReAct cycle? | speaking after the action it should have preceded |
| **correctness** (S3) | was what it said well-formed, on-topic, and right for the domain? | producing a verdict the policy cannot act on |
| **outcome** (S4) | did the rollout end better, net of what the intervention cost? | helping locally and hurting downstream |

Aggregating them into one number hides exactly the distinction that makes them
useful, so `review_library` keeps all four per skill and only sums when asked.
A skill that scores well on timing and badly on outcome is a different problem
from one that never fires at all, and the fix is different too.

Signals need trajectories with PF activation records, so this runs on the
rollouts the cycle's own evaluation just produced — no extra generation.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HASP = Path(__file__).resolve().parents[1]
if str(_HASP) not in sys.path:
    sys.path.insert(0, str(_HASP))

#: The 15 sub-signals, four families. Enabled by default; narrow the list to
#: score on fewer.
DEFAULT_SUBS = [
    "s1.tp", "s1.fp", "s1.fn", "s1.phase",                    # timing
    "s2.pre_action", "s2.post_obs", "s2.pre_reasoning", "s2.post_action",  # modality
    "s3.syntactic", "s3.semantic", "s3.domain",               # correctness
    "s4.local", "s4.downstream", "s4.cost", "s4.side_effect",  # outcome
]

FAMILIES = ("timing", "modality", "correctness", "outcome")


def _aggregator(subs: Optional[Sequence[str]] = None):
    from training.signals.aggregator import AggregatorConfig, SignalAggregator
    import training.signals  # noqa: F401  — importing registers the sub-signals
    return SignalAggregator(AggregatorConfig(enabled=list(subs or DEFAULT_SUBS),
                                             mode="coarse"))


def as_trajectory(sample_id: str, question: str, response: str, exact_match: bool,
                  pf_records: Sequence[Any], selected_ids: Sequence[str] = ()):
    """Shape one evaluated rollout the way the signals expect.

    `EpisodeTrajectory` / `StepRecord` / `PFActivationRecord` are the schema the
    signals read, so they are imported rather than mirrored. The distinction
    they need is between the action the agent *proposed* and the one it ended up
    taking: without it a successful rewrite is indistinguishable from a false
    positive.
    """
    from training.signals.trajectory import (
        EpisodeTrajectory, PFActivationRecord, StepRecord,
    )
    steps: List[StepRecord] = []
    for i, r in enumerate(pf_records or []):
        activated = bool(getattr(r, "activated", False))
        acts = []
        if activated:
            acts.append(PFActivationRecord(
                pf_id=getattr(r, "skill_id", ""),
                activated=True,
                intervention_type=getattr(r, "intervention_type", "") or "noop",
                reason=getattr(r, "reason", "") or "",
                modified_action=getattr(r, "new_action_type", None),
                modified_arg=getattr(r, "new_action_arg", None),
                injected_text=getattr(r, "context_text", None),
            ))
        proposed = getattr(r, "action_type", "FINAL") or "FINAL"
        final = getattr(r, "new_action_type", None) or proposed
        steps.append(StepRecord(
            step_index=getattr(r, "step_index", i),
            proposed_action_type=proposed,
            proposed_action_arg=getattr(r, "arg", "") or "",
            final_action_type=final,
            final_action_arg=getattr(r, "new_action_arg", "") or "",
            was_modified=bool(getattr(r, "new_action_arg", None)),
            pf_activations=acts,
            step_context_snapshot={"question": question, "response": response},
        ))
    traj = EpisodeTrajectory(sample_id=sample_id, question=question,
                             selected_pf_ids=list(selected_ids), steps=steps,
                             final_answer=response, exact_match=bool(exact_match))
    try:
        traj.compute_stats()
    except Exception:
        pass
    return traj


def review_library(trajectories: Sequence[Any],
                   subs: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """-> {skill_id: {timing, modality, correctness, outcome, n_fires}}.

    Each family is averaged over the steps where that skill actually fired, so
    a skill is judged on its own interventions rather than diluted by rollouts
    it stayed out of.
    """
    agg = _aggregator(subs)
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for traj in trajectories:
        for step in getattr(traj, "steps", []):
            fired = [a.pf_id for a in step.pf_activations if a.activated and a.pf_id]
            if not fired:
                continue
            try:
                coarse = agg.breakdown_coarse(traj, step, {})
            except Exception:
                continue
            for sid in fired:
                for fam in FAMILIES:
                    if fam in coarse:
                        acc[sid][fam].append(float(coarse[fam]))
                acc[sid]["_fires"].append(1.0)

    out: Dict[str, Dict[str, float]] = {}
    for sid, fams in acc.items():
        row = {fam: (sum(v) / len(v) if (v := fams.get(fam)) else 0.0)
               for fam in FAMILIES}
        row["n_fires"] = float(len(fams.get("_fires", [])))
        out[sid] = row
    return out


def flag_for_retirement(review: Dict[str, Dict[str, float]], *,
                        min_fires: int = 3,
                        outcome_floor: float = 0.0,
                        timing_floor: float = 0.0) -> List[Dict[str, Any]]:
    """Skills whose own interventions are not paying off, and why.

    Returns a list rather than deleting anything: retiring a skill is a
    decision about the library, and the signals are evidence for it, not the
    decision itself. A skill that fired too rarely to judge is left alone —
    silence is the expected behaviour for most skills on most rollouts.
    """
    flagged = []
    for sid, r in sorted(review.items()):
        if r.get("n_fires", 0) < min_fires:
            continue
        reasons = []
        if r["outcome"] < outcome_floor:
            reasons.append(f"outcome {r['outcome']:+.2f} — the rollouts it touched "
                           "ended no better, net of what it cost")
        if r["timing"] < timing_floor:
            reasons.append(f"timing {r['timing']:+.2f} — it is firing on steps that "
                           "were not the risky ones")
        if r["correctness"] < 0:
            reasons.append(f"correctness {r['correctness']:+.2f} — what it said was "
                           "malformed or off-topic often enough to hurt")
        if reasons:
            flagged.append(dict(skill_id=sid, n_fires=int(r["n_fires"]),
                                reasons=reasons, **{f: r[f] for f in FAMILIES}))
    return flagged


def render(review: Dict[str, Dict[str, float]], limit: int = 12) -> str:
    """One line per skill, worst outcome first — the order to read them in."""
    if not review:
        return "  (no skill fired on this cycle's rollouts)"
    rows = sorted(review.items(), key=lambda kv: kv[1]["outcome"])[:limit]
    head = (f"  {'skill':<34}{'timing':>9}{'modality':>10}"
            f"{'correct':>9}{'outcome':>9}{'fires':>7}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for sid, r in rows:
        lines.append(f"  {sid:<34}{r['timing']:>+9.2f}{r['modality']:>+10.2f}"
                     f"{r['correctness']:>+9.2f}{r['outcome']:>+9.2f}"
                     f"{int(r['n_fires']):>7}")
    return "\n".join(lines)
