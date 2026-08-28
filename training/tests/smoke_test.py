"""Smoke test — runs the full data + reward pipeline without any GPU.

Run via:
    python -m training.tests.smoke_test

Validates:
  1. Signal registration + aggregator end-to-end
  2. Objective-A data builder on a synthetic trajectory
  3. Objective-B data builder on a synthetic candidate + review
  4. GRPO reward fn with TRL-style kwargs
  5. Candidate/review loader against a real self_improving output (if present)
  6. YAML parseability of every configs/**/*.yaml

Exits 0 iff all stages pass. Any failure prints the traceback and exits 1.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402



def _pass(name: str):
    print(f"  [OK]   {name}")


def _fail(name: str, exc: BaseException) -> bool:
    print(f"  [FAIL] {name}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return False


def make_synthetic_trajectory():
    from training.signals.trajectory import (
        EpisodeTrajectory, StepRecord, PFActivationRecord,
    )
    steps = [
        StepRecord(
            step_index=0,
            proposed_action_type="FINAL",
            proposed_action_arg="Paris",
            proposed_reasoning="Guess.",
            final_action_type="SEARCH",
            final_action_arg="capital of France",
            was_modified=True,
            pf_activations=[PFActivationRecord(
                pf_id="premature_final", activated=True,
                intervention_type="modify_action",
                original_action="FINAL", original_arg="Paris",
                modified_action="SEARCH", modified_arg="capital of France",
            )],
            step_context_snapshot={"step_count": 0, "has_read": False, "search_count": 0},
        ),
        StepRecord(
            step_index=1,
            proposed_action_type="READ",
            proposed_action_arg="0",
            proposed_reasoning="Check top result.",
            final_action_type="READ",
            final_action_arg="0",
            was_modified=False,
            pf_activations=[PFActivationRecord(pf_id="answer_completeness", activated=False)],
            step_context_snapshot={"step_count": 1, "has_read": False, "search_count": 1},
        ),
        StepRecord(
            step_index=2,
            proposed_action_type="FINAL",
            proposed_action_arg="Paris",
            proposed_reasoning="Confirmed.",
            final_action_type="FINAL",
            final_action_arg="Paris",
            was_modified=False,
            pf_activations=[PFActivationRecord(pf_id="answer_completeness", activated=False)],
            step_context_snapshot={"step_count": 2, "has_read": True, "search_count": 1},
        ),
    ]
    return EpisodeTrajectory(
        sample_id="syn_0",
        question="What is the capital of France?",
        gold_answers=["Paris"],
        dataset_name="synthetic",
        skills_enabled=True,
        selected_pf_ids=["premature_final"],
        steps=steps,
        final_answer="Paris",
        exact_match=True,
        f1_score=1.0,
    )


def make_synthetic_candidate_and_review():
    from skills_construct.candidate import CandidateSkill
    from skills_construct.candidate import ReviewResult
    cand = CandidateSkill(
        skill_id="syn_skill_0",
        name="synthetic_skill",
        category="reasoning_guard",
        target_failure_pattern="premature_final",
        md_spec="# Synthetic\nTriggers when FINAL before READ.\n",
        pf_code="class SyntheticPF(ProgramFunction):\n    def should_activate(...): pass\n",
        raw_response="<fake>",
    )
    review = ReviewResult(
        skill_id="syn_skill_0",
        q_concept=0.8, q_trigger=0.7, q_intervene=0.7, q_exec=0.9, q_val=0.6,
        q_skill=0.74,
        decision="accept",
        feedback="looks good",
    )
    return cand, review


# ----------------------------------------------------------------------

def test_signals_and_aggregator() -> bool:
    try:
        from training.signals import SignalAggregator, SignalRegistry
        from training.signals.aggregator import AggregatorConfig

        all_signals = SignalRegistry.list_signals()
        assert len(all_signals) >= 14, f"too few signals registered: {all_signals}"

        traj = make_synthetic_trajectory()
        agg = SignalAggregator(AggregatorConfig(enabled=["s4.downstream", "s4.local"]))
        step_score = agg.score_step(traj, traj.steps[0])
        ep_score = agg.score_episode(traj)
        assert isinstance(step_score, float)
        assert isinstance(ep_score, float)

        bd = agg.breakdown(traj, traj.steps[0])
        assert "s4.local" in bd
        _pass("signals_and_aggregator")
        return True
    except Exception as e:
        return _fail("signals_and_aggregator", e)


def test_use_pfs_builder() -> bool:
    try:
        from training.data import UsePFsBuilder
        from training.data.use_pfs_builder import UsePFsBuilderConfig
        from training.signals import SignalAggregator
        from training.signals.aggregator import AggregatorConfig

        traj = make_synthetic_trajectory()
        with tempfile.TemporaryDirectory() as tmp:
            agg = SignalAggregator(AggregatorConfig(enabled=["s4.local", "s4.downstream"]))
            b = UsePFsBuilder(
                UsePFsBuilderConfig(
                    output_dir=tmp,
                    enabled_signals=["s4.local", "s4.downstream"],
                    threshold=0.1,
                    formats=["sft", "dpo", "prompt"],
                ),
                agg,
            )
            outs = b.build([traj])
            for k in ("sft", "dpo", "prompt"):
                p = outs[k]
                assert Path(p).exists(), f"missing {k}"
                rows = [json.loads(l) for l in open(p)]
                assert rows, f"{k} empty"

            # prompt jsonl MUST expose columns needed by GRPO reward fn
            prompt_rows = [json.loads(l) for l in open(outs["prompt"])]
            for r in prompt_rows:
                for col in ("messages", "sample_id", "step_index", "question", "step_context"):
                    assert col in r, f"prompt row missing {col}"
        _pass("use_pfs_builder")
        return True
    except Exception as e:
        return _fail("use_pfs_builder", e)


def test_evolve_builder() -> bool:
    try:
        from training.data import EvolveBuilder
        from training.data.evolve_builder import EvolveBuilderConfig
        from training.signals import SignalAggregator
        from training.signals.aggregator import AggregatorConfig

        cand, review = make_synthetic_candidate_and_review()
        with tempfile.TemporaryDirectory() as tmp:
            agg = SignalAggregator(AggregatorConfig(enabled=["s4.downstream"]))
            b = EvolveBuilder(
                EvolveBuilderConfig(
                    output_dir=tmp,
                    q_skill_threshold=0.5,
                    formats=["sft", "prompt"],
                ),
                agg,
            )
            outs = b.build([cand], [review])
            for k in ("sft", "prompt"):
                p = outs[k]
                rows = [json.loads(l) for l in open(p)]
                assert rows, f"{k} empty"
        _pass("evolve_builder")
        return True
    except Exception as e:
        return _fail("evolve_builder", e)


def test_grpo_reward_fn() -> bool:
    try:
        from training.grpo.reward import build_reward_fn

        fn = build_reward_fn(
            enabled_signals=["s1.tp", "s3.syntactic", "s4.local"],
            mode="action",
        )
        completions = [
            "Action: SEARCH(capital of France)",
            "Action: FINAL(Paris)",
            "not an action",
        ]
        rewards = fn(
            completions,
            sample_id=["s0", "s0", "s1"],
            step_index=[0, 0, 0],
            step_context=[
                {"step_count": 0, "has_read": False},
                {"step_count": 0, "has_read": False},
                {"step_count": 0, "has_read": False},
            ],
            question=["q", "q", "q"],
        )
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        _pass("grpo_reward_fn")
        return True
    except Exception as e:
        return _fail("grpo_reward_fn", e)


def test_candidates_loader_real() -> bool:
    """Only runs if a real self_improving output is present."""
    try:
        si_dir = ROOT / "outputs" / "self_improving"
        if not si_dir.exists():
            print("  [SKIP] candidates_loader_real (no self_improving output)")
            return True
        from training.sft.train import _load_candidates_and_reviews
        cands, revs = _load_candidates_and_reviews(str(si_dir))
        print(f"  [INFO] loaded {len(cands)} candidates, {len(revs)} reviews from real data")
        # Non-empty if epochs actually produced data
        if cands and revs:
            assert hasattr(cands[0], "md_spec")
            assert hasattr(revs[0], "q_skill")
        _pass("candidates_loader_real")
        return True
    except Exception as e:
        return _fail("candidates_loader_real", e)


def test_trajectory_loader_real() -> bool:
    try:
        si_dir = ROOT / "outputs" / "self_improving"
        if not si_dir.exists():
            print("  [SKIP] trajectory_loader_real (no self_improving output)")
            return True
        from training.sft.train import _load_trajectories
        trajs = _load_trajectories(str(si_dir))
        print(f"  [INFO] loaded {len(trajs)} trajectories from real data")
        _pass("trajectory_loader_real")
        return True
    except Exception as e:
        return _fail("trajectory_loader_real", e)


def test_yamls_parse() -> bool:
    try:
        bad = []
        for f in (ROOT / "training" / "configs").rglob("*.yaml"):
            try:
                yaml.safe_load(open(f))
            except Exception as e:
                bad.append((str(f), str(e)))
        assert not bad, f"bad yamls: {bad}"
        _pass("yamls_parse")
        return True
    except Exception as e:
        return _fail("yamls_parse", e)


# ----------------------------------------------------------------------

def main() -> int:
    os.chdir(ROOT)
    print("=== training/ smoke test ===")
    results = [
        test_signals_and_aggregator(),
        test_use_pfs_builder(),
        test_evolve_builder(),
        test_grpo_reward_fn(),
        test_candidates_loader_real(),
        test_trajectory_loader_real(),
        test_yamls_parse(),
    ]
    ok = all(results)
    print("=== %s ===" % ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
