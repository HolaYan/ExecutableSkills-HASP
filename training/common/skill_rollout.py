"""Skill-aware rollout shim used by RS (E2/E5) and Distill (E3/E6).

Both routes now run full ReAct episodes through the production inference
framework (`SkillAgentRunner`) so PF selection + skill library + tool
environment behave identically to evaluation.

Pipeline produced for each input sample:
    question, gold_answers, sample_id
        → SkillAgentRunner.run_episode (multi-step ReAct, PFs fire live)
        → Episode (trajectory, final answer, pf_records)

`flatten_to_per_step()` converts a batch of Episodes into the per-step
{messages, generation, sample_id, step_index, gold_answers} rows that
PFVerifier and SFT data-builders downstream expect.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dynamic PF loader — exec-imports a per-experiment dynamic_program_functions.py
# so generated PFs actually populate the global registry.
#
# Why this exists: library_manager.py (and lite_evolve_step) APPEND PF code to
# `{library_dir}/dynamic_program_functions.py`, but until this loader nothing
# imported the file at runtime — so generated PFs were dead code. SKILL.md
# descriptions still reached the agent via prompt injection, but the
# deterministic should_activate / intervene side never fired.
# ----------------------------------------------------------------------

from skills_layout import resolve as _resolve  # noqa: E402


def _load_dynamic_pfs(library_dir: Path) -> None:
    """Exec-import `{library_dir}/dynamic_program_functions.py` so its
    @register_pf decorators register their PFs. Idempotent no-op if missing."""
    pf_file = library_dir / "dynamic_program_functions.py"
    if not pf_file.exists():
        return
    import importlib.util
    mod_name = f"_dynamic_pfs_{hash(str(pf_file)) & 0xFFFFFF:06x}"
    spec = importlib.util.spec_from_file_location(mod_name, str(pf_file))
    if spec is None or spec.loader is None:
        logger.warning("Could not spec dynamic PFs from %s", pf_file)
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        logger.info("Loaded dynamic PFs from %s", pf_file)
    except Exception as e:
        logger.warning("Failed to exec dynamic PFs from %s: %s", pf_file, e)


# ----------------------------------------------------------------------
# Sample loading
# ----------------------------------------------------------------------

def _scan_gold_table(raw_data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Build a `{question: {gold, eval_test_code, entry_point}}` map from raw
    benchmark jsonl files. Used to enrich prompt rows that don't carry these
    fields themselves.

    The original return shape was ``{question: [gold_answers]}`` — we now
    return a dict so code-domain prompts can also recover the per-row
    ``eval_test_code`` (EvalPlus / BCB combined-test driver) and
    ``entry_point`` for sandbox scoring.
    """
    table: Dict[str, Dict[str, Any]] = {}
    root = Path(raw_data_dir)
    if not root.is_dir():
        logger.warning("Raw data dir %s not found — gold lookup will be empty", root)
        return table
    for f in sorted(root.glob("*.jsonl")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    obj = json.loads(line)
                    q = obj.get("question")
                    if not q or q in table:
                        continue
                    a = obj.get("answer", obj.get("golden_answers"))
                    gold: List[str] = []
                    if a is not None:
                        gold = [str(x) for x in (a if isinstance(a, list) else [a])]
                    table[q] = {
                        "gold": gold,
                        "eval_test_code": obj.get("eval_test_code") or "",
                        "entry_point": obj.get("entry_point") or (
                            (obj.get("metadata") or {}).get("entry_point") or ""
                        ),
                    }
        except Exception as e:
            logger.warning("Skipping %s during gold scan: %s", f, e)
    logger.info("Gold table: %d unique questions from %s", len(table), root)
    return table


def load_training_samples(
    prompts_path: str,
    raw_data_dir: str = "data/web_search",
) -> List[Dict[str, Any]]:
    """Build {question, gold_answers, sample_id} list from a prompts jsonl.

    Dedupes per sample_id (objA_prompts.jsonl has one row per step; we only
    want one full-episode rollout per question). Gold answers are looked up
    against the raw benchmark files.
    """
    gold_table = _scan_gold_table(raw_data_dir)
    seen = set()
    samples: List[Dict[str, Any]] = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid in seen:
                continue
            seen.add(sid)
            q = row.get("question") or ""
            if not q:
                continue
            entry = gold_table.get(q) or {}
            gold = (
                row.get("gold_answers")
                or entry.get("gold")
                or []
            )
            samples.append({
                "sample_id": str(sid),
                "question": q,
                "gold_answers": gold,
                # Empty string for non-code domains; populated for HumanEval+/
                # MBPP+/BCB so flatten_to_per_step can run sandbox pass@1.
                "eval_test_code": row.get("eval_test_code") or entry.get("eval_test_code") or "",
                "entry_point": row.get("entry_point") or entry.get("entry_point") or "",
            })
    logger.info("Loaded %d unique training samples from %s", len(samples), prompts_path)
    return samples


# ----------------------------------------------------------------------
# Skill-rollout runner
# ----------------------------------------------------------------------

class SkillRolloutRunner:
    """Wraps SkillAgentRunner setup for training-time rollouts.

    Supports two backends:
      - `backend="vllm"`: local student model loaded via vLLM.
      - `backend="api"`: teacher model served through an API wrapper
        (matches `SkillAgentRunner.use_api`).
    """

    def __init__(
        self,
        model_path: str,
        skill_library_dir: str,
        backend: str = "vllm",
        api_provider: Optional[str] = None,
        api_model: Optional[str] = None,
        api_key: Optional[str] = None,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        max_num_seqs: int = 128,
        max_steps: int = 10,
        max_search_calls: int = 8,
        max_read_calls: int = 8,
        timeout_seconds: int = 300,
        pf_top_k: int = 10,
        enable_pf_selection: bool = True,
        pf_selection_model: Optional[str] = None,
        parallel_episodes: int = 32,    # bumped from 16: more concurrent episodes hides API latency
        domain: str = "web_search",
    ):
        self.model_path = model_path
        self.skill_library_dir = skill_library_dir
        self.backend = backend
        self.api_provider = api_provider
        self.api_model = api_model
        self.api_key = api_key
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_steps = max_steps
        self.max_search_calls = max_search_calls
        self.max_read_calls = max_read_calls
        self.timeout_seconds = timeout_seconds
        self.pf_top_k = pf_top_k
        self.enable_pf_selection = enable_pf_selection
        self.pf_selection_model = pf_selection_model or ""
        self.parallel_episodes = parallel_episodes
        self.domain = domain

        self._runner = None

    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self._runner is not None:
            return
        from src.skills_agent.eval.agent_runner import RunnerConfig
        from src.skills_agent.eval.tools import ToolEnvironment
        from src.skills_agent.agent.skill_agent_runner import SkillAgentRunner
        from src.skills_agent.agent.config import SkillAgentConfig
        from src.skills_agent.skills.skill import SkillLibrary

        serp = os.environ.get("SERPAPI_API_KEY", "")
        oai = os.environ.get("OPENAI_API_KEY", "")

        # Match ablation arms (ablation/runner.py:_create_runner):
        #   backend="vllm" → PF_NO_TEACHER:  PF/OT run in code-only fallback,
        #                    PF selector heuristic, difficulty gate heuristic,
        #                    zero LLM API calls. Preserves all skill structure
        #                    + signal generation but no per-step gpt-4o cost.
        #   backend="api"  → PF_WITH_TEACHER: full PF helper wired into PF.intervene,
        #                    PF selector, etc. Used for distill PF helper trajectory.
        _use_teacher = self.backend == "api"
        skill_config = SkillAgentConfig(
            skill_library_path=str(self.skill_library_dir),
            skill_source_format="markdown",
            skills_enabled=True,
            pf_only_mode=True,
            enable_program_functions=True,
            # Disable the vote-based verify_* handler cascade: each FINAL
            # would otherwise fire 5-7 handlers × ~2s gpt-4o call = 10-15s
            # of GPU-idle per step. PF program functions (cheap modify_action)
            # still fire. Matches configs/agent_eval.yaml eval default.
            enable_skill_handlers=False,
            handler_vote_threshold=4,
            enable_pf_selection=self.enable_pf_selection,
            pf_top_k=self.pf_top_k,
            # PF helper attached only on api backend (distill helper rollouts).
            # vllm backend (student rollout) gets None → PF.intervene falls back
            # to its code-only path, no PF helper.generate calls.
            teacher_api_provider=self.api_provider if _use_teacher else None,
            teacher_api_model=self.api_model if _use_teacher else None,
            teacher_api_key=self.api_key if _use_teacher else None,
            # PF selector: same PF helper when api-backend; heuristic when vllm.
            # Setting these to None explicitly avoids the prior "Failed to create
            # model: Unsupported provider: None" warning that came from passing
            # pf_selection_model="gpt-4o" without a provider.
            pf_selection_model=self.api_model if _use_teacher else None,
            pf_selection_provider=self.api_provider if _use_teacher else None,
            pf_selection_model_name=self.api_model if _use_teacher else None,
            pf_selection_api_key=self.api_key if _use_teacher else None,
            # Difficulty gating uses heuristic mode (no PF helper API call).
            difficulty_model=None,
        )

        runner_config = RunnerConfig(
            max_steps=self.max_steps,
            max_search_calls=self.max_search_calls,
            max_read_calls=self.max_read_calls,
            timeout_seconds=self.timeout_seconds,
            model_type="base",
            serpapi_key=serp,
            openai_key=oai,
            domain=self.domain,
        )
        env = ToolEnvironment(serpapi_key=serp, openai_key=oai)

        # skill_library_dir may be either a leaf dir of skills, or a root with
        # seed/ and generated/ subdirs — merge them if the latter.
        lib_root = Path(self.skill_library_dir)
        seed_sub = lib_root / "seed"
        gen_sub = lib_root / "generated"

        # Load dynamic PFs from `{lib_root}/dynamic_program_functions.py` if it
        # exists. Without this exec_module call, PF code appended by lite_evolve
        # / library_manager was dead — only SKILL.md descriptions reached the
        # agent through prompt injection. Side-effect: each @register_pf in the
        # file populates the global PF registry (program_functions._PF_REGISTRY).
        for _mod in _resolve(lib_root).pf_modules or [lib_root / "dynamic_program_functions.py"]:
            _load_dynamic_pfs(_mod.parent)

        if seed_sub.is_dir():
            skill_library = SkillLibrary.load_from_directory(str(seed_sub))
            if gen_sub.is_dir() and any(gen_sub.iterdir()):
                generated = SkillLibrary.load_from_directory(str(gen_sub))
                for sid, skill in generated._skills.items():
                    skill_library._skills[sid] = skill
        else:
            # `skills/<domain>` is a virtual path: on disk the library is split
            # into skills/{textual,executable}/<domain>. resolve() maps either
            # form — and a flat run-scoped copy — onto the dirs to scan.
            _dirs = _resolve(lib_root).skill_dirs or [lib_root]
            skill_library = SkillLibrary.load_from_directory(str(_dirs[0]))
            for _d in _dirs[1:]:
                for _s in SkillLibrary.load_from_directory(str(_d)).get_all():
                    skill_library._skills[_s.skill_id] = _s

        if self.backend == "vllm":
            from src.skills_agent.eval.model_loader import load_model_vllm
            model, tokenizer = load_model_vllm(
                self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
            )
        elif self.backend == "api":
            # IMPORTANT: use the APIModelWrapper from model_loader, not
            # self_improving.pipeline. agent_runner detects API path via
            # `isinstance(model, model_loader.APIModelWrapper)`; the two
            # classes are distinct and isinstance would return False with
            # the wrong one, causing tokenizer=None to hit the HF path.
            from src.skills_agent.eval.model_loader import load_model_api
            if not (self.api_provider and self.api_model and self.api_key):
                raise ValueError("api backend requires api_provider, api_model, api_key")
            model, _ = load_model_api(
                self.api_provider, self.api_model, self.api_key,
                max_concurrent=self.parallel_episodes,
            )
            tokenizer = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self._runner = SkillAgentRunner(
            model=model,
            tokenizer=tokenizer,
            config=runner_config,
            env=env,
            skill_library=skill_library,
            skill_config=skill_config,
        )

    # ------------------------------------------------------------------

    def run(
        self,
        samples: List[Dict[str, Any]],
        mode: str = "clean",
    ) -> List[Any]:
        """Run full ReAct episodes for each sample. Returns List[Episode]."""
        self.setup()
        batch = [
            {
                "sample_id": s.get("sample_id", str(i)),
                "question": s["question"],
                "gold_answers": s.get("gold_answers", []),
            }
            for i, s in enumerate(samples)
        ]
        parallel_n = max(1, self.parallel_episodes) if (
            self._runner.use_vllm or self._runner.use_api
        ) else 1
        if parallel_n > 1:
            episodes = self._runner.run_batch(
                batch, mode=mode,
                parallel_episodes=parallel_n, verbose=False,
            )
        else:
            episodes = [
                self._runner.run_episode(
                    question=s["question"],
                    gold_answers=s["gold_answers"],
                    sample_id=s["sample_id"],
                    mode=mode,
                )
                for s in batch
            ]
        return episodes


# ----------------------------------------------------------------------
# Episode → per-step rows
# ----------------------------------------------------------------------

def flatten_to_per_step(
    episodes: List[Any],
    samples: List[Dict[str, Any]],
    domain: str = "web_search",
) -> List[Dict[str, Any]]:
    """Flatten each Episode into per-step (messages, generation) rows.

    Output row schema (matches what PFVerifier and SFT builders consume):
        {
          "sample_id": str,
          "step_index": int,
          "messages": [...prefix...],          # chat up to (but not including) this step's assistant turn
          "generation": str,                   # model's assistant output at this step
          "gold_answers": [...],
          "question": str,
          "final_answer": str,                 # episode-level
          "exact_match": bool,                 # episode-level answer-EM vs gold
        }

    `domain="math"` selects MathAnswerEvaluator (numeric+LaTeX equivalence,
    so "7" matches "7.0") instead of string-EM AnswerEvaluator.
    `domain="code"` runs the candidate against the per-sample
    `eval_test_code` driver in CodeSandbox — actual pass@1, not string-EM.
    """
    from src.skills_agent.eval.metrics import (
        AnswerEvaluator, CodeAnswerEvaluator, MathAnswerEvaluator,
    )
    _ev = AnswerEvaluator()

    rows: List[Dict[str, Any]] = []
    for sample, episode in zip(samples, episodes):
        final_answer = ""
        if episode.final:
            final_answer = episode.final.get("answer", "") or ""
        gold = sample.get("gold_answers", [])
        if domain == "math":
            em = bool(MathAnswerEvaluator.exact_match(final_answer, gold))
        elif domain == "code":
            etc = sample.get("eval_test_code") or ""
            ep = sample.get("entry_point") or ""
            if etc:
                try:
                    em = bool(CodeAnswerEvaluator.exact_match(
                        final_answer, [], eval_test_code=etc, entry_point=ep,
                    ))
                except Exception:
                    em = False
            else:
                em = bool(_ev.exact_match(final_answer, gold))
        else:
            em = bool(_ev.exact_match(final_answer, gold))

        trajectory = getattr(episode, "_trajectory", []) or []
        for step in trajectory:
            rows.append({
                "sample_id": str(sample.get("sample_id", "")),
                "step_index": int(step.get("step", 0)),
                "question": sample.get("question", ""),
                "gold_answers": sample.get("gold_answers", []),
                "messages": step.get("messages", []),
                "generation": step.get("response", ""),
                "final_answer": final_answer,
                "exact_match": em,
                # code domain: carried so the RS filter can re-run the spec's
                # own examples on the episode's answer (SpecExampleVerifier)
                "entry_point": sample.get("entry_point", ""),
                "public_test_code": sample.get("public_test_code", "") or sample.get("eval_test_code", ""),
            })
    return rows
