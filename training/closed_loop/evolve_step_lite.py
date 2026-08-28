"""Lightweight skill evolution for closed-loop training.

Cheap one-shot skill proposal every N epochs. Skips the full SelfImprovingPipeline
(Phase A/B/D/E/F/G/H). Only does:
  1. Sample failed trajectories from a fixed bootstrap pool
  2. Single prompt to the current student vLLM ckpt (no PF helper API)
  3. Parse 1-K candidate skills (SKILL.md + PF code) from the response
  4. ast.parse + stub-exec compile-check on the PF code
  5. skill_id dedup against the existing library
  6. Write accepted skills to `{library_dir}/generated/{id}/SKILL.md`
     and append PF code to `{library_dir}/dynamic_program_functions.py`

Vs `run_evolve_step` (full pipeline):
  - 1 LLM call instead of ~5 (proposer × clusters + reviewer + analyzer)
  - 0 teacher (gpt-4o) calls — student vLLM only
  - No clustering / no review / no pseudo-gradient / no SFT rebuild

The skills land in disk locations that `SkillRolloutRunner.setup()` already
loads (after the dynamic PF loader patch in training/common/skill_rollout.py),
so the next training rollout picks them up automatically.
"""

from __future__ import annotations

import ast
import gc
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Failure sampling
# ============================================================================

def _sample_failures(
    trajectories_path: str,
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Pick N failed trajectories (exact_match=False, has steps) from a jsonl pool."""
    path = Path(trajectories_path)
    if not path.exists():
        logger.warning("[lite_evolve] trajectory pool not found: %s", path)
        return []

    failures: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            if t.get("exact_match"):
                continue
            if not t.get("steps"):
                continue
            failures.append(t)

    rng = random.Random(seed)
    rng.shuffle(failures)
    sampled = failures[:n]
    logger.info("[lite_evolve] sampled %d failures from %d total", len(sampled), len(failures))
    return sampled


# ============================================================================
# Existing skill IDs (for dedup + prompt grounding)
# ============================================================================

def _list_existing_skill_ids(library_dir: str) -> set:
    """Existing IDs across seed/, generated/, and the static seed PF registry."""
    lib = Path(library_dir)
    ids = set()
    for sub in ("seed", "generated"):
        sub_dir = lib / sub
        if not sub_dir.exists():
            continue
        for child in sub_dir.iterdir():
            if child.is_dir() and (child / "SKILL.md").exists():
                ids.add(child.name)
    # Also the statically-registered PFs (we shouldn't propose collisions).
    try:
        from src.skills_agent.skills.program_functions import _PF_REGISTRY
        ids.update(_PF_REGISTRY.keys())
    except Exception:
        pass
    return ids


# ============================================================================
# Prompt construction (single shot, no clustering)
# ============================================================================

_REFLECTION_SYSTEM = """\
You are designing skills (Program Functions = PFs) for a ReAct web search agent.

Each step the agent issues exactly one Action: SEARCH(q) | READ(doc_id) | FINAL(answer).
Skills observe step context and either rewrite the action or inject context text.

PF Python interface (must inherit ProgramFunction, must be deterministic — no LLM in should_activate):
    @register_pf("<skill_id>")
    class MyPF(ProgramFunction):
        def should_activate(self, step_context, action_type, arg) -> bool: ...
        def intervene(self, step_context, action_type, arg, helper=None) -> Intervention: ...

step_context fields you can use:
  question, step_count, search_count, read_count, has_read, empty_results,
  contradictory_sources, max_steps, action_history, last_search_results_text,
  all_read_contents, thought.

Intervention kwargs (use ONLY these):
  type=InterventionType.MODIFY_ACTION (must set new_action_type, new_action_arg)
       OR InterventionType.INJECT_CONTEXT (must set context_text)
       OR InterventionType.NOOP
  reason="<why triggered>"
  skill_id="<must match @register_pf id>"

Output STRICT format — no commentary outside the skill blocks.
"""

_REFLECTION_USER_TEMPLATE = """\
Existing skill_ids (do NOT propose duplicates):
{existing_ids}

Below are {n} recent failed trajectories. Reflect on shared failure patterns and
propose 1 to {max_k} NEW skills targeting patterns the existing skills don't catch.

== Failure Cases ==
{failure_blocks}

For each new skill, output the two blocks below back-to-back. Separate skills with
a line containing exactly `---SKILL-BREAK---`.

```yaml
---
skill_id: <new_unique_id>
name: <Display Name>
version: 1
priority: 0.7
error_category: <category>
applicable_modes: [all]
system_summary: <one-line summary>
phases:
  pre_final:
    conditions: [always]
    action: verify_<new_unique_id>
---

# <Display Name>

<2-3 sentence description of what failure pattern this addresses>

## Detection Triggers
- <bullet>
- <bullet>

## Avoidance Strategies
- <bullet>
- <bullet>
```

```python
@register_pf("<new_unique_id>")
class SomePF(ProgramFunction):
    def should_activate(self, step_context, action_type, arg):
        # deterministic check on step_context fields
        if action_type == "FINAL" and step_context.get("read_count", 0) == 0:
            return True
        return False

    def intervene(self, step_context, action_type, arg, helper=None):
        return Intervention(
            type=InterventionType.INJECT_CONTEXT,
            context_text="...",
            reason="...",
            skill_id="<new_unique_id>",
        )
```
"""


def _format_failure_block(traj: Dict[str, Any], max_step_arg_chars: int = 80) -> str:
    """Compact failure record for prompt context."""
    q = (traj.get("question", "") or "")[:200]
    gold = traj.get("gold_answers", [])
    final = (traj.get("final_answer", "") or "")[:120]
    f1 = traj.get("f1_score", 0.0)

    step_lines = []
    for s in (traj.get("steps") or [])[:4]:
        a_type = s.get("final_action_type", "")
        a_arg = (s.get("final_action_arg", "") or "")[:max_step_arg_chars]
        step_lines.append(f"  [{s.get('step_index', '?')}] {a_type}({a_arg!r})")
    steps_text = "\n".join(step_lines) if step_lines else "  (no steps)"

    return (
        f"Q: {q}\n"
        f"Gold: {gold}\n"
        f"Steps:\n{steps_text}\n"
        f"Final: {final}  (F1={f1:.2f})"
    )


def _build_prompt(
    failures: List[Dict[str, Any]],
    existing_ids: set,
    max_skills: int,
) -> Tuple[str, str]:
    """Returns (system, user)."""
    blocks = "\n\n".join(_format_failure_block(t) for t in failures)
    user = _REFLECTION_USER_TEMPLATE.format(
        existing_ids=", ".join(sorted(existing_ids)[:50]),
        n=len(failures),
        max_k=max_skills,
        failure_blocks=blocks,
    )
    return _REFLECTION_SYSTEM, user


# ============================================================================
# Response parsing
# ============================================================================

_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n(---\s*\n.*?\n---)", re.DOTALL)
_PY_BLOCK_RE = re.compile(r"```python\s*\n(@register_pf\(.*?\n.*?)```", re.DOTALL)
_SKILL_ID_RE = re.compile(r"^\s*skill_id\s*:\s*([\w\-]+)", re.MULTILINE)
_SKILL_BREAK = "---SKILL-BREAK---"


def _split_skill_sections(response: str) -> List[str]:
    """Split on the explicit `---SKILL-BREAK---` separator. Falls back to a
    single section if the model didn't honor the separator."""
    if _SKILL_BREAK in response:
        return [s.strip() for s in response.split(_SKILL_BREAK) if s.strip()]
    # Fallback: try `### Skill <n>` headers
    sections = re.split(r"\n(?=###\s*Skill\s*\d)", response)
    return [s.strip() for s in sections if s.strip()] or [response]


def _parse_one_skill(section: str) -> Optional[Dict[str, str]]:
    """Pull (md_spec, pf_code, skill_id) out of one section."""
    yml_m = _YAML_BLOCK_RE.search(section)
    py_m = _PY_BLOCK_RE.search(section)
    if not yml_m or not py_m:
        return None

    md_yaml = yml_m.group(1)
    pf_code = py_m.group(1).rstrip()

    # Markdown body sits between the closing ``` of yaml and the python block.
    body_segment = section[yml_m.end():py_m.start()]
    md_body = re.sub(r"```\w*\n?", "", body_segment).strip()
    md_spec = md_yaml + "\n\n" + md_body if md_body else md_yaml + "\n"

    sid_m = _SKILL_ID_RE.search(md_yaml)
    if not sid_m:
        return None

    return {
        "skill_id": sid_m.group(1).strip(),
        "md_spec": md_spec,
        "pf_code": pf_code,
    }


def _parse_skills(response: str, max_k: int) -> List[Dict[str, str]]:
    sections = _split_skill_sections(response)
    out: List[Dict[str, str]] = []
    for s in sections[:max_k]:
        parsed = _parse_one_skill(s)
        if parsed:
            out.append(parsed)
    return out


# ============================================================================
# Compile-check (no real registration; we just validate the code)
# ============================================================================

def _compile_check(pf_code: str) -> Tuple[bool, str]:
    """Validate the PF code: AST parses + execs in a stub namespace.
    No real PF registration happens here — that comes when the dynamic loader
    in skill_rollout.py exec_module's the file at the next rollout setup."""
    try:
        ast.parse(pf_code)
    except SyntaxError as e:
        return False, f"syntax: {e}"

    class _Stub:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, _): return self

    ns: Dict[str, Any] = {
        "ProgramFunction": _Stub,
        "Intervention": _Stub,
        "InterventionType": type("InterventionType", (), {
            "NOOP": "noop", "MODIFY_ACTION": "modify_action", "INJECT_CONTEXT": "inject_context",
        }),
        "register_pf": lambda sid: (lambda cls: cls),  # no-op decorator
        "PFRecord": _Stub,
        # Common modules the model often imports
        "re": __import__("re"),
        "logging": __import__("logging"),
    }
    try:
        exec(compile(pf_code, "<lite_evolve_pf>", "exec"), ns)
    except Exception as e:
        return False, f"exec: {type(e).__name__}: {e}"
    return True, ""


# ============================================================================
# Persistence
# ============================================================================

_PF_FILE_HEADER = '''\
"""
Dynamically generated Program Functions (lite_evolve_step).
Auto-loaded by training.common.skill_rollout._load_dynamic_pfs at rollout setup.
"""

import re
import logging
from typing import Dict, Any, Optional, List

from src.skills_agent.skills.program_functions import (
    ProgramFunction, Intervention, InterventionType,
    register_pf, PFRecord,
)

logger = logging.getLogger(__name__)

# === Dynamically added PFs below ===
'''


def _save_skill(skill: Dict[str, str], library_dir: str, epoch: int) -> None:
    """Write SKILL.md + metadata.json under generated/, append PF to dynamic file."""
    lib = Path(library_dir)
    skill_dir = lib / "generated" / skill["skill_id"]
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text(skill["md_spec"], encoding="utf-8")
    (skill_dir / "metadata.json").write_text(
        json.dumps({
            "skill_id": skill["skill_id"],
            "epoch_added": epoch,
            "source": "lite_evolve",
        }, indent=2),
        encoding="utf-8",
    )

    pf_file = lib / "dynamic_program_functions.py"
    if not pf_file.exists():
        pf_file.write_text(_PF_FILE_HEADER, encoding="utf-8")
    with open(pf_file, "a", encoding="utf-8") as f:
        f.write(f"\n# === Skill: {skill['skill_id']} (lite_evolve epoch {epoch}) ===\n\n")
        f.write(skill["pf_code"])
        f.write("\n")


# ============================================================================
# Generation — vLLM (high GPU util keeps SLURM watchdog happy)
# ============================================================================

def _generate_with_vllm(
    model_path: str,
    system: str,
    user: str,
    temperature: float,
    max_new_tokens: int,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
) -> str:
    """One-shot vLLM generation with explicit teardown.

    Why vLLM over HF transformers here:
      - HF transformers single-batch inference held GPU at <30% util for ~60s
        per evolve trigger; with N=5 iters that's 5 minutes of low-util time
        which can trip the SLURM low-GPU-util watchdog.
      - vLLM's CUDA-graph + paged-attention keeps both GPUs busy during
        generation, mirroring the training-rollout phase.

    Teardown invariant:
      - The LLM object is dereferenced + gc.collect() + torch.cuda.empty_cache()
        before returning, so the next training iter's vLLM (also TP=2) can
        reclaim GPU memory cleanly. EngineCore worker subprocesses spawned by
        TP > 1 die when the LLM is destroyed.

    NOTE: requires the standard training-sbatch env vars to already be set:
      VLLM_WORKER_MULTIPROC_METHOD=spawn   (for TP > 1 spawn safety)
      LD_LIBRARY_PATH=...nvidia/{cublas,cuda_runtime,...}/lib...
      PYTHONPATH=".:src:$PYTHONPATH"
    """
    import gc
    import torch
    from src.skills_agent.eval.model_loader import load_model_vllm
    from vllm import SamplingParams

    model_wrapper, tok_wrapper = load_model_vllm(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=4,  # only one prompt — small batch
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        templated = tok_wrapper.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        templated = f"{system}\n\n{user}"

    sp = SamplingParams(
        temperature=max(temperature, 0.01),  # vLLM requires temp > 0 for sampling
        max_tokens=max_new_tokens,
        top_p=0.95,
    )
    outputs = model_wrapper.llm.generate([templated], sampling_params=sp)
    response = outputs[0].outputs[0].text

    # Explicit teardown so the next training iter's vLLM can claim memory.
    del outputs, model_wrapper, tok_wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


# ============================================================================
# Public entry point
# ============================================================================

def _teacher_score_skill(
    teacher_call,
    md_spec: str,
    pf_code: str,
) -> float:
    """Single-prompt helper review → q_skill in [0, 1]. Returns -1 on parse failure."""
    sys_msg = (
        "You are reviewing a candidate Program-Function (PF) skill for a ReAct web "
        "search agent. Score it on a 5-dim rubric and emit a final Q_skill in [0,1]."
    )
    user_msg = (
        "Skill spec (Markdown + YAML front-matter):\n"
        f"{md_spec}\n\n"
        "Skill implementation (Python PF):\n"
        f"```python\n{pf_code}\n```\n\n"
        "Score these 5 dims (each 0–1) and combine as the weighted sum below:\n"
        "  Q_concept (0.25): is the failure pattern real and worth catching?\n"
        "  Q_trigger (0.20): is should_activate precise (no false positives)?\n"
        "  Q_intervene (0.20): does intervene actually fix the pattern?\n"
        "  Q_exec    (0.20): code looks correct (no obvious bug)?\n"
        "  Q_val     (0.15): generalizes beyond the failure cases shown?\n\n"
        "Output EXACTLY one line:\n"
        "Q_skill: <number in [0,1]>"
    )
    try:
        resp = teacher_call([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ])
        m = re.search(r"Q_skill\s*:\s*([0-9]*\.?[0-9]+)", resp)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception as e:
        logger.warning("[lite_evolve] helper review failed: %s", e)
    return -1.0


def lite_evolve_step(
    student_ckpt_path: str,
    library_dir: str,
    bootstrap_trajectories_path: str = (
        "outputs/bootstrap_rollouts/epoch_0/trajectories/trajectories.jsonl"
    ),
    epoch: int = 0,
    n_failures: int = 20,
    max_new_skills: int = 3,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    seed: int = 42,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 8192,
    enable_compile_check: bool = True,
    enable_teacher_review: bool = False,
    teacher_review_threshold: float = 0.5,
    teacher_call=None,
) -> Dict[str, Any]:
    """One-shot skill proposal driven by the current student vLLM ckpt.

    Args:
        student_ckpt_path: HF-compatible directory of the current merged ckpt.
        library_dir: Experiment-scoped library root (contains seed/, generated/,
                     and dynamic_program_functions.py).
        bootstrap_trajectories_path: jsonl pool to sample failures from.
        epoch: closed_loop iteration index (used in metadata + summary filename).
        n_failures: how many failed trajectories to feed the model.
        max_new_skills: cap on accepted skills per evolve.
        temperature: sampling temperature for the proposal generation.
        max_new_tokens: cap on response length.
        tensor_parallel_size: vLLM TP size (use 2 to match training rollout's
            TP and keep both GPUs hot during generation).
        gpu_memory_utilization: vLLM gpu_memory_utilization. 0.85 leaves slack
            so the next training iter's vLLM can reclaim memory cleanly.
        max_model_len: vLLM max_model_len. 8192 is enough for our prompt
            (≤3K input + 4K output) and matches training rollout setting.

    Returns:
        Summary dict with counts and accepted skill_ids.
    """
    failures = _sample_failures(bootstrap_trajectories_path, n_failures, seed=seed)
    if not failures:
        return {"n_proposed": 0, "n_accepted": 0, "skill_ids": [], "epoch": epoch}

    existing_ids = _list_existing_skill_ids(library_dir)
    system, user = _build_prompt(failures, existing_ids, max_new_skills)

    logger.info(
        "[lite_evolve] epoch=%d, ckpt=%s, %d failures → proposing ≤%d skills",
        epoch, student_ckpt_path, len(failures), max_new_skills,
    )
    response = _generate_with_vllm(
        student_ckpt_path, system, user,
        temperature=temperature, max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    skills = _parse_skills(response, max_new_skills)
    n_compile_fail = 0
    n_dedup_fail = 0
    n_teacher_fail = 0
    teacher_scores: Dict[str, float] = {}
    accepted: List[str] = []
    for sk in skills:
        sid = sk["skill_id"]
        if sid in existing_ids:
            n_dedup_fail += 1
            logger.info("[lite_evolve] skill_id collision: %s — skipping", sid)
            continue
        if enable_compile_check:
            ok, reason = _compile_check(sk["pf_code"])
            if not ok:
                n_compile_fail += 1
                logger.info("[lite_evolve] PF compile fail (%s): %s", sid, reason)
                continue
        if enable_teacher_review and teacher_call is not None:
            q = _teacher_score_skill(teacher_call, sk["md_spec"], sk["pf_code"])
            teacher_scores[sid] = q
            if q < teacher_review_threshold:
                n_teacher_fail += 1
                logger.info("[lite_evolve] PF helper reject (%s): Q=%.2f < %.2f",
                             sid, q, teacher_review_threshold)
                continue
        _save_skill(sk, library_dir, epoch)
        accepted.append(sid)
        existing_ids.add(sid)
        logger.info("[lite_evolve] accepted skill: %s", sid)

    summary = {
        "n_proposed": len(skills),
        "n_accepted": len(accepted),
        "n_rejected_compile": n_compile_fail,
        "n_rejected_dedup": n_dedup_fail,
        "n_rejected_teacher": n_teacher_fail,
        "teacher_scores": teacher_scores,
        "skill_ids": accepted,
        "filters": {
            "compile_check": enable_compile_check,
            "teacher_review": enable_teacher_review,
            "teacher_threshold": teacher_review_threshold,
        },
        "epoch": epoch,
    }
    summary_path = Path(library_dir) / f"lite_evolve_epoch{epoch}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({**summary, "raw_response_excerpt": response[:2000]}, indent=2),
        encoding="utf-8",
    )

    # Also save the raw response for audit (may help debug parsing failures).
    raw_path = Path(library_dir) / f"lite_evolve_epoch{epoch}.raw.txt"
    raw_path.write_text(response, encoding="utf-8")

    logger.info("[lite_evolve epoch %d] %s", epoch, summary)
    return summary
