"""Path and cache resolution for HASP.

Nothing in this repository hard-codes a filesystem layout. Everything that
lives outside the repository is named by an environment variable, and the ones
that can be derived from another have a default:

| variable | default | what it points at |
|---|---|---|
| `HASP_CACHE_ROOT`   | `<repo>/.cache`                 | torch/vllm/triton compile caches |
| `HASP_DATA`         | `<repo>/data`                   | mined corpora and eval sets — kept outside a release checkout |
| `HASP_AGENTIC_RL`   | —                               | the upstream evaluation repo (only to locate its results) |
| `HASP_ROLLOUTS_DIR` | `$HASP_AGENTIC_RL/results/v1/base_qwen3_4b_inst_2507_react` | the paired base-model rollouts the math corpora are mined from |
| `HASP_SKILLS_AGENT` | —                               | the upstream agent repo holding web/code episodes |
| `HASP_CODE_EPISODES`| `$HASP_SKILLS_AGENT/outputs/factorial_ablation_code/pf_only_ablation_code` | code episode files |
| `HASP_CODE_DATA`    | `$HASP_SKILLS_AGENT/data/code`  | raw humaneval+/mbpp+ problem files |
| `HASP_MATH_DATA`    | `$HASP_SKILLS_AGENT/data/math`  | raw AIME/AMC/GameOf24 problem files |
| `HASP_WEB_SOURCE`   | —                               | the web source jsonls the validation split is carved from |
| `HASP_WEB_EPISODES` | `$HASP_SKILLS_AGENT/outputs/skill_eval_adaptive_qwen2_5_7b_clean_format/skill_eval_best/three_forced_skills` | web episode files |

Set them in the shell, or once in `~/.hasp_site`, which `scripts/slurm/env.sh`
sources for every job. A missing *required* variable raises with the name to
set, rather than failing later on a path that does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

HASP_ROOT = Path(__file__).resolve().parent

_MISSING = (
    "{var} is not set.\n"
    "  {what}\n"
    "  Set it in your shell or in ~/.hasp_site:  export {var}=/path/to/...\n"
    "  See hasp_paths.py for the full list."
)


def _require(var: str, what: str) -> Path:
    v = os.environ.get(var)
    if not v:
        raise RuntimeError(_MISSING.format(var=var, what=what))
    p = Path(v).expanduser()
    if not p.exists():
        raise RuntimeError(f"{var}={p} does not exist.")
    return p


def _derived(var: str, parent: Path, suffix: str) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else parent / suffix


# --------------------------------------------------------------------------
# compile caches
# --------------------------------------------------------------------------

def data_dir() -> Path:
    """Where mined corpora and eval sets live.

    Defaults to `<repo>/data`, but a release checkout ships code and skills
    without the corpora, so `HASP_DATA` points at wherever they were kept.
    """
    return Path(os.environ.get("HASP_DATA", HASP_ROOT / "data")).expanduser()


def cache_root() -> Path:
    """Root for torch/vllm/triton compile caches.

    Keep this off a quota-limited home directory: vLLM's caches are large.
    """
    return Path(os.environ.get("HASP_CACHE_ROOT", HASP_ROOT / ".cache")).expanduser()


def setup_compile_caches() -> Path:
    """Point every compile cache at a HASP-owned directory.

    Call this **before** importing torch or vllm — the variables are read at
    import time. `setdefault` is deliberate: an explicit value from the job
    script still wins, which is how each SLURM job gets its own cache and two
    concurrent jobs stop racing into one.
    """
    root = cache_root()
    for k, v in {
        "VLLM_CACHE_ROOT": root / "vllm",
        "TRITON_CACHE_DIR": root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "XDG_CACHE_HOME": root,
    }.items():
        os.environ.setdefault(k, str(v))
        os.makedirs(os.environ[k], exist_ok=True)
    return root


# --------------------------------------------------------------------------
# the paired base-model rollouts
# --------------------------------------------------------------------------
# The scorer these rollouts were graded with is NOT imported from there — it is
# inlined in verifiers/reference_em.py, so only the data is external.

def agentic_root() -> Path:
    return _require(
        "HASP_AGENTIC_RL",
        "The upstream evaluation repository, used only to locate its results "
        "directory. Set HASP_ROLLOUTS_DIR instead to point straight at the "
        "rollouts.",
    )


def rollouts_dir() -> Path:
    """The paired base-model rollouts the math corpora are mined from.

    NOTE: the `*_results.json` files in here store `question[:100]`. Never build
    a corpus from those fields — see the data hazards section of the README.
    """
    v = os.environ.get("HASP_ROLLOUTS_DIR")
    if v:
        return Path(v).expanduser()
    return agentic_root() / "results/v1/base_qwen3_4b_inst_2507_react"


# --------------------------------------------------------------------------
# upstream agent repo (web and code episodes)
# --------------------------------------------------------------------------

def skills_agent_root() -> Path:
    return _require(
        "HASP_SKILLS_AGENT",
        "The upstream agent repository holding the web and code episode files.",
    )


def code_episodes_dir() -> Path:
    if os.environ.get("HASP_CODE_EPISODES"):
        return Path(os.environ["HASP_CODE_EPISODES"]).expanduser()
    return skills_agent_root() / "outputs/factorial_ablation_code/pf_only_ablation_code"


def code_data_dir() -> Path:
    if os.environ.get("HASP_CODE_DATA"):
        return Path(os.environ["HASP_CODE_DATA"]).expanduser()
    return skills_agent_root() / "data/code"


def math_data_dir() -> Path:
    """Raw math problem files (AIME24 / AMC23 / GameOf24)."""
    return _derived("HASP_MATH_DATA", skills_agent_root(), "data/math")


def web_source_dir() -> Path:
    """The web source jsonls a validation split is carved out of.

    No default: this one lives in a different upstream repository than the
    episode files, so there is nothing to derive it from.
    """
    return _require("HASP_WEB_SOURCE",
                    "The web source jsonls the validation split is carved from.")


def web_episodes_dir() -> Path:
    if os.environ.get("HASP_WEB_EPISODES"):
        return Path(os.environ["HASP_WEB_EPISODES"]).expanduser()
    return (
        skills_agent_root()
        / "outputs/skill_eval_adaptive_qwen2_5_7b_clean_format/skill_eval_best/three_forced_skills"
    )
