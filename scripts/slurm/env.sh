# Shared environment for every HASP SLURM job. Sourced by scripts/slurm/*.sbatch.
#
# Nothing here is tied to a particular cluster. Every value can be overridden
# from the environment, or set once in a site file that is never committed:
#
#   HASP_ROOT        repository root          default: $SLURM_SUBMIT_DIR, else $PWD
#   HASP_CONDA_SH    path to conda.sh         default: auto-detected from `conda`
#   HASP_CONDA_ENV   environment to activate  default: hasp
#   HASP_CACHE_ROOT  compile-cache root       default: $HASP_ROOT/.cache
#   HASP_SITE_FILE   extra file to source     default: $HOME/.hasp_site
#   HF_HOME          HuggingFace cache        default: left untouched
#   HF_HUB_OFFLINE   1 on clusters without egress          default: 0
#
# The harnesses additionally need the upstream corpora. Those are resolved by
# hasp_paths.py, which documents every variable and raises with the name to set
# when one is missing:
#
#   HASP_AGENTIC_RL    upstream evaluation repo (rollout data only)
#   HASP_SKILLS_AGENT  upstream agent repo (web and code episode files)
#
# Partition, account and GPU type are deliberately NOT in the #SBATCH headers,
# so the scripts stay portable. Pass them at submit time:
#
#   sbatch --partition=<partition> --account=<account> scripts/slurm/anchor.sbatch
#
# or put the defaults in ~/.hasp_site and use a wrapper of your own.

set -uo pipefail

HASP_ROOT=${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
HASP_SITE_FILE=${HASP_SITE_FILE:-$HOME/.hasp_site}
[ -f "$HASP_SITE_FILE" ] && . "$HASP_SITE_FILE"

# --- conda ------------------------------------------------------------------
if [ -z "${HASP_CONDA_SH:-}" ] && command -v conda >/dev/null 2>&1; then
    HASP_CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ -n "${HASP_CONDA_SH:-}" ] && [ -f "$HASP_CONDA_SH" ]; then
    . "$HASP_CONDA_SH"
    conda activate "${HASP_CONDA_ENV:-hasp}" || {
        echo "[hasp] cannot activate '${HASP_CONDA_ENV:-hasp}' — set HASP_CONDA_ENV" >&2
        exit 1
    }
else
    echo "[hasp] no conda.sh found; using the ambient python" >&2
fi

cd "$HASP_ROOT" || exit 1
mkdir -p logs

export PYTHONPATH="$HASP_ROOT:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# --- per-job compile caches -------------------------------------------------
# Two jobs compiling the same model into one shared cache race, and one dies
# with a torch FXGraphCacheMiss. Give every job its own cache root, and keep it
# off a quota-limited home directory.
CJ="${HASP_CACHE_ROOT:-$HASP_ROOT/.cache}/job_${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$CJ/vllm" TRITON_CACHE_DIR="$CJ/triton" \
       TORCHINDUCTOR_CACHE_DIR="$CJ/torchinductor" XDG_CACHE_HOME="$CJ"
mkdir -p "$CJ"/{vllm,triton,torchinductor}
