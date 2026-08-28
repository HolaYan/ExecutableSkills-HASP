#!/bin/bash
#SBATCH --job-name=hasp_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval-%j.log
#SBATCH --error=logs/eval-%j.err
#
# skills_off vs pf_select for one model (canonical n=64 protocol).
# A full sweep of 3 datasets at n=64 does not fit in 12h; per-dataset results
# are flushed as each dataset finishes, so a timeout only loses unfinished ones.
#
#   sbatch --partition=<p> --account=<a> scripts/slurm/eval_models.sbatch
#   MODEL=Qwen/Qwen3-32B TAG=q3_32b NGPU=4 sbatch --gres=gpu:4 --mem=160GB ... scripts/slurm/eval_models.sbatch
#   THINKING=1 TAG=q3_8b_think N=16 sbatch ... scripts/slurm/eval_models.sbatch   # thinking mode is long

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

MODEL=${MODEL:-Qwen/Qwen3-8B}
TAG=${TAG:-q3_8b}
NGPU=${NGPU:-1}
N=${N:-64}
DATASETS=${DATASETS:-aime24,amc23,olympiadbench}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
TH=""; [ "${THINKING:-0}" = "1" ] && TH="--thinking"

echo "[eval] model=$MODEL tag=$TAG n=$N datasets=$DATASETS thinking=${THINKING:-0} ngpu=$NGPU"
python pf_select/eval_models.py --model "$MODEL" --tag "$TAG" --datasets "$DATASETS" --n "$N" \
    --tp "$NGPU" --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" $TH \
    2>&1 | grep -vE "program_functions|INFO 0|WARNING 0" | tee "logs/eval-${TAG}.txt"
exit 0
