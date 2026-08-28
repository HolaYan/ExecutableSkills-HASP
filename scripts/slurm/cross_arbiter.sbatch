#!/bin/bash
#SBATCH --job-name=hasp_xarb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/xarb-%j.log
#SBATCH --error=logs/xarb-%j.err
#
# Cross-model arbiter: re-arbitrate an existing regen_<TAG>.jsonl with a judge
# from a DIFFERENT model family, then re-score. Motivation: with one model as
# locator, voter and arbiter its misreadings are correlated, and the residual
# broke cases survive every same-model gate.
#
#   sbatch --partition=<p> --account=<a> scripts/slurm/cross_arbiter.sbatch
#   ARB=Qwen/Qwen3.5-9B ARBTAG=q35_9b sbatch ... scripts/slurm/cross_arbiter.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

TAG=${TAG:-q8b_v2}
ARB=${ARB:-openai/gpt-oss-20b}
ARBTAG=${ARBTAG:-gptoss20b}
EXTRA=${EXTRA:---no-thinking}     # gpt-oss has no enable_thinking kwarg; harmless for Qwen3.5

echo "[xarb] regen tag=$TAG  arbiter=$ARB ($ARBTAG)"
python anchor/llm_locate.py --stage arbiter --judge "$ARB" --tag "$TAG" --arb-tag "$ARBTAG" --tp 1 \
    --max-model-len 16384 $EXTRA || exit 1
python anchor/llm_locate.py --stage score --tag "$TAG" --arb-tag "$ARBTAG" | tee "logs/anchor-score-${TAG}_${ARBTAG}.txt"
exit 0
