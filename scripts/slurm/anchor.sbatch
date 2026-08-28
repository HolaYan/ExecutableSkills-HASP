#!/bin/bash
#SBATCH --job-name=hasp_anchor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/anchor-%j.log
#SBATCH --error=logs/anchor-%j.err
#
# LLM-anchor offline experiment: locate (judge) -> regen (policy) -> score.
# Submit from the repository root, supplying your site's partition/account:
#
#   sbatch --partition=<p> --account=<a> scripts/slurm/anchor.sbatch
#   JUDGE=Qwen/Qwen3-32B TAG=q32b NGPU=4 sbatch --gres=gpu:4 scripts/slurm/anchor.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

JUDGE=${JUDGE:-Qwen/Qwen3-8B}
POLICY=${POLICY:-Qwen/Qwen3-4B-Instruct-2507}
TAG=${TAG:-q8b}
NGPU=${NGPU:-1}
CONTROL=${CONTROL:-60}
VOTES=${VOTES:-3}

echo "[anchor] judge=$JUDGE policy=$POLICY tag=$TAG ngpu=$NGPU"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

[ -f data/llm_anchor/cases.jsonl ] || python anchor/llm_locate.py --stage build --control-per-ds "$CONTROL"
# Each stage is skipped when its output already exists (resume-friendly; lets a
# re-run after a score-stage edit reuse the expensive generations).
[ -f "data/llm_anchor/locate_${TAG}.jsonl" ]  || python anchor/llm_locate.py --stage locate  --judge "$JUDGE"  --tag "$TAG" --tp "$NGPU" --votes "$VOTES" || exit 1
[ -f "data/llm_anchor/regen_${TAG}.jsonl" ]   || python anchor/llm_locate.py --stage regen   --policy "$POLICY" --tag "$TAG" --tp 1 || exit 1
[ -f "data/llm_anchor/arbiter_${TAG}.jsonl" ] || python anchor/llm_locate.py --stage arbiter --judge "$JUDGE"  --tag "$TAG" --tp "$NGPU" || exit 1
python anchor/llm_locate.py --stage score --tag "$TAG" | tee "logs/anchor-score-${TAG}.txt"
exit 0
