#!/bin/bash
#SBATCH --job-name=hasp_web
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/web-%j.log
#SBATCH --error=logs/web-%j.err
#
# Web grounding check on replayed episodes (no live search):
# build -> dispatch -> locate -> regen -> score.
#   TAG=web1 sbatch --partition=<p> --account=<a> scripts/slurm/web_polished.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

TAG=${TAG:-web1}
F() { python anchor/eval_web_polished.py --stage "$1" --tag "$TAG" "${@:2}" 2>&1 | grep -vE "INFO 0|WARNING 0"; }
[ -f data/web_polished/cases_$TAG.jsonl ]    || F build || exit 1
[ -f data/web_polished/dispatch_$TAG.jsonl ] || F dispatch || exit 1
[ -f data/web_polished/locate_$TAG.jsonl ]   || F locate --judge Qwen/Qwen3-8B || exit 1
[ -f data/web_polished/regen_$TAG.jsonl ]    || F regen  --policy Qwen/Qwen2.5-7B-Instruct || exit 1
F score | tee logs/web-score-$TAG.txt
exit 0
