#!/bin/bash
#SBATCH --job-name=hasp_code
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/code-%j.log
#SBATCH --error=logs/code-%j.err
#
# Code PFs, first configuration: build -> dispatch -> regen -> score.
#   CODE_BENCHMARKS=humaneval_plus,mbpp_plus TAG=code1 \
#     sbatch --partition=<p> --account=<a> scripts/slurm/code_polished.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

TAG=${TAG:-code1}
export CODE_BENCHMARKS=${CODE_BENCHMARKS:-humaneval_plus,mbpp_plus}
F() { python anchor/eval_code_polished.py --stage "$1" --tag "$TAG" "${@:2}" 2>&1 | grep -vE "INFO 0|WARNING 0"; }
[ -f data/code_polished/cases_$TAG.jsonl ]    || F build || exit 1
[ -f data/code_polished/dispatch_$TAG.jsonl ] || F dispatch || exit 1
[ -f data/code_polished/regen_$TAG.jsonl ]    || F regen || exit 1
F score | tee logs/code-score-$TAG.txt
exit 0
