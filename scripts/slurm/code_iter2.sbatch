#!/bin/bash
#SBATCH --job-name=hasp_code2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/code2-%j.log
#SBATCH --error=logs/code2-%j.err
#
# Code PFs, accepted configuration: best-of-4 repair accepting the first sample
# that passes the spec's own examples. alt -> diff -> regen(n=4) -> score.
#   TAG=code3 sbatch --partition=<p> --account=<a> scripts/slurm/code_iter2.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

TAG=${TAG:-code3}
F() { python anchor/eval_code_iter2.py --stage "$1" --tag "$TAG" "${@:2}" 2>&1 | grep -vE "INFO 0|WARNING 0"; }
[ -f data/code_polished/alt_$TAG.jsonl ]   || F alt   || exit 1
[ -f data/code_polished/diff_$TAG.jsonl ]  || F diff  || exit 1
[ -f data/code_polished/regen_$TAG.jsonl ] || F regen --n 4 || exit 1
F score | tee logs/code2-score-$TAG.txt
exit 0
