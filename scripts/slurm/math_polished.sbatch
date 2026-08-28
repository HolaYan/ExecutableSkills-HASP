#!/bin/bash
#SBATCH --job-name=hasp_pol
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/pol-%j.log
#SBATCH --error=logs/pol-%j.err
#
# Math evidence PFs on the FINAL path: dispatch -> regen -> score.
#   sbatch --partition=<p> --account=<a> scripts/slurm/math_polished.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

TAG=${TAG:-pol1}; CASES=${CASES:-data/llm_anchor/cases.jsonl}
F() { python anchor/eval_polished_pfs.py --stage "$1" --cases "$CASES" --tag "$TAG" "${@:2}" 2>&1 | grep -vE "program_functions|INFO 0|WARNING 0"; }
[ -f data/polished/dispatch_$TAG.jsonl ] || F dispatch || exit 1
[ -f data/polished/regen_$TAG.jsonl ]    || F regen || exit 1
F score | tee logs/pol-score-$TAG.txt
exit 0
