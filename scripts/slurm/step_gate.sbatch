#!/bin/bash
#SBATCH --job-name=hasp_sg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/sg-%j.log
#SBATCH --error=logs/sg-%j.err
#
# Step-level dual-consent PF intervention, offline on the paired cases.
#   sbatch --partition=<p> --account=<a> scripts/slurm/step_gate.sbatch
#   CASES=data/llm_anchor/cases_ctrl600.jsonl TAG=sg1_ctrl sbatch ... scripts/slurm/step_gate.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

CASES=${CASES:-data/llm_anchor/cases.jsonl}
TAG=${TAG:-sg1}
C="--cases $CASES --tag $TAG"
F() { python anchor/step_gate.py --stage "$1" $C "${@:2}" 2>&1 | grep -vE "program_functions|INFO 0|WARNING 0"; }

[ -f data/step_gate/anchor_$TAG.jsonl ]   || F anchor || exit 1
[ -f data/step_gate/consent_$TAG.jsonl ]  || F consent  --policy Qwen/Qwen3-4B-Instruct-2507 || exit 1
[ -f data/step_gate/evidence_$TAG.jsonl ] || F evidence --judge Qwen/Qwen3-8B || exit 1
[ -f data/step_gate/regen_$TAG.jsonl ]    || F regen    --policy Qwen/Qwen3-4B-Instruct-2507 || exit 1
F score | tee "logs/sg-score-$TAG.txt"
exit 0
