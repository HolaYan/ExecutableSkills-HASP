#!/bin/bash
#SBATCH --job-name=hasp_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/cpu-%j.log
#SBATCH --error=logs/cpu-%j.err
#
# Generic CPU job wrapper for sandbox-heavy analyses. Never run these on a
# login node: they loop rlimited subprocess execution over a whole dataset.
#
#   CMD='python skills_construct/mining/code_error_taxonomy.py' LOG=logs/tax.log \
#     sbatch --partition=<cpu-partition> --account=<a> scripts/slurm/cpu.sbatch
source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

: "${CMD:?set CMD to the command to run}"
LOG=${LOG:-logs/cpu-${SLURM_JOB_ID:-local}.out}
echo "[cpu] $CMD -> $LOG"
bash -c "$CMD" > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
