#!/bin/bash
#SBATCH --job-name=hasp_loop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/loop-%j.log
#SBATCH --error=logs/loop-%j.err
#
# One round of the forge loop: propose -> screen -> probe -> refine -> re-probe.
# Needs a GPU for the proposer and for the Case-C regeneration in the probe.
# Afterwards the round prints its own tier-2 accuracy command and admit command.
#
#   DOMAIN=math sbatch --partition=<p> --account=<a> scripts/slurm/forge_loop.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

DOMAIN=${DOMAIN:-math}; MODEL=${MODEL:-Qwen/Qwen3-8B}; TAG=${TAG:-${DOMAIN}_loop}
POLICY=${POLICY:-Qwen/Qwen3-4B-Instruct-2507}
FAMILIES=${FAMILIES:-6}; K=${K:-3}; N=${N:-2}; TP=${TP:-1}
REFINE=${REFINE:-1}; [ "$REFINE" = "0" ] && NOREF="--no-refine" || NOREF=""

echo "[loop] domain=$DOMAIN proposer=$MODEL policy=$POLICY families=$FAMILIES k=$K"
python -m skills_construct.forge --stage ledger --domain "$DOMAIN" || true
python -m skills_construct.forge --stage round --domain "$DOMAIN" --tag "$TAG" \
    --model "$MODEL" --policy "$POLICY" --families "$FAMILIES" --k "$K" --n "$N" \
    --tp "$TP" $NOREF 2>&1 | grep -vE "program_functions|INFO 0|WARNING 0" \
    | tee logs/loop-round-$TAG.txt
echo "[loop] done — see the tier-2 command printed above"
exit 0
