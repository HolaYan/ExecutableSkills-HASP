#!/bin/bash
#SBATCH --job-name=hasp_forge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/forge-%j.log
#SBATCH --error=logs/forge-%j.err
#
# forge propose stage (1 GPU, no API): cluster -> propose -> screen.
# The cluster/screen/emit stages are CPU-only — see skills_construct/forge/README.md.
#
#   DOMAIN=math MODEL=Qwen/Qwen3-8B sbatch --partition=<p> --account=<a> scripts/slurm/forge.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

DOMAIN=${DOMAIN:-math}; MODEL=${MODEL:-Qwen/Qwen3-8B}; TAG=${TAG:-${DOMAIN}1}
FAMILIES=${FAMILIES:-6}; K=${K:-3}; N=${N:-2}; TP=${TP:-1}
echo "[forge] domain=$DOMAIN model=$MODEL tag=$TAG families=$FAMILIES k=$K n=$N"

# cluster is cheap; run it here so propose always has a fresh families.json
python -m skills_construct.forge --stage cluster --domain "$DOMAIN" --tag "$TAG" || exit 1
python -m skills_construct.forge --stage propose --domain "$DOMAIN" --tag "$TAG" \
    --model "$MODEL" --families "$FAMILIES" --k "$K" --n "$N" --tp "$TP" || exit 1

# screening is CPU-only and safe to run in the same allocation
python -m skills_construct.forge --stage screen --domain "$DOMAIN" --tag "$TAG" \
    | tee logs/forge-screen-$TAG.txt
echo "[forge] emit is a separate, explicit step:"
echo "  python -m skills_construct.forge --stage emit --domain $DOMAIN --tag $TAG [--register]"
exit 0
