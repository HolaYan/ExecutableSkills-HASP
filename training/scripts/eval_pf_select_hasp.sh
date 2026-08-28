#!/bin/bash
# Post-training evaluation — ALWAYS pf_select, via HASP's canonical harness.
#   bash training/scripts/eval_pf_select_hasp.sh <checkpoint_or_hf_path> <tag> [datasets]
# Runs the n=64 protocol (skills_off + pf_select with the HASP polished PF
# library) and writes data/model_eval/<tag>/summary. Report accuracy
# before → after only.
set -euo pipefail
CKPT=${1:?checkpoint path}; TAG=${2:?tag}; DS=${3:-aime24,amc23,olympiadbench}
cd "${HASP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Pass your site's partition/account through SBATCH_ARGS.
MODEL="$CKPT" TAG="$TAG" DATASETS="$DS" N=64 \
  sbatch ${SBATCH_ARGS:-} scripts/slurm/eval_models.sbatch
