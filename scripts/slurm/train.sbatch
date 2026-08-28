#!/bin/bash
#SBATCH --job-name=hasp_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.log
#SBATCH --error=logs/train-%j.err
#
# Train with PFs. The training method is read from the config's `method:` key,
# so one script covers rs / sft / grpo.
#
#   CONFIG=configs/training/H1_math_rs.yaml \
#     sbatch --partition=<p> --account=<a> scripts/slurm/train.sbatch
#
# Raise --gres / --mem / --time for the larger configurations.

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

CONFIG=${CONFIG:?set CONFIG to a training config, e.g. configs/training/H1_math_rs.yaml}
[ -f "$CONFIG" ] || { echo "[train] no such config: $CONFIG" >&2; exit 1; }

METHOD=$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get('method',''))" "$CONFIG")
case "$METHOD" in
    rs)   MODULE=training.rejection_sampling.train ;;
    sft)  MODULE=training.sft.train ;;
    grpo) MODULE=training.grpo.train ;;
    *)    echo "[train] config has no usable 'method:' key (got '$METHOD')" >&2; exit 2 ;;
esac

TAG=$(basename "$CONFIG" .yaml)
echo "[train] config=$CONFIG method=$METHOD -> $MODULE"
python -m "$MODULE" --config "$CONFIG" 2>&1 | tee "logs/train-${TAG}.log"
