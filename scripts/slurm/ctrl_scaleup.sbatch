#!/bin/bash
#SBATCH --job-name=hasp_ctrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/ctrl-%j.log
#SBATCH --error=logs/ctrl-%j.err
#
# Validate "broke = 0" at scale: a large committed-CORRECT control set run
# through the full chain (locate -> regen both arms -> two cross-model arbiters
# -> score). On 146 controls broke=0 only bounds the true broke rate at ~2%
# (rule of three); ~600 controls brings that to ~0.5%.
#
#   sbatch --partition=<p> --account=<a> scripts/slurm/ctrl_scaleup.sbatch

source "${HASP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/env.sh"

CT=${CT:-ctrl600}
CONTROL=${CONTROL:-120}      # per dataset x 5 = ~600 committed-correct rollouts
TAG=${TAG:-q8b}
COMMON="--tag $TAG --cases-tag $CT"

[ -f "data/llm_anchor/cases_${CT}.jsonl" ] || python anchor/llm_locate.py --stage build $COMMON --control-per-ds "$CONTROL" --wrong-per-ds 0 || exit 1
[ -f "data/llm_anchor/locate_${TAG}_${CT}.jsonl" ] || python anchor/llm_locate.py --stage locate $COMMON --judge Qwen/Qwen3-8B --tp 1 --votes 1 || exit 1
[ -f "data/llm_anchor/regen_${TAG}_${CT}.jsonl" ]  || python anchor/llm_locate.py --stage regen  $COMMON --policy Qwen/Qwen3-4B-Instruct-2507 --tp 1 || exit 1
[ -f "data/llm_anchor/arbiter_${TAG}_gptoss20b_${CT}.jsonl" ] || python anchor/llm_locate.py --stage arbiter $COMMON --arb-tag gptoss20b --judge openai/gpt-oss-20b --tp 1 --max-model-len 16384 --no-thinking || exit 1
[ -f "data/llm_anchor/arbiter_${TAG}_q25_7b_${CT}.jsonl" ]    || python anchor/llm_locate.py --stage arbiter $COMMON --arb-tag q25_7b   --judge Qwen/Qwen2.5-7B-Instruct --tp 1 --max-model-len 16384 || exit 1
python anchor/llm_locate.py --stage score $COMMON --arb-tag gptoss20b | tee "logs/anchor-score-${TAG}_${CT}_gptoss20b.txt"
python anchor/llm_locate.py --stage score $COMMON --arb-tag q25_7b    | tee "logs/anchor-score-${TAG}_${CT}_q25_7b.txt"
exit 0
