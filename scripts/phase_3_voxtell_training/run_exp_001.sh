#!/bin/bash
# ===============================================================================
# SCRIPT:         Launch Multi-GPU Exp 001 Fine-Tuning
# LOCATION:       scripts/phase_3_voxtell_training/run_exp_001.sh
# ===============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_PORT=29525
mkdir -p logs/phase_3_voxtell_training/exp_001_naive_finetuning

nohup .venv/bin/torchrun --nproc_per_node=3 --master_port=${MASTER_PORT} \
    scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py \
    --epochs 50 \
    --batch_size 1 \
    --lr 1e-4 \
    --num_workers 2 \
    > logs/phase_3_voxtell_training/exp_001_naive_finetuning/nohup.log 2>&1 &

PID=$!
echo "Exp 001 Multi-GPU training launched with PID: ${PID}"
