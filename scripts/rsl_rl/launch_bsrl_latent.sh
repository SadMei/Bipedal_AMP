#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES="${BSRL_LATENT_GPU:-7}"
export BSRL_EXPLICIT_CLOSED_ANKLES=0
export BSRL_ANKLE_TRANSMISSION_DR=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-BSRL-v0 --num_envs "${BSRL_LATENT_ENVS:-8192}" \
    --max_iterations "${BSRL_LATENT_ITERS:-25000}" --seed 42 \
    --headless --device cuda:0 --logger tensorboard \
    --run_name "${BSRL_LATENT_RUN:-accad34_reduced_latent20_h10_recent2_predict33_kl001_nomirror_8192}" \
    --bsrl_latent_baseline \
    logs/rsl_rl/bsrl_amp/2026-09-16_21-06-48_accad34_reduced_parallel_headingpriv_noestimator_nomirror_8192
