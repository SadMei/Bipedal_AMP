#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES="${T1_CLEAN_AMP_GPU:-7}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-T1-v0 --num_envs "${T1_CLEAN_AMP_ENVS:-8192}" \
    --max_iterations "${T1_CLEAN_AMP_ITERS:-40000}" --seed 42 \
    --headless --device cuda:0 --logger tensorboard \
    --run_name "${T1_CLEAN_AMP_RUN:-latent20_cleanamp_uniform08_06_15_noheading_nocond_nomirror_40k_8192}" \
    --t1_clean_amp_baseline \
    logs/rsl_rl/men_t1_amp/2026-09-17_18-03-20_latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192
