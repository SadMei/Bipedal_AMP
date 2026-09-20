#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES="${T1_LATENT_GPU:-6}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Valid-transition batch sizes vary; expandable segments limit fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-T1-v0 --num_envs "${T1_LATENT_ENVS:-8192}" \
    --max_iterations "${T1_LATENT_ITERS:-25000}" --seed 42 \
    --headless --device cuda:0 --logger tensorboard \
    --run_name "${T1_LATENT_RUN:-latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192}" \
    --t1_latent_baseline \
    logs/rsl_rl/men_t1_amp/2026-09-15_19-40-00_resetmatch_noactorheading_noestimator_mirror01_noclip_jointlimit5_formal1_8192
