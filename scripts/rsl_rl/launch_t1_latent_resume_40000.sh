#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES="${T1_RESUME_GPU:-6}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/resume_t1_latent.py \
    --checkpoint logs/rsl_rl/men_t1_amp/2026-09-17_18-03-20_latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192/model_24999.pt \
    --max_iterations "${T1_RESUME_TOTAL:-40000}" --num_envs "${T1_RESUME_ENVS:-8192}" \
    --run_name "${T1_RESUME_RUN:-latent20_resume24999_to40000_sameconfig_nomirror_noactorheading_8192}" \
    --headless --device cuda:0
