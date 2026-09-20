#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES=7
export BSRL_EXPLICIT_CLOSED_ANKLES=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-BSRL-v0 --num_envs 8192 --max_iterations 30000 \
    --seed 42 --headless --device cuda:0 --logger tensorboard \
    --run_name serial_pd_hip150_knee200_ankle100_noestimator_nomirror_formal1_8192 \
    --bsrl_serial_no_mirror_baseline \
    logs/rsl_rl/bsrl_amp/2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192
