#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
# A new run under the existing BSRL task; do not stop reduced-ankle or T1 runs.
export CUDA_VISIBLE_DEVICES=4
export BSRL_EXPLICIT_CLOSED_ANKLES=1
export BSRL_ANKLE_TRANSMISSION_DR=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p logs/launches
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --headless --device cuda:0 --task LeggedLab-Isaac-AMP-BSRL-v0 \
    --num_envs 8192 --max_iterations 30000 --seed 42 \
    --run_name explicit_closedankle_tgs12v6_phys1ms_pd200_gapmonitor_noestimator_mirror01_formal1_8192 \
    agent.save_interval=100 \
    > logs/launches/bsrl_closedankle_tgs126_formal1_20260915.log 2>&1
