#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
# Only this process selects the new nominal transmission baseline. Keep all
# existing tmux jobs alive; no smoke run and no estimator/normalization change.
export CUDA_VISIBLE_DEVICES=5
export BSRL_EXPLICIT_CLOSED_ANKLES=0
export BSRL_ANKLE_TRANSMISSION_DR=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --headless --device cuda:0 --task LeggedLab-Isaac-AMP-BSRL-v0 \
    --num_envs 8192 --max_iterations 30000 \
    --run_name reduced_parallelankle_nominal_pd200_noestimator_mirror01_formal1_8192 \
    agent.save_interval=100 \
    > logs/launches/bsrl_reducedankle_formal1_20260915.log 2>&1
