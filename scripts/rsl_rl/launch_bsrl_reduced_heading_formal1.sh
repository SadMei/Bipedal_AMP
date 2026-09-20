#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES=3
export BSRL_EXPLICIT_CLOSED_ANKLES=0
export BSRL_ANKLE_TRANSMISSION_DR=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-BSRL-v0 --num_envs 8192 --max_iterations 30000 \
    --seed 42 --headless --device cuda:0 --logger tensorboard \
    --run_name reduced_parallelankle_headingframevel_criticpriv_mirror01_noactorheading_formal1_8192 \
    --bsrl_reduced_heading_baseline \
    logs/rsl_rl/bsrl_amp/2026-09-15_15-14-55_reduced_parallelankle_nominal_pd200_noestimator_mirror01_formal1_8192
