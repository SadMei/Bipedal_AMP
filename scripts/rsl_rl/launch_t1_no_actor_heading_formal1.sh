#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
# New observation-only comparison; do not stop or modify existing runs.
export CUDA_VISIBLE_DEVICES=6
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-T1-v0 --num_envs 8192 --max_iterations 30000 \
    --seed 42 --headless --device cuda:0 --logger tensorboard \
    --run_name resetmatch_noactorheading_noestimator_mirror01_noclip_jointlimit5_formal1_8192 \
    --t1_no_actor_heading_baseline \
    logs/rsl_rl/men_t1_amp/2026-09-14_21-52-57_resetmatch_noestimator_mirror01_noclip_jointlimit5_formal1_8192
