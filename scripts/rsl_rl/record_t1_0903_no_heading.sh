#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
export CUDA_VISIBLE_DEVICES=7
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac-AMP-T1-v0 --num_envs 1 --seed 42 --headless --device cuda:0 \
    --checkpoint logs/rsl_rl/men_t1_amp/2026-09-03_11-02-15_speedmatched_no_policy_heading_estimator4d_formal1_8192/model_24500.pt \
    --use_estimator --restore_saved_t1_config --no-export_policy \
    --nominal_eval --command_profile t1_forward_06 --video --video_length 1500 \
    --video_name t1_20260903_1102_noactorheading_model24500_forward06_nominal_30s_20260915 \
    --replay_audit_dir reports/t1_20260903_1102_model24500_noactorheading_video_20260915_retry2 \
    --debug_state --debug_interval 50
