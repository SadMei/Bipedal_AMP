#!/usr/bin/env bash
set -euo pipefail
cd /root/mjw/Bipedal_AMP
case "${1:-}" in
    off) export CUDA_VISIBLE_DEVICES=1; experiment_variant=nomirror ;;
    on) export CUDA_VISIBLE_DEVICES=5; experiment_variant=mirror01 ;;
    *) echo 'Usage: launch_bsrl_accad_mirror_pair.sh off|on' >&2; exit 2 ;;
esac
export BSRL_EXPLICIT_CLOSED_ANKLES=0
export BSRL_ANKLE_TRANSMISSION_DR=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec /root/miniconda3/envs/isaaclab/bin/python -u scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-BSRL-v0 --num_envs 8192 --max_iterations 30000 \
    --seed 42 --headless --device cuda:0 --logger tensorboard \
    --run_name "accad34_reduced_parallel_headingpriv_noestimator_${experiment_variant}_8192" \
    --bsrl_accad_baseline logs/rsl_rl/bsrl_amp/2026-09-16_10-51-46_reduced_parallelankle_headingframevel_criticpriv_mirror01_noactorheading_formal1_8192 \
    --bsrl_accad_manifest source/legged_lab/legged_lab/data/MotionData/bsrl/amp/accad_g1full_34_20260916/manifest.json \
    --bsrl_accad_mirror "$1"
