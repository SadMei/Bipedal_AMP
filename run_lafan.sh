#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode=${1:-help}
if [[ "$mode" != train && "$mode" != play ]]; then
  echo 'Usage: ./run_lafan.sh train|play g1|bsrl [extra train.py/play.py arguments]'
  echo 'Example: ./run_lafan.sh train bsrl --max_iterations 1000'
  echo 'Example: ./run_lafan.sh play bsrl --checkpoint /absolute/path/model_1000.pt --video'
  exit 0
fi
shift
robot=g1
if [[ $# -gt 0 && "$1" != -* ]]; then
  robot=$1
  shift
fi
source ./env.sh
case "$robot" in
  g1) task=LeggedLab-Isaac-AMP-G1-v0; data_robot=g1_29dof ;;
  bsrl) task=LeggedLab-Isaac-AMP-BSRL-v0; data_robot=bsrl_12dof ;;
  *) echo "Unknown robot: $robot" >&2; exit 2 ;;
esac
export LEGGED_LAB_MOTION_DIR="${LEGGED_LAB_MOTION_DIR:-${LAFAN_DATASET_ROOT:-$BIPEDAL_AMP_ROOT/../datasets/lafan1}/amp/$data_robot}"
if [[ ! -f "$LEGGED_LAB_MOTION_DIR/manifest.json" ]]; then
  echo "Missing converted LAFAN dataset: $LEGGED_LAB_MOTION_DIR" >&2
  exit 1
fi
if [[ "$mode" == train ]]; then
  exec python scripts/rsl_rl/train.py --task "$task" --headless --device cuda:0 \
    --num_envs 1024 --logger tensorboard "$@"
else
  exec python scripts/rsl_rl/play.py --task "$task" --headless --device cuda:0 \
    --num_envs 4 "$@"
fi
