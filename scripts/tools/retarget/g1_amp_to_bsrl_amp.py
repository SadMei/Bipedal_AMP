import argparse
from pathlib import Path

import joblib
import numpy as np


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BSRL_JOINT_NAMES = [
    "joint_left_hip_yaw",
    "joint_right_hip_yaw",
    "joint_left_hip_roll",
    "joint_right_hip_roll",
    "joint_left_hip_pitch",
    "joint_right_hip_pitch",
    "joint_left_knee_pitch",
    "joint_right_knee_pitch",
    "joint_left_ankle_pitch",
    "joint_right_ankle_pitch",
    "joint_left_ankle_roll",
    "joint_right_ankle_roll",
]

G1_TO_BSRL_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# G1 key-body order in g1_amp_env_cfg.py:
# left_ankle_roll, right_ankle_roll, left_wrist_yaw, right_wrist_yaw, left_shoulder_roll, right_shoulder_roll
G1_KEY_BODY_INDICES_FOR_BSRL = [1, 0]  # right ankle, left ankle


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slice G1 AMP motion data into a BSRL 12DoF transitional dataset.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run",
        help="Directory containing source G1 AMP motion pickle files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="source/legged_lab/legged_lab/data/MotionData/bsrl_12dof/amp/walk_and_run",
        help="Directory where converted BSRL AMP motion pickle files will be written.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"No source motions found in {input_dir}")

    g1_indices = [G1_JOINT_NAMES.index(name) for name in G1_TO_BSRL_JOINT_NAMES]

    for source_path in source_files:
        motion = joblib.load(source_path)

        output_motion = {
            "fps": float(motion["fps"]),
            "root_pos": np.asarray(motion["root_pos"], dtype=np.float32),
            "root_rot": np.asarray(motion["root_rot"], dtype=np.float32),
            "dof_pos": np.asarray(motion["dof_pos"][:, g1_indices], dtype=np.float32),
            "loop_mode": int(motion["loop_mode"]),
            "key_body_pos": np.asarray(motion["key_body_pos"][:, G1_KEY_BODY_INDICES_FOR_BSRL, :], dtype=np.float32),
            "joint_names": BSRL_JOINT_NAMES,
            "source_motion": source_path.stem,
        }

        output_path = output_dir / source_path.name
        joblib.dump(output_motion, output_path)
        print(f"[INFO] Wrote {output_path}")

    print(f"[INFO] Converted {len(source_files)} motions into {output_dir}")


if __name__ == "__main__":
    main()
