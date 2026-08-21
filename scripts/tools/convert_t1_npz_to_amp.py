"""Convert validated wbt_est T1 NPZ motions into Bipedal_AMP PKL motions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


T1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "torso_pitch_joint",
    "torso_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

T1_KEY_BODY_NAMES = (
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
)

REQUIRED_KEYS = (
    "fps",
    "robot_model",
    "robot_dof",
    "joint_names",
    "body_names",
    "actuated_joint_order",
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
)


def _unique_indices(available: list[str], required: tuple[str, ...], label: str) -> list[int]:
    if len(available) != len(set(available)):
        raise ValueError(f"{label} contains duplicate names")
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{label} is missing required names: {missing}")
    return [available.index(name) for name in required]


def convert_motion(source_path: Path, output_path: Path) -> None:
    with np.load(source_path, allow_pickle=False) as source:
        missing_keys = [key for key in REQUIRED_KEYS if key not in source]
        if missing_keys:
            raise ValueError(f"{source_path}: missing keys {missing_keys}")

        robot_model = str(source["robot_model"].item()).lower()
        robot_dof = int(source["robot_dof"].item())
        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
        if robot_model != "t1" or robot_dof != len(T1_JOINT_NAMES):
            raise ValueError(
                f"{source_path}: expected T1/{len(T1_JOINT_NAMES)} DoF, got {robot_model}/{robot_dof}"
            )
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{source_path}: invalid fps {fps}")

        joint_names = source["joint_names"].tolist()
        body_names = source["body_names"].tolist()
        joint_indices = _unique_indices(joint_names, T1_JOINT_NAMES, "joint_names")
        key_body_indices = _unique_indices(body_names, T1_KEY_BODY_NAMES, "body_names")
        root_index = _unique_indices(body_names, ("pelvis_Link",), "body_names")[0]

        declared_order = tuple(source["actuated_joint_order"].tolist())
        if declared_order != T1_JOINT_NAMES:
            raise ValueError(
                f"{source_path}: actuated_joint_order does not match the T1 policy/AMP contract"
            )

        joint_pos = np.asarray(source["joint_pos"][:, joint_indices], dtype=np.float32)
        root_pos = np.asarray(source["body_pos_w"][:, root_index], dtype=np.float32)
        root_rot = np.asarray(source["body_quat_w"][:, root_index], dtype=np.float32)
        key_body_pos = np.asarray(source["body_pos_w"][:, key_body_indices], dtype=np.float32)

    frame_count = len(joint_pos)
    expected_shapes = {
        "joint_pos": (frame_count, len(T1_JOINT_NAMES)),
        "root_pos": (frame_count, 3),
        "root_rot": (frame_count, 4),
        "key_body_pos": (frame_count, len(T1_KEY_BODY_NAMES), 3),
    }
    arrays = {
        "joint_pos": joint_pos,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "key_body_pos": key_body_pos,
    }
    if frame_count < 5:
        raise ValueError(f"{source_path}: at least five frames are required for the four-step AMP window")
    for name, values in arrays.items():
        if values.shape != expected_shapes[name]:
            raise ValueError(f"{source_path}: {name} shape {values.shape}, expected {expected_shapes[name]}")
        if not np.isfinite(values).all():
            raise ValueError(f"{source_path}: {name} contains NaN or Inf")

    quaternion_norm = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(quaternion_norm < 1.0e-6):
        raise ValueError(f"{source_path}: root_rot contains a zero quaternion")
    root_rot = root_rot / quaternion_norm
    consecutive_dot = np.sum(root_rot[:-1] * root_rot[1:], axis=1)
    if np.any(consecutive_dot < 0.0):
        raise ValueError(f"{source_path}: root_rot has quaternion sign discontinuities")

    motion = {
        "fps": fps,
        "loop_mode": 0,
        "root_pos": root_pos,
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": joint_pos,
        "key_body_pos": key_body_pos,
        "joint_names": T1_JOINT_NAMES,
        "key_body_names": T1_KEY_BODY_NAMES,
        "source_file": source_path.name,
    }
    joblib.dump(motion, output_path, compress=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dir", type=Path, help="Directory containing wbt_est T1 .npz files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated .pkl files")
    args = parser.parse_args()

    source_paths = sorted(args.motion_dir.glob("*.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No .npz files found in {args.motion_dir}")

    for source_path in source_paths:
        output_path = source_path.with_suffix(".pkl")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace generated data")
        convert_motion(source_path, output_path)
        print(f"converted {source_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
