"""Convert T1 GMR PKL motions to the Bipedal_AMP reference format.

The input files are preserved.  The converter uses the Men T1 0722 new-shoe
URDF for forward kinematics and writes converted files to a separate output
directory.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pinocchio as pin


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

REQUIRED_KEYS = ("root_pos", "root_rot", "dof_pos", "dof_names", "fps")


def _joint_limits(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    missing = [name for name in T1_JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"{urdf_path}: missing limits for {missing}")
    lower, upper = zip(*(limits[name] for name in T1_JOINT_NAMES))
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def _normalize_quaternions_xyzw(quaternions: np.ndarray, source_path: Path) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-6):
        raise ValueError(f"{source_path}: root_rot contains a zero quaternion")
    quaternions = quaternions / norms
    # q and -q encode the same rotation. Keep consecutive samples in the same hemisphere.
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def _key_body_forward_kinematics(
    model: pin.Model,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    data = model.createData()
    frame_ids = [model.getFrameId(name) for name in T1_KEY_BODY_NAMES]
    if any(frame_id >= model.nframes for frame_id in frame_ids):
        raise ValueError("0722 URDF does not contain every required T1 key body")

    configuration = np.empty(model.nq, dtype=np.float64)
    key_body_pos = np.empty((len(root_pos), len(frame_ids), 3), dtype=np.float32)
    for frame_index in range(len(root_pos)):
        # Pinocchio's free-flyer configuration is position followed by xyzw quaternion.
        configuration[:3] = root_pos[frame_index]
        configuration[3:7] = root_quat_xyzw[frame_index]
        configuration[7:] = dof_pos[frame_index]
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        for body_index, body_frame_id in enumerate(frame_ids):
            key_body_pos[frame_index, body_index] = data.oMf[body_frame_id].translation
    return key_body_pos


def convert_motion(source_path: Path, output_path: Path, urdf_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace generated data")

    source = joblib.load(source_path)
    if not isinstance(source, dict):
        raise ValueError(f"{source_path}: expected a motion dictionary")
    missing = [key for key in REQUIRED_KEYS if key not in source]
    if missing:
        raise ValueError(f"{source_path}: missing keys {missing}")

    source_joint_names = tuple(str(name) for name in np.asarray(source["dof_names"]).tolist())
    if len(source_joint_names) != len(set(source_joint_names)):
        raise ValueError(f"{source_path}: dof_names contains duplicates")
    missing_joints = [name for name in T1_JOINT_NAMES if name not in source_joint_names]
    if missing_joints:
        raise ValueError(f"{source_path}: missing T1 joints {missing_joints}")
    joint_indices = [source_joint_names.index(name) for name in T1_JOINT_NAMES]

    root_pos = np.asarray(source["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(source["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(source["dof_pos"][:, joint_indices], dtype=np.float32)
    fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    frame_count = len(root_pos)
    expected_shapes = {
        "root_pos": (frame_count, 3),
        "root_rot": (frame_count, 4),
        "dof_pos": (frame_count, len(T1_JOINT_NAMES)),
    }
    for name, values in (("root_pos", root_pos), ("root_rot", root_quat_xyzw), ("dof_pos", dof_pos)):
        if values.shape != expected_shapes[name]:
            raise ValueError(f"{source_path}: {name} has shape {values.shape}, expected {expected_shapes[name]}")
        if not np.isfinite(values).all():
            raise ValueError(f"{source_path}: {name} contains NaN or Inf")
    if frame_count < 5:
        raise ValueError(f"{source_path}: at least five frames are required")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{source_path}: invalid fps {fps}")

    root_quat_xyzw = _normalize_quaternions_xyzw(root_quat_xyzw, source_path)
    lower, upper = _joint_limits(urdf_path)
    violation_mask = (dof_pos < lower) | (dof_pos > upper)
    clipped_values = int(violation_mask.sum())
    dof_pos = np.clip(dof_pos, lower, upper)

    model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    if model.nq != 7 + len(T1_JOINT_NAMES):
        raise ValueError(f"{urdf_path}: expected {len(T1_JOINT_NAMES)} actuated joints, got nq={model.nq}")
    model_joint_names = tuple(model.names[2:])
    if model_joint_names != T1_JOINT_NAMES:
        raise ValueError(f"{urdf_path}: actuated joint order does not match the T1 AMP contract")
    key_body_pos = _key_body_forward_kinematics(model, root_pos, root_quat_xyzw, dof_pos)

    # Isaac Lab consumes quaternions in wxyz order.
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    output = {
        "fps": fps,
        "loop_mode": 0,
        "root_pos": root_pos,
        "root_rot": root_quat_wxyz,
        "dof_pos": dof_pos.astype(np.float32),
        "key_body_pos": key_body_pos,
        "joint_names": T1_JOINT_NAMES,
        "key_body_names": T1_KEY_BODY_NAMES,
        "source_file": source_path.name,
        "source_format": "gmr_xyzw",
        "urdf_file": urdf_path.name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(output, temporary_path, compress=3)
    os.replace(temporary_path, output_path)
    print(
        f"converted {source_path.name}: frames={frame_count}, fps={fps:g}, "
        f"clipped_joint_values={clipped_values} -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No PKL files found in {args.input_dir}")
    if not args.urdf.is_file():
        raise FileNotFoundError(args.urdf)
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Input and output directories must differ so source PKLs remain untouched")

    for source_path in input_files:
        convert_motion(source_path, args.output_dir / source_path.name, args.urdf, args.overwrite)


if __name__ == "__main__":
    main()
