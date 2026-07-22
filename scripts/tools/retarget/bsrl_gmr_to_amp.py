import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np


AMP_JOINT_NAMES = [
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

AMP_KEY_BODY_NAMES = [
    "link_right_ankle_roll",
    "link_left_ankle_roll",
]


def convert_motion(source_path, output_path, loop_mode):
    with open(source_path, "rb") as source_file:
        source = pickle.load(source_file)

    source_joint_names = source["dof_names"]
    source_key_body_names = source["key_body_names"]
    joint_indices = [source_joint_names.index(name) for name in AMP_JOINT_NAMES]
    key_body_indices = [source_key_body_names.index(name) for name in AMP_KEY_BODY_NAMES]

    root_rot_xyzw = np.asarray(source["root_rot"], dtype=np.float32)
    output = {
        "fps": float(source["fps"]),
        "root_pos": np.asarray(source["root_pos"], dtype=np.float32),
        "root_rot": root_rot_xyzw[:, [3, 0, 1, 2]],
        "dof_pos": np.asarray(source["dof_pos"][:, joint_indices], dtype=np.float32),
        "loop_mode": int(loop_mode),
        "key_body_pos": np.asarray(source["key_body_pos"][:, key_body_indices], dtype=np.float32),
        "joint_names": AMP_JOINT_NAMES,
        "key_body_names": AMP_KEY_BODY_NAMES,
        "source_motion": source.get("source_motion", source_path.name),
    }

    frame_count = len(output["root_pos"])
    for key in ("root_rot", "dof_pos", "key_body_pos"):
        if len(output[key]) != frame_count:
            raise ValueError(f"{source_path}: {key} has {len(output[key])} frames, expected {frame_count}")
    if not all(np.isfinite(output[key]).all() for key in ("root_pos", "root_rot", "dof_pos", "key_body_pos")):
        raise ValueError(f"{source_path}: motion contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, output_path)
    print(f"[INFO] Wrote {output_path} ({frame_count} frames)")


def main():
    parser = argparse.ArgumentParser(description="Convert BSRL GMR motions into the AMP training format.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--loop", choices=("clamp", "wrap"), default="clamp")
    args = parser.parse_args()

    source_files = sorted(args.input_dir.rglob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"No GMR pickle files found in {args.input_dir}")

    loop_mode = 0 if args.loop == "clamp" else 1
    for source_path in source_files:
        relative_path = source_path.relative_to(args.input_dir)
        convert_motion(source_path, args.output_dir / relative_path, loop_mode)

    print(f"[INFO] Converted {len(source_files)} motions into {args.output_dir}")


if __name__ == "__main__":
    main()
