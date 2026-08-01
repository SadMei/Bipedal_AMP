from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BSRL_JOINT_COORDINATE_CONVENTION = "bsrl_xyz_positive_v1"
BSRL_GROUNDING_CONVENTION = "bsrl_per_frame_collision_v1"

BSRL_AMP_JOINT_NAMES = (
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
)

BSRL_AMP_KEY_BODY_NAMES = ("link_right_ankle_roll", "link_left_ankle_roll")

BSRL_URDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "source"
    / "legged_lab"
    / "legged_lab"
    / "data"
    / "Robots"
    / "BSRL_urdf"
    / "urdf"
    / "export.urdf"
)


def load_bsrl_joint_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(BSRL_URDF_PATH).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def validate_bsrl_joint_positions(joint_pos, joint_names, source: str, tolerance: float = 1.0e-5) -> None:
    names = tuple(joint_names)
    limits = load_bsrl_joint_limits()
    if len(names) != len(set(names)) or set(names) != set(limits):
        raise ValueError(f"{source}: joint names do not match {BSRL_URDF_PATH}")

    positions = np.asarray(joint_pos)
    if positions.ndim != 2 or positions.shape[1] != len(names):
        raise ValueError(f"{source}: invalid dof_pos shape {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError(f"{source}: dof_pos contains non-finite values")

    violations = []
    for index, name in enumerate(names):
        lower, upper = limits[name]
        actual_lower = float(positions[:, index].min())
        actual_upper = float(positions[:, index].max())
        if actual_lower < lower - tolerance or actual_upper > upper + tolerance:
            violations.append((name, actual_lower, actual_upper, lower, upper))
    if violations:
        raise ValueError(f"{source}: joint limit violations: {violations}")
