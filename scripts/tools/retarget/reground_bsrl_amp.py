"""Recompute BSRL AMP grounding and key-body positions with the current URDF.

This is intended for already-retargeted AMP motions when the source BVH is no
longer available.  It preserves the root XY trajectory, root orientation, and
joint trajectory, while recomputing the URDF-dependent root height and forward
kinematics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mujoco as mj
import numpy as np

from bsrl_conventions import (
    BSRL_AMP_JOINT_NAMES,
    BSRL_AMP_KEY_BODY_NAMES,
    BSRL_GROUNDING_CONVENTION,
    BSRL_JOINT_COORDINATE_CONVENTION,
    BSRL_URDF_PATH,
    validate_bsrl_joint_positions,
)


def _name(model: mj.MjModel, object_type: mj.mjtObj, object_id: int) -> str:
    name = mj.mj_id2name(model, object_type, object_id)
    if name is None:
        raise ValueError(f"Unnamed MuJoCo object: type={object_type}, id={object_id}")
    return name


def _joint_names_in_qpos_order(model: mj.MjModel) -> list[str]:
    joints = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        joints.append(
            (
                int(model.jnt_qposadr[joint_id]),
                _name(model, mj.mjtObj.mjOBJ_JOINT, joint_id),
            )
        )
    return [name for _, name in sorted(joints)]


class UrdfCollisionGrounder:
    """Find the lowest point of collision geoms imported from a URDF."""

    def __init__(self, model: mj.MjModel):
        self.model = model
        self.data = mj.MjData(model)
        # MuJoCo's URDF importer puts collision geoms in group 0 and visual
        # geoms in group 1.  The visual meshes must not affect grounding.
        self.geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_group[geom_id]) == 0
        ]
        if not self.geom_ids:
            raise ValueError("The URDF contains no collision geoms in MuJoCo group 0")

    def _geom_min_z(self, geom_id: int) -> float:
        geom_type = self.model.geom_type[geom_id]
        geom_pos = self.data.geom_xpos[geom_id]
        geom_mat = self.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mj.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            start = self.model.mesh_vertadr[mesh_id]
            count = self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[start : start + count] * self.model.mesh_scale[mesh_id]
            return float((geom_pos + vertices @ geom_mat.T)[:, 2].min())

        if geom_type == mj.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = geom_size[:3]
            corners = np.array(
                [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
            )
            return float((geom_pos + corners @ geom_mat.T)[:, 2].min())

        if geom_type == mj.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])

        if geom_type in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            axis = geom_mat[:, 2]
            radius = geom_size[0]
            half_length = geom_size[1]
            z_extent = abs(axis[2]) * half_length
            z_extent += radius * np.sqrt(max(0.0, 1.0 - axis[2] ** 2))
            if geom_type == mj.mjtGeom.mjGEOM_CAPSULE:
                z_extent += radius * abs(axis[2])
            return float(geom_pos[2] - z_extent)

        raise ValueError(f"Unsupported collision geom type {geom_type} for geom {geom_id}")

    def min_z(self, qpos: np.ndarray) -> float:
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        return min(self._geom_min_z(geom_id) for geom_id in self.geom_ids)


def _build_qpos(model: mj.MjModel, motion: dict, source: Path) -> np.ndarray:
    required = {
        "root_pos",
        "root_rot",
        "dof_pos",
        "joint_names",
        "key_body_names",
        "joint_coordinate_convention",
        "grounding_convention",
    }
    missing = sorted(required - motion.keys())
    if missing:
        raise ValueError(f"{source}: missing fields {missing}")
    if tuple(motion["joint_names"]) != BSRL_AMP_JOINT_NAMES:
        raise ValueError(f"{source}: unexpected AMP joint order")
    if tuple(motion["key_body_names"]) != BSRL_AMP_KEY_BODY_NAMES:
        raise ValueError(f"{source}: unexpected key-body order")
    if motion["joint_coordinate_convention"] != BSRL_JOINT_COORDINATE_CONVENTION:
        raise ValueError(f"{source}: unexpected joint coordinate convention")
    if motion["grounding_convention"] != BSRL_GROUNDING_CONVENTION:
        raise ValueError(f"{source}: unexpected grounding convention")

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    frame_count = len(root_pos)
    if root_pos.shape != (frame_count, 3):
        raise ValueError(f"{source}: invalid root_pos shape {root_pos.shape}")
    if root_rot.shape != (frame_count, 4):
        raise ValueError(f"{source}: invalid root_rot shape {root_rot.shape}")
    if dof_pos.shape != (frame_count, len(BSRL_AMP_JOINT_NAMES)):
        raise ValueError(f"{source}: invalid dof_pos shape {dof_pos.shape}")
    if not all(np.isfinite(array).all() for array in (root_pos, root_rot, dof_pos)):
        raise ValueError(f"{source}: motion contains non-finite values")
    quaternion_norm = np.linalg.norm(root_rot, axis=1)
    if np.max(np.abs(quaternion_norm - 1.0)) > 1.0e-4:
        raise ValueError(f"{source}: root quaternion is not normalized")
    validate_bsrl_joint_positions(dof_pos, BSRL_AMP_JOINT_NAMES, str(source))

    model_joint_names = _joint_names_in_qpos_order(model)
    if set(model_joint_names) != set(BSRL_AMP_JOINT_NAMES):
        raise ValueError(f"{source}: URDF joint set differs from the AMP joint set")
    source_indices = {name: index for index, name in enumerate(BSRL_AMP_JOINT_NAMES)}
    qpos = np.empty((frame_count, model.nq), dtype=np.float64)
    qpos[:, :3] = root_pos
    # AMP and MuJoCo both use scalar-first (w, x, y, z) quaternions here.
    qpos[:, 3:7] = root_rot
    for model_index, name in enumerate(model_joint_names):
        qpos[:, 7 + model_index] = dof_pos[:, source_indices[name]]
    return qpos


def _body_positions(
    model: mj.MjModel, qpos: np.ndarray, body_names: tuple[str, ...]
) -> np.ndarray:
    body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if any(body_id < 0 for body_id in body_ids):
        raise ValueError(f"URDF is missing one of the key bodies: {body_names}")
    data = mj.MjData(model)
    positions = np.empty((len(qpos), len(body_ids), 3), dtype=np.float32)
    for frame_index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mj.mj_forward(model, data)
        positions[frame_index] = data.xpos[body_ids]
    return positions


def reground_motion(source: Path, target: Path, urdf: Path) -> None:
    motion = joblib.load(source)
    if not isinstance(motion, dict):
        raise TypeError(f"{source}: expected a dictionary, got {type(motion)!r}")
    model = mj.MjModel.from_xml_path(str(urdf))
    if model.nq != 7 + len(BSRL_AMP_JOINT_NAMES):
        raise ValueError(f"{urdf}: unexpected nq={model.nq}")
    qpos = _build_qpos(model, motion, source)

    old_root_z = qpos[:, 2].copy()
    grounder = UrdfCollisionGrounder(model)
    pre_ground_min_z = np.array([grounder.min_z(frame) for frame in qpos])
    qpos[:, 2] -= pre_ground_min_z
    post_ground_min_z = np.array([grounder.min_z(frame) for frame in qpos])
    if np.max(np.abs(post_ground_min_z)) > 2.0e-6:
        raise RuntimeError(f"{source}: failed to place collision geometry on z=0")

    output = dict(motion)
    output["root_pos"] = qpos[:, :3].astype(np.float32)
    output["key_body_pos"] = _body_positions(model, qpos, BSRL_AMP_KEY_BODY_NAMES)
    output["dof_pos"] = np.asarray(motion["dof_pos"], dtype=np.float32)
    output["root_rot"] = np.asarray(motion["root_rot"], dtype=np.float32)
    if not all(
        np.isfinite(output[key]).all()
        for key in ("root_pos", "root_rot", "dof_pos", "key_body_pos")
    ):
        raise ValueError(f"{source}: regenerated motion contains non-finite values")

    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, target)
    root_delta = output["root_pos"][:, 2] - old_root_z
    print(
        f"[OK] {source.name}: {len(qpos)} frames -> {target} | "
        f"root-z delta {root_delta.min():+.6f}..{root_delta.max():+.6f} m | "
        f"ground residual {np.max(np.abs(post_ground_min_z)):.2e} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-ground existing BSRL AMP motions with the current URDF."
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=BSRL_URDF_PATH.with_name("export_floating.urdf"),
    )
    args = parser.parse_args()

    source_files = sorted(args.input_dir.glob("*.pkl"))
    if not source_files:
        raise FileNotFoundError(f"No PKL files found in {args.input_dir}")
    for source in source_files:
        reground_motion(source, args.output_dir / source.name, args.urdf)


if __name__ == "__main__":
    main()
