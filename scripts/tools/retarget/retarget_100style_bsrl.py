"""Retarget official Neutral 100STYLE BVHs with local GMR; isolated preview outputs.

The source is X-left/Y-up/Z-forward, with Y-X-Z BVH rotation channels.
Both positions and rotation bases are converted to X-forward/Y-left/Z-up.
This is kinematic IK, not a locomotion policy or a dynamic feasibility test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


AXIS_CHANGE = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
MOVEMENTS = ("FW", "BW", "SW", "FR", "BR", "SR", "ID", "TR1")
LABELS = dict(FW="Forward walk", BW="Backward walk", SW="Sidestep walk",
              FR="Forward run", BR="Backward run", SR="Sidestep run",
              ID="Idle", TR1="Transitions")


def convert_global_frames(quaternions, positions):
    """Convert centimeters and both world/local rotation bases, without reflecting."""
    positions = np.asarray(positions) @ AXIS_CHANGE.T * .01
    shape = quaternions.shape
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True).as_matrix()
    rotations = AXIS_CHANGE @ rotations @ AXIS_CHANGE.T
    quaternions = Rotation.from_matrix(rotations).as_quat(scalar_first=True).reshape(shape)
    return quaternions, positions


def source_motion(path, cuts_path):
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
    from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk

    text = path.read_text()
    match = re.search(r"Frame Time:\s*([\d.eE+-]+)", text)
    if match is None:
        raise ValueError(f"Missing Frame Time: {path}")
    frame_dt = float(match[1])
    motion_type = path.stem.split("_")[-1]
    with cuts_path.open(newline="") as source:
        cuts = next(row for row in csv.DictReader(source) if row["STYLE_NAME"] == "Neutral")
    start, stop = int(cuts[f"{motion_type}_START"]), int(cuts[f"{motion_type}_STOP"])
    # The vendor reader has unusual start/end semantics; read fully, then slice explicitly.
    animation = read_bvh(path)
    if not (0 <= start < stop <= len(animation.quats)):
        raise ValueError(f"Invalid official frame cuts {start}:{stop} for {path}")
    q, x = quat_fk(animation.quats[start:stop], animation.pos[start:stop], animation.parents)
    q, x = convert_global_frames(q, x)
    root_idx = animation.bones.index("Hips")
    root_matrix = Rotation.from_quat(q[0, root_idx], scalar_first=True).as_matrix()
    initial_yaw = np.arctan2(root_matrix[1, 0], root_matrix[0, 0])
    alignment = Rotation.from_euler("z", -initial_yaw)
    origin = x[0, root_idx].copy()
    origin[2] = 0.
    x = alignment.apply((x - origin).reshape(-1, 3)).reshape(x.shape)
    q = (alignment * Rotation.from_quat(q.reshape(-1, 4), scalar_first=True)).as_quat(
        scalar_first=True).reshape(q.shape)
    metadata = {"file": path.name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_frames": len(animation.quats), "start_frame": start,
                "stop_frame_exclusive": stop, "fps": 1. / frame_dt,
                "initial_heading_removed_rad": float(initial_yaw),
                "trim_convention": "official Frame_Cuts.csv interpreted as zero-based [START, STOP)"}
    return q, x, animation, metadata


def build_config(gmr_dir, animation, output_dir):
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT

    base = json.loads((gmr_dir / "general_motion_retargeting/ik_configs/bvh_lafan1_to_bsrl.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["bsrl"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    human_lengths, robot_lengths = [], []
    for side in ("Left", "Right"):
        human_lengths.append(sum(np.linalg.norm(animation.offsets[animation.bones.index(side + part)])
                                 for part in ("Knee", "Ankle")) * .01)
        hip, knee, ankle = [data.body(f"link_{side.lower()}_{part}").xpos.copy()
                            for part in ("thigh", "shank", "ankle_roll")]
        robot_lengths.append(np.linalg.norm(knee-hip) + np.linalg.norm(ankle-knee))
    scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
    name_map = {"Hips": "Hips", "LeftUpLeg": "LeftHip", "RightUpLeg": "RightHip",
                "LeftLeg": "LeftKnee", "RightLeg": "RightKnee",
                "LeftFootMod": "LeftFootMod", "RightFootMod": "RightFootMod"}
    base["human_scale_table"] = {new: scale for new in name_map.values()}
    # In 100STYLE all neutral bone bases are aligned; LAFAN-specific bone offsets do not apply.
    # Hip rotations belong to the thigh (after hip pitch), not the upstream hip-roll link.
    hip_center = .5 * (data.body("link_left_thigh").xpos + data.body("link_right_thigh").xpos)
    root_offset = (data.body("base_link").xpos - hip_center).tolist()
    for table in ("ik_match_table1", "ik_match_table2"):
        mapped = {}
        for body, entry in base[table].items():
            entry = list(entry)
            entry[0] = name_map[entry[0]]
            entry[3], entry[4] = [0., 0., 0.], [1., 0., 0., 0.]
            if body == "base_link":
                entry[3] = root_offset
            body = body.replace("link_left_hip_roll", "link_left_thigh").replace(
                "link_right_hip_roll", "link_right_thigh")
            mapped[body] = entry
        base[table] = mapped
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "bvh_100style_to_bsrl.json"
    contents = json.dumps(base, indent=2) + "\n"
    if config_path.exists() and config_path.read_text() != contents:
        raise ValueError(f"Refusing to replace different config: {config_path}")
    if not config_path.exists():
        config_path.write_text(contents)
    return config_path, scale


def motion_statistics(qpos, fps, model, foot_clearance, source_q, source_x, source_bones, scale):
    rotation = Rotation.from_quat(qpos[:, 3:7], scalar_first=True)
    matrices = rotation.as_matrix()
    yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    velocity = np.gradient(qpos[:, :3], 1./fps, axis=0)
    body_velocity = rotation.inv().apply(velocity)
    root_index = source_bones.index("Hips")
    source_rotation = Rotation.from_quat(source_q[:, root_index], scalar_first=True)
    source_velocity = source_rotation.inv().apply(np.gradient(source_x[:, root_index], 1./fps, axis=0)) * scale
    scalar_joints = np.flatnonzero(model.jnt_type != 0)
    lows = model.jnt_range[scalar_joints, 0]
    highs = model.jnt_range[scalar_joints, 1]
    qj = qpos[:, model.jnt_qposadr[scalar_joints]]
    dqj = np.gradient(qj, 1./fps, axis=0)
    margin = np.minimum(qj - lows, highs - qj)
    return {
        "frames": len(qpos), "fps": fps, "duration_s": (len(qpos)-1)/fps,
        "finite": bool(np.isfinite(qpos).all()), "joint_limit_violation_rad": float(max(0., -margin.min())),
        "joint_velocity_max_abs_rad_s": float(np.abs(dqj).max()),
        "near_limit_frame_fraction": float(np.mean(np.any(margin < .01, axis=1))),
        "base_height_min_max_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
        "body_velocity_median_m_s": np.median(body_velocity, axis=0).tolist(),
        "source_scaled_body_velocity_median_m_s": np.median(source_velocity, axis=0).tolist(),
        "net_yaw_deg": float(np.rad2deg(yaw[-1]-yaw[0])),
        "mean_abs_yaw_rate_rad_s": float(np.abs(np.gradient(yaw, 1./fps)).mean()),
        "sole_clearance_min_p50_p95_max_m": np.quantile(foot_clearance, [0., .5, .95, 1.]).tolist(),
        "grounding": "One constant vertical translation for the entire clip; no per-frame foot pinning.",
        "warning": "Kinematic preview only. No dynamic balance/contact-force validation; not installed in AMP data.",
    }


def retarget_file(path, args):
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.grounding import RobotGrounder
    from general_motion_retargeting.params import (
        BSRL_JOINT_COORDINATE_CONVENTION, BSRL_KEY_BODY_NAMES, IK_CONFIG_DICT,
    )
    from general_motion_retargeting.utils.mujoco_model import (
        body_positions, joint_names_in_qpos_order, local_body_positions,
    )

    gmr_path = args.output_dir / "gmr" / (path.stem + ".pkl")
    if gmr_path.exists():
        print(f"[SKIP] Existing completed motion: {gmr_path}", flush=True)
        return
    q, x, animation, source = source_motion(path, args.input_dir / "Frame_Cuts.csv")
    config_path, scale = build_config(args.gmr_dir, animation, args.output_dir)
    IK_CONFIG_DICT.setdefault("bvh_100style", {})["bsrl"] = config_path
    retargeter = GeneralMotionRetargeting("bvh_100style", "bsrl", verbose=False)
    grounder = RobotGrounder("bsrl")
    source_names = ("Hips", "LeftHip", "RightHip", "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle")
    indices = [animation.bones.index(name) for name in source_names]
    target_names = source_names[:5] + ("LeftFootMod", "RightFootMod")
    count = len(q) if args.max_frames is None else min(len(q), args.max_frames)
    q, x = q[:count], x[:count]
    qpos = np.empty((count, retargeter.model.nq))
    started = time.monotonic()
    for frame_idx in range(count):
        def frame():
            # GMR mutates its input dictionary, so provide fresh arrays, including during warmup.
            return {name: [x[frame_idx, index].copy(), q[frame_idx, index].copy()]
                    for name, index in zip(target_names, indices)}
        if frame_idx == 0:
            for _ in range(20):
                retargeter.retarget(frame())
        qpos[frame_idx] = retargeter.retarget(frame())
        if (frame_idx+1) % 300 == 0:
            print(f"[{path.stem}] IK {frame_idx+1}/{count}, {(frame_idx+1)/(time.monotonic()-started):.1f} fps", flush=True)
    if not np.isfinite(qpos).all():
        raise ValueError(f"Nonfinite IK result: {path}")
    raw_clearance = np.array([grounder.qpos_min_z(frame) for frame in qpos])
    vertical_shift = -float(raw_clearance.min())
    qpos[:, 2] += vertical_shift
    world_body_pos, body_names = body_positions(retargeter.model, qpos)
    local_body_pos, local_names = local_body_positions(retargeter.model, qpos)
    if body_names != local_names:
        raise ValueError("Inconsistent body ordering")
    key_indices = [body_names.index(name) for name in BSRL_KEY_BODY_NAMES]
    motion = {
        "fps": source["fps"], "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]], "dof_pos": qpos[:, 7:],
        "dof_names": joint_names_in_qpos_order(retargeter.model),
        "local_body_pos": local_body_pos, "link_body_list": body_names,
        "key_body_pos": world_body_pos[:, key_indices], "key_body_names": list(BSRL_KEY_BODY_NAMES),
        "source_motion": path.name, "source_metadata": source,
        "joint_coordinate_convention": BSRL_JOINT_COORDINATE_CONVENTION,
        "grounding_convention": "bsrl_constant_collision_preview_v1",
        "vertical_shift_m": vertical_shift, "human_to_robot_scale": scale,
        "kinematic_preview_only": True,
    }
    stats = motion_statistics(qpos, source["fps"], retargeter.model, raw_clearance+vertical_shift,
                              q, x, animation.bones, scale)
    stats.update(source=source, scale=scale, vertical_shift_m=vertical_shift,
                 ik_config=str(config_path), label=LABELS[path.stem.split("_")[-1]])
    for folder in ("gmr", "source_fk", "metrics"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    with gmr_path.open("xb") as output:
        pickle.dump(motion, output, protocol=4)
    np.savez_compressed(args.output_dir / "source_fk" / (path.stem + ".npz"),
                        positions=x * scale + [0., 0., vertical_shift],
                        bones=np.array(animation.bones), parents=animation.parents,
                        source_floor_translation=vertical_shift)
    (args.output_dir / "metrics" / (path.stem + ".json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[DONE] {path.stem}: {stats['duration_s']:.1f}s, limit violation {stats['joint_limit_violation_rad']:.3g} rad", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmr-dir", type=Path, default=Path("/root/mjw/GMR"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--movements", nargs="+", choices=MOVEMENTS, default=list(MOVEMENTS))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.gmr_dir))
    for movement in args.movements:
        retarget_file(args.input_dir / f"Neutral_{movement}.bvh", args)


if __name__ == "__main__":
    main()
