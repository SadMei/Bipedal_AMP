"""Check BSRL self-collisions while sweeping one joint at a time to its hard limits.

The test has no terrain or gravity and uses a fixed base.  Each lower/upper sweep
starts from the asset default pose, while every other joint remains at default.
This isolates articulation self-collisions from falls and ground contacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=21, help="Samples from the default angle to each hard limit.")
parser.add_argument("--steps_per_sample", type=int, default=2, help="Physics steps used to expose contact per sample.")
parser.add_argument("--force_threshold", type=float, default=1.0, help="Contact-force threshold in newtons.")
parser.add_argument("--output", type=str, default=None, help="Optional JSON report path.")
parser.add_argument(
    "--ignore_base_collision",
    action="store_true",
    help="Use a temporary URDF without the base collision to isolate the remaining joint-limit tests.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli, fast_shutdown=True)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.version as isaaclab_version
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from legged_lab.assets.bsrl import BSRL_CFG


if not hasattr(isaaclab_version, "get_isaac_sim_version"):
    isaaclab_version.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)


JOINT_NAMES = (
    "joint_right_hip_yaw",
    "joint_right_hip_roll",
    "joint_right_hip_pitch",
    "joint_right_knee_pitch",
    "joint_right_ankle_pitch",
    "joint_right_ankle_roll",
    "joint_left_hip_yaw",
    "joint_left_hip_roll",
    "joint_left_hip_pitch",
    "joint_left_knee_pitch",
    "joint_left_ankle_pitch",
    "joint_left_ankle_roll",
)


ROBOT_CFG = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
ROBOT_CFG.spawn.fix_base = True
ROBOT_CFG.spawn.rigid_props.disable_gravity = True
ROBOT_CFG.spawn.articulation_props.enabled_self_collisions = True

# A base mesh on a dynamic articulation is convexified by PhysX.  If its
# convex hull overlaps the hip assembly at the default pose, it masks every
# other limit test.  This opt-in diagnostic copy removes only that collider;
# the canonical URDF and all other collisions remain untouched.
_temporary_urdf_dir = None
if args_cli.ignore_base_collision:
    source_urdf = Path(ROBOT_CFG.spawn.asset_path).resolve()
    _temporary_urdf_dir = tempfile.TemporaryDirectory(prefix="bsrl_joint_limit_no_base_")
    temporary_root = Path(_temporary_urdf_dir.name)
    temporary_urdf_dir = temporary_root / "urdf"
    temporary_urdf_dir.mkdir(parents=True)
    os.symlink(source_urdf.parent.parent / "meshes", temporary_root / "meshes", target_is_directory=True)
    tree = ET.parse(source_urdf)
    base_link = tree.getroot().find("./link[@name='base_link']")
    if base_link is None:
        raise RuntimeError("Could not find base_link in BSRL URDF")
    for collision in list(base_link.findall("collision")):
        base_link.remove(collision)
    temporary_urdf = temporary_urdf_dir / source_urdf.name
    tree.write(temporary_urdf, encoding="utf-8", xml_declaration=True)
    ROBOT_CFG.spawn.asset_path = str(temporary_urdf)


@configclass
class BSRLJointLimitCollisionSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ROBOT_CFG
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )


def _evaluate_pose(sim, scene, joint_pos: torch.Tensor) -> tuple[float, list[dict]]:
    robot = scene["robot"]
    sensor = scene["contact_forces"]
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)
    scene.write_data_to_sim()
    sensor.reset()

    peak_by_body = torch.zeros(len(sensor.body_names), device=scene.device)
    for _ in range(args_cli.steps_per_sample):
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        force_norm = torch.linalg.vector_norm(sensor.data.net_forces_w[0], dim=-1)
        peak_by_body = torch.maximum(peak_by_body, force_norm)

    active_ids = torch.nonzero(peak_by_body > args_cli.force_threshold).flatten().tolist()
    active = [
        {"body": sensor.body_names[index], "force_n": float(peak_by_body[index].item())}
        for index in active_ids
    ]
    return float(peak_by_body.max().item()), active


def main() -> None:
    if args_cli.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args_cli.steps_per_sample < 1:
        raise ValueError("--steps_per_sample must be at least 1")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(BSRLJointLimitCollisionSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    default_pos = robot.data.default_joint_pos.clone()
    joint_ids, resolved_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(resolved_names) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint order: {resolved_names}")
    hard_limits = robot.data.joint_pos_limits[0, joint_ids].detach().cpu()

    baseline_force, baseline_bodies = _evaluate_pose(sim, scene, default_pos.clone())
    report = {
        "self_collisions_enabled": bool(ROBOT_CFG.spawn.articulation_props.enabled_self_collisions),
        "base_collision_ignored": args_cli.ignore_base_collision,
        "force_threshold_n": args_cli.force_threshold,
        "samples_per_sweep": args_cli.samples,
        "baseline": {"max_force_n": baseline_force, "contact_bodies": baseline_bodies},
        "tests": [],
    }
    print(
        f"[baseline] max_force={baseline_force:.6f} N "
        f"contacts={[item['body'] for item in baseline_bodies]}",
        flush=True,
    )

    for joint_id, joint_name, limits in zip(joint_ids, resolved_names, hard_limits):
        default_angle = float(default_pos[0, joint_id].item())
        for bound_name, limit in (("lower", float(limits[0].item())), ("upper", float(limits[1].item()))):
            first_contact = None
            max_force = 0.0
            max_force_bodies: list[dict] = []

            # Every direction starts from the complete default pose.  Each
            # sample is then written directly so prior contact impulses cannot
            # move another joint and contaminate the sweep.
            _evaluate_pose(sim, scene, default_pos.clone())
            for sample_index, alpha in enumerate(torch.linspace(0.0, 1.0, args_cli.samples).tolist()):
                pose = default_pos.clone()
                angle = default_angle + alpha * (limit - default_angle)
                pose[0, joint_id] = angle
                force, bodies = _evaluate_pose(sim, scene, pose)
                if force > max_force:
                    max_force = force
                    max_force_bodies = bodies
                if bodies and first_contact is None:
                    first_contact = {
                        "sample": sample_index,
                        "alpha": alpha,
                        "angle_rad": angle,
                        "max_force_n": force,
                        "contact_bodies": bodies,
                    }

            result = {
                "joint": joint_name,
                "bound": bound_name,
                "default_rad": default_angle,
                "limit_rad": limit,
                "collision": first_contact is not None,
                "first_contact": first_contact,
                "max_force_n": max_force,
                "max_force_bodies": max_force_bodies,
            }
            report["tests"].append(result)
            status = "COLLISION" if result["collision"] else "clear"
            detail = ""
            if first_contact is not None:
                detail = (
                    f" first_angle={first_contact['angle_rad']:.6f}"
                    f" bodies={[item['body'] for item in first_contact['contact_bodies']]}"
                )
            print(
                f"[{status}] {joint_name:30s} {bound_name:5s} limit={limit:+.6f}"
                f" max_force={max_force:.6f} N{detail}",
                flush=True,
            )

    collision_count = sum(test["collision"] for test in report["tests"])
    print(f"[summary] collisions={collision_count}/{len(report['tests'])}", flush=True)
    if args_cli.output:
        output_path = Path(args_cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[summary] report={output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
