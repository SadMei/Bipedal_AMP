import argparse
import math
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Minimal BSRL foot contact-force smoke test.")
parser.add_argument(
    "--disable_self_collisions",
    action="store_true",
    help="Disable articulation self-collisions for an A/B contact comparison.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli, fast_shutdown=True)
simulation_app = app_launcher.app

import torch

import isaaclab.utils.version as isaaclab_version

if not hasattr(isaaclab_version, "get_isaac_sim_version"):
    isaaclab_version.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from legged_lab.assets.bsrl import BSRL_CFG, BSRL_DEFAULT_ROOT_HEIGHT


FOOT_NAMES = ["link_left_ankle_roll", "link_right_ankle_roll"]
ROOT_HEIGHTS = [
    BSRL_DEFAULT_ROOT_HEIGHT + 0.02,
    BSRL_DEFAULT_ROOT_HEIGHT,
    BSRL_DEFAULT_ROOT_HEIGHT - 0.02,
]

ROBOT_CFG = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
ROBOT_CFG.spawn.articulation_props.enabled_self_collisions = not args_cli.disable_self_collisions


@configclass
class BSRLContactSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=5, track_air_time=True)
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


def reset_robot(scene: InteractiveScene, root_height: float):
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = scene.env_origins
    root_state[:, 2] += root_height
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=scene.device)
    root_state[:, 7:13] = 0.0

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.set_joint_velocity_target(joint_vel)
    scene.write_data_to_sim()


def sample_forces(sim: sim_utils.SimulationContext, scene: InteractiveScene, root_height: float):
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    dt = sim.get_physics_dt()

    reset_robot(scene, root_height)
    sim.step()
    scene.update(dt)

    body_names = list(robot.body_names)
    sensor_body_names = list(contact_sensor.body_names)
    foot_body_ids = [body_names.index(name) for name in FOOT_NAMES]
    foot_sensor_ids = [sensor_body_names.index(name) for name in FOOT_NAMES]

    force_rows = []
    height_rows = []
    for step in range(360):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        if step >= 180:
            forces = contact_sensor.data.net_forces_w[0, foot_sensor_ids, :].detach().cpu()
            heights = robot.data.body_pos_w[0, foot_body_ids, 2].detach().cpu()
            force_rows.append(forces)
            height_rows.append(heights)

    forces = torch.stack(force_rows, dim=0)
    heights = torch.stack(height_rows, dim=0)
    norms = torch.linalg.norm(forces, dim=-1)
    vertical = forces[:, :, 2]
    total_mass = robot.data.default_mass[0].sum().item()
    total_weight = total_mass * 9.81
    base_z = robot.data.root_pos_w[0, 2].item()

    return {
        "root_height": root_height,
        "body_names": body_names,
        "sensor_body_names": sensor_body_names,
        "foot_sensor_ids": foot_sensor_ids,
        "base_z": base_z,
        "total_mass": total_mass,
        "total_weight": total_weight,
        "mean_forces": forces.mean(dim=0),
        "mean_norms": norms.mean(dim=0),
        "mean_vertical": vertical.mean(dim=0),
        "max_norms": norms.max(dim=0).values,
        "mean_heights": heights.mean(dim=0),
    }


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(BSRLContactSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    print("[bsrl_contact_check] robot bodies:")
    for i, name in enumerate(scene["robot"].body_names):
        print(f"[bsrl_contact_check]   body[{i:02d}]={name}")
    print("[bsrl_contact_check] contact sensor bodies:")
    for i, name in enumerate(scene["contact_forces"].body_names):
        print(f"[bsrl_contact_check]   sensor_body[{i:02d}]={name}")

    for root_height in ROOT_HEIGHTS:
        result = sample_forces(sim, scene, root_height)
        left_force = result["mean_forces"][0].tolist()
        right_force = result["mean_forces"][1].tolist()
        left_norm = result["mean_norms"][0].item()
        right_norm = result["mean_norms"][1].item()
        left_fz = result["mean_vertical"][0].item()
        right_fz = result["mean_vertical"][1].item()
        total_fz = left_fz + right_fz
        symmetry = abs(left_fz - right_fz) / max(abs(total_fz), 1.0)

        print(f"[bsrl_contact_check] root_height={root_height:.3f} final_base_z={result['base_z']:.4f}")
        print(
            "[bsrl_contact_check]   "
            f"left_force_xyz={left_force} right_force_xyz={right_force}"
        )
        print(
            "[bsrl_contact_check]   "
            f"left_norm={left_norm:.3f} right_norm={right_norm:.3f} "
            f"left_fz={left_fz:.3f} right_fz={right_fz:.3f} "
            f"sum_fz={total_fz:.3f} weight={result['total_weight']:.3f} "
            f"fz_symmetry_error={symmetry:.3f}"
        )
        print(
            "[bsrl_contact_check]   "
            f"left_foot_z={result['mean_heights'][0].item():.4f} "
            f"right_foot_z={result['mean_heights'][1].item():.4f}"
        )

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
