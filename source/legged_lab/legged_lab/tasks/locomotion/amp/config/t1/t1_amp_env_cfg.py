"""AMP locomotion configuration for the Men T1 25-DoF robot."""

import math
import os
from dataclasses import MISSING

import isaaclab.envs.mdp as il_mdp
import joblib
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.men_t1 import T1_ACTION_SCALE, T1_CFG, T1_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
import legged_lab.tasks.locomotion.amp.mdp as mdp


ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
BASE_BODY_NAME = "pelvis_Link"
FOOT_BODY_NAMES = ("left_ankle_pitch_Link", "right_ankle_pitch_Link")
T1_KEY_BODY_NAMES = (
    "left_ankle_pitch_Link",
    "right_ankle_pitch_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "left_shoulder_roll_Link",
    "right_shoulder_roll_Link",
)
T1_MOTION_DIR = os.path.join(
    LEGGED_LAB_ROOT_DIR, "data", "MotionData", "t1_25dof", "amp", "walk_and_run_amp_ready"
)

PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}


def _motion_weights() -> dict[str, float]:
    """Discover uploaded T1 AMP reference files without hard-coding names."""
    files = sorted(file_name for file_name in os.listdir(T1_MOTION_DIR) if file_name.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(
            "No T1 AMP reference motions were found. Upload .pkl files to "
            f"{T1_MOTION_DIR}; the required joint/key-body order is documented in that directory."
        )
    for file_name in files:
        path = os.path.join(T1_MOTION_DIR, file_name)
        motion = joblib.load(path)
        if not isinstance(motion, dict):
            raise ValueError(f"{path}: expected a motion dictionary")
        if tuple(motion.get("joint_names", ())) != tuple(T1_JOINT_NAMES):
            raise ValueError(f"{path}: AMP joint order does not match the T1 policy order")
        if tuple(motion.get("key_body_names", ())) != T1_KEY_BODY_NAMES:
            raise ValueError(f"{path}: AMP key-body order does not match T1_KEY_BODY_NAMES")
        frame_count = len(motion.get("root_pos", ()))
        expected_shapes = {
            "root_pos": (frame_count, 3),
            "root_rot": (frame_count, 4),
            "dof_pos": (frame_count, len(T1_JOINT_NAMES)),
            "key_body_pos": (frame_count, len(T1_KEY_BODY_NAMES), 3),
        }
        if frame_count < AMP_NUM_STEPS + 1:
            raise ValueError(f"{path}: at least {AMP_NUM_STEPS + 1} frames are required")
        for key, expected_shape in expected_shapes.items():
            values = np.asarray(motion.get(key))
            if values.shape != expected_shape:
                raise ValueError(f"{path}: expected {key} shape {expected_shape}, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {key} contains NaN or Inf")
        quat_norm = np.linalg.norm(np.asarray(motion["root_rot"]), axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
            raise ValueError(f"{path}: root_rot contains non-unit quaternions")
    return {os.path.splitext(file_name)[0]: 1.0 for file_name in files}


def joint_mechanical_power_abs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sum absolute mechanical power over selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


@configclass
class T1EventCfg:
    """T1 domain randomization copied from the current wbt_est training setup."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "force_range": (-15.0, 15.0),
            "torque_range": (-4.0, 4.0),
        },
    )
    reset_from_ref = EventTerm(func=mdp.reset_from_ref, mode="reset", params=MISSING)
    reset_to_default = None
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 5.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class T1AmpRewardsCfg:
    """Velocity task rewards plus T1 penalties reused from wbt_est where applicable."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    # The tracking task in wbt_est has several dense reference rewards that offset its
    # stronger smoothness costs. AMP does not, so keep these two costs at the native AMP
    # scale; otherwise they dominate both the velocity and style rewards at initialization.
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(T1_JOINT_NAMES), preserve_order=True)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_pitch_Link$)(?!right_ankle_pitch_Link$)"
                    r"(?!left_wrist_roll_Link$)(?!right_wrist_roll_Link$)"
                    r"(?!left_wrist_pitch_Link$)(?!right_wrist_pitch_Link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    arm_energy = RewTerm(
        func=joint_mechanical_power_abs,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class T1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Training configuration isolated from the existing G1 and BSRL tasks."""

    events: T1EventCfg = T1EventCfg()
    rewards: T1AmpRewardsCfg = T1AmpRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 8192
        self.episode_length_s = 10.0

        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.actions.joint_pos.joint_names = list(T1_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True

        self.motion_data.motion_dataset.motion_data_dir = T1_MOTION_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights()
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        joint_cfg = SceneEntityCfg("robot", joint_names=list(T1_JOINT_NAMES), preserve_order=True)
        key_body_cfg = SceneEntityCfg("robot", body_names=list(T1_KEY_BODY_NAMES), preserve_order=True)
        for group_name in ("policy", "critic", "disc"):
            group = getattr(self.observations, group_name)
            group.joint_pos.params = {"asset_cfg": joint_cfg}
            group.joint_vel.params = {"asset_cfg": joint_cfg}
            group.key_body_pos_b.params = {"asset_cfg": key_body_cfg}

        self.observations.disc.history_length = AMP_NUM_STEPS
        self.terminal_obs_groups = ("disc",)
        for term_name in (
            "ref_root_local_rot_tan_norm",
            "ref_root_ang_vel_b",
            "ref_joint_pos",
            "ref_joint_vel",
            "ref_key_body_pos_b",
        ):
            getattr(self.observations.disc_demo, term_name).params["animation"] = ANIMATION_TERM_NAME

        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": joint_cfg,
            "height_offset": 0.0,
        }

        # Match the command support to the uploaded T1 clips. Asking for G1's 3 m/s
        # running command while only providing <=1.1 m/s T1 demonstrations makes the
        # task and discriminator objectives contradictory.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.55, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_BODY_NAME]
        self.terminations.base_height.params["minimum_height"] = 0.5
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(60.0)


@configclass
class T1AmpEnvCfgPlay(T1AmpEnvCfg):
    """Deterministic single-environment T1 playback configuration."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        for event_name in (
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "actuator_gains",
            "body_mass",
            "joint_params",
            "joint_friction",
            "base_external_force_torque",
            "push_robot",
        ):
            setattr(self.events, event_name, None)

        self.events.reset_from_ref = None
        self.events.reset_to_default = EventTerm(
            func=il_mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
