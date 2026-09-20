"""AMP locomotion configuration for the Men T1 25-DoF robot."""

import math
import os
from dataclasses import MISSING

import joblib
import numpy as np
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.men_t1 import T1_ACTION_SCALE, T1_CFG, T1_JOINT_NAMES
# Historical URDF-inset target clipping, retained for old-run replay:
# from legged_lab.assets.men_t1 import T1_ACTION_CLIP
from legged_lab.state_est import observations as state_est_obs
from legged_lab.tasks.locomotion.amp.amp_env_cfg import CommandsCfg, LocomotionAmpEnvCfg, ObservationsCfg
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


@configclass
class T1CommandsCfg(CommandsCfg):
    """T1 commands with persistent coverage of straight forward walking."""

    base_velocity = mdp.StraightMixtureVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        straight_command_probability=0.5,
        straight_lin_vel_x_range=(0.2, 0.8),
        single_axis_commands=False,
        motion_speed_animation=None,
        ranges=mdp.StraightMixtureVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
            heading=None,
        ),
    )


def _motion_weights() -> dict[str, float]:
    """Discover uploaded T1 AMP reference files without hard-coding names."""
    files = sorted(file_name for file_name in os.listdir(T1_MOTION_DIR) if file_name.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(
            "No T1 AMP reference motions were found. Upload .pkl files to "
            f"{T1_MOTION_DIR}; the required joint/key-body order is documented in that directory."
        )
    weights = {}
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
        weights[os.path.splitext(file_name)[0]] = float(frame_count)
    # Weight clips by frame count so every uploaded reference frame has the
    # same sampling probability. Equal per-file weights over-sampled the many
    # short turning clips and under-sampled the longer forward-walking clips.
    return weights


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
class T1ObservationsCfg(ObservationsCfg):
    """T1 AMP observations, including translational motion in the style signal."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Actor ablation: key-body positions remain privileged critic inputs and
        # part of the AMP discriminator, but are not exposed to the deployed policy.
        key_body_pos_b = None
        # Historical resetmatch baseline includes actor heading. The explicit
        # train.py --t1_no_actor_heading_baseline experiment disables only this
        # policy term, retaining the critic term and auditing the saved baseline.
        heading_error = ObsTerm(
            func=mdp.integrated_heading_error,
            params={"command_name": "base_velocity"},
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        heading_error = ObsTerm(
            func=mdp.integrated_heading_error,
            params={"command_name": "base_velocity"},
        )

    critic: CriticCfg = CriticCfg()

    @configclass
    class StateEstCriticCfg(ObsGroup):
        """Missing privileged targets appended only to estimator critics."""

        # The base critic already provides base_lin_vel_b(3). This group adds
        # the remaining value from the estimator's 4-D supervision target.
        base_height = ObsTerm(
            func=state_est_obs.base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # The former 8-D estimator privileges are intentionally disabled:
        # foot_contact = ObsTerm(func=state_est_obs.foot_contact, ...)
        # foot_height_w = ObsTerm(func=state_est_obs.foot_height_w, ...)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    state_est_critic: StateEstCriticCfg = StateEstCriticCfg()

    @configclass
    class RewardPrivilegedCfg(ObsGroup):
        """Simulator-only state needed to make contact rewards Markov for the critic."""

        foot_contact = ObsTerm(
            func=state_est_obs.foot_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True
                ),
                "force_threshold": 1.0,
            },
        )
        # T1 currently has no feet_air_time reward, so unlike BSRL it does not
        # need current_air_time/current_contact_time in its critic.
        # foot_mode_time = ObsTerm(...)
        foot_lin_vel_w_xy = ObsTerm(
            func=state_est_obs.body_lin_vel_w_xy,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True)
            },
        )
        undesired_contact_count = ObsTerm(
            func=state_est_obs.contact_violation_count,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!left_ankle_pitch_Link$)(?!right_ankle_pitch_Link$)"
                        r"(?!left_wrist_roll_Link$)(?!right_wrist_roll_Link$)"
                        r"(?!left_wrist_pitch_Link$)(?!right_wrist_pitch_Link$).+$"
                    ],
                ),
                "force_threshold": 1.0,
            },
        )
        base_contact = ObsTerm(
            func=state_est_obs.contact_violation_count,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_BODY_NAME]),
                "force_threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    reward_privileged: RewardPrivilegedCfg = RewardPrivilegedCfg()

    @configclass
    class StateEstLabelCfg(ObsGroup):
        """Simulator-only supervision for the deployable state estimator."""

        label = ObsTerm(
            func=state_est_obs.state_estimator_label,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Enabled only by ``train.py --use_estimator``.  Ordinary T1 AMP runs do
    # not compute simulator-only estimator labels or instantiate an estimator.
    state_est_label: StateEstLabelCfg | None = None

    @configclass
    class DiscriminatorCfg(ObservationsCfg.DiscriminatorCfg):
        # The base AMP configuration omits linear velocity. Without it, an
        # in-place gait is indistinguishable from a translating reference.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObservationsCfg.DiscriminatorDemoCfg):
        ref_root_lin_vel_b = ObsTerm(
            func=mdp.ref_root_lin_vel_b,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )
        ref_velocity_commands = ObsTerm(
            func=mdp.ref_velocity_commands,
            params={"animation": ANIMATION_TERM_NAME, "command_name": "base_velocity"},
        )

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class T1AmpRewardsCfg:
    """Velocity task rewards plus T1 penalties reused from wbt_est where applicable."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    zero_yaw_initial_heading_lin_vel_xy_exp = RewTerm(
        func=mdp.zero_yaw_initial_heading_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.4, "command_threshold": 1.0e-3},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_base_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Disabled for future runs: this zero-command penalty did not materially
    # improve straight walking and can over-constrain natural yaw corrections.
    # zero_command_yaw_rate_l2 = RewTerm(
    #     func=mdp.zero_command_yaw_rate_l2,
    #     weight=-2.0,
    #     params={"command_name": "base_velocity", "command_threshold": 0.1},
    # )
    # Disabled for future runs: the integrated heading target was not accurate
    # enough in practice and could reward corrections toward a drifting target.
    # Keep the term here for traceability instead of deleting it.
    # zero_command_heading_hold_exp = RewTerm(
    #     func=mdp.integrated_heading_tracking_exp,
    #     weight=0.5,
    #     params={"command_name": "base_velocity", "std": 0.15},
    # )
    # Keep a useful gradient when the exponential reward is near zero. This is
    # essential while the policy is still learning to translate instead of
    # settling into a high-style in-place gait.
    lin_vel_xy_error_excess_l2 = RewTerm(
        func=mdp.lin_vel_xy_base_frame_error_excess_l2,
        weight=-0.25,
        params={"command_name": "base_velocity", "threshold": 0.25, "max_excess": 2.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    # Historical AMP smoothness weights, retained for old-run comparison:
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # Match wbt_est T1 mimic for subsequent runs; existing processes are unchanged.
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_smoothness = RewTerm(func=mdp.ActionSecondOrderL2, weight=-1.0e-3)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        # Match wbt_est T1 mimic: penalize actual joint positions beyond soft limits.
        # Historical AMP value: -1.0. This is not a penalty on the policy target.
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(T1_JOINT_NAMES), preserve_order=True)},
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
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class T1CurriculumCfg:
    """Use a long forward-only warm-up before exposing the complete command set."""

    velocity_range_and_tracking_std = CurrTerm(
        func=mdp.velocity_range_and_tracking_std,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "steps_per_iteration": 24,
            "phase_boundaries": (8000, 16000),
            "lin_vel_x_ranges": ((0.2, 0.6), (-0.55, 0.6), (-0.7, 0.8)),
            "lin_vel_y_ranges": ((0.0, 0.0), (-0.45, 0.35), (-0.6, 0.5)),
            "ang_vel_z_ranges": ((0.0, 0.0), (-1.15, 1.2), (-2.05, 1.9)),
            "tracking_stds": (0.4, 0.4, 0.4),
        },
    )


@configclass
class T1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Primary T1 AMP task: command-matched references and integrated heading."""

    events: T1EventCfg = T1EventCfg()
    commands: T1CommandsCfg = T1CommandsCfg()
    observations: T1ObservationsCfg = T1ObservationsCfg()
    rewards: T1AmpRewardsCfg = T1AmpRewardsCfg()
    curriculum: T1CurriculumCfg = T1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 8192
        self.episode_length_s = 10.0

        self.actions.joint_pos.scale = T1_ACTION_SCALE
        # Historical 5%-inset URDF target clipping retained for comparison:
        # self.actions.joint_pos.clip = T1_ACTION_CLIP
        # Allow PD targets beyond joint limits; physical limits and actuator limits remain.
        self.actions.joint_pos.clip = None
        self.actions.joint_pos.joint_names = list(T1_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True

        self.motion_data.motion_dataset.motion_data_dir = T1_MOTION_DIR
        self.motion_data.motion_dataset.motion_data_weights = _motion_weights()
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.animation.animation.command_name = "base_velocity"
        self.animation.animation.command_motion_patterns = {
            "forward": ("model_walk*",),
            "backward": ("walk_back*",),
            "lateral_left": ("lateral_left*",),
            "lateral_right": ("lateral_right*",),
            "rotate_left": ("rotate_left*",),
            "rotate_right": ("rotate_right*",),
        }
        self.animation.animation.command_component_scales = (0.8, 0.6, 2.05)

        joint_cfg = SceneEntityCfg("robot", joint_names=list(T1_JOINT_NAMES), preserve_order=True)
        key_body_cfg = SceneEntityCfg("robot", body_names=list(T1_KEY_BODY_NAMES), preserve_order=True)
        for group_name in ("policy", "critic", "disc"):
            group = getattr(self.observations, group_name)
            group.joint_pos.params = {"asset_cfg": joint_cfg}
            group.joint_vel.params = {"asset_cfg": joint_cfg}
            if group.key_body_pos_b is not None:
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
        # Deployment commands yaw rate directly.  Do not silently replace the
        # sampled yaw-rate command with a closed-loop world-heading target.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # There is no standing clip in the current T1 AMP reference set.
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_BODY_NAME]
        self.terminations.base_height.params["minimum_height"] = 0.5
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(60.0)
