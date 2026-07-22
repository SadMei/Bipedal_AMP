import math
import os

import isaaclab.envs.mdp as il_mdp
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from legged_lab.assets.bsrl import BSRL_CFG # 引入你的机器人本体配置
import legged_lab.tasks.locomotion.amp.mdp as amp_mdp

LEGGED_LAB_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4
BASE_LINK_NAME = "base_link"
FOOT_NAMES = [
    "link_left_ankle_roll",
    "link_right_ankle_roll",
]
FOOT_REGEX = "link_.*_ankle_roll"
JOINT_NAMES = [
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

@configclass
class BSRLAmpEnvCfg(LocomotionAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 将场景中的机器人替换为你的 BSRL 配置
        self.scene.robot = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- 以下是基础的配置调整，你需要根据你的机器人进行修改 ---

        # 1. 调整控制步长 (默认可能是控制 50Hz, 仿真 200Hz)
        self.decimation = 4

        # 2. 初始状态 (你可以在资产里配，也可以在这里覆盖)
        # self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)

        # 3. 动作控制
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.preserve_order = True

        # 运动数据
        # Use motions retargeted directly to BSRL's 12 joints.
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "bsrl_12dof", "amp", "walk_and_run"
        )
        motion_files = sorted(
            file_name for file_name in os.listdir(self.motion_data.motion_dataset.motion_data_dir) if file_name.endswith(".pkl")
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            os.path.splitext(file_name)[0]: 1.0 for file_name in motion_files
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # 4. 观测空间 (AMP 核心)
        # 你的机器人只有下肢，所以关键部位(Key Body)只有脚踝（或脚底）。
        # 这些名字必须和你的 URDF/USD 里的 link 名字完全匹配！
        KEY_BODY_NAMES = [
            "link_right_ankle_roll",
            "link_left_ankle_roll",
        ]
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        joint_obs_asset_cfg = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
        self.observations.policy.joint_pos.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.policy.joint_vel.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.critic.joint_pos.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.critic.joint_vel.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.disc.joint_pos.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.disc.joint_vel.params = {"asset_cfg": joint_obs_asset_cfg}
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME
        self.animation.animation.motion_data_components[6] = "key_body_pos_b"


        # Commands: match the working PPO setup.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.rel_standing_envs = 0.02

        # Rewards: borrow the task-side shaping from the working PPO setup,
        # but keep AMP-specific style learning untouched.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = amp_mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = amp_mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2 = RewTerm(
            func=il_mdp.base_height_l2,
            weight=0.0,
            params={
                "target_height": 0.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
                "sensor_cfg": None,
            },
        )
        self.rewards.body_lin_acc_l2 = RewTerm(
            func=il_mdp.body_lin_acc_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME])},
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.joint_vel_l2 = RewTerm(
            func=il_mdp.joint_vel_l2,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits = RewTerm(
            func=il_mdp.joint_vel_limits,
            weight=0.0,
            params={
                "soft_ratio": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            },
        )
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.func = amp_mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
            "threshold": 0.5,
        }
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{FOOT_REGEX}).*"]
        self.rewards.contact_forces = RewTerm(
            func=il_mdp.contact_forces,
            weight=0.0,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[FOOT_REGEX]),
            },
        )
        self.rewards.termination_penalty = RewTerm(func=amp_mdp.is_terminated, weight=-200.0)
        self.rewards.joint_deviation_hip_l1 = RewTerm(
            func=amp_mdp.joint_deviation_l1,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
        )
        self.rewards.joint_pos_penalty = RewTerm(
            func=amp_mdp.joint_deviation_l1,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        )
        self.rewards.feet_slide = RewTerm(
            func=amp_mdp.feet_slide,
            weight=-0.6,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
            },
        )

        # Events
        self.events.physics_material.params["asset_cfg"].body_names = [f"^(?!.*{FOOT_REGEX}).*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = [BASE_LINK_NAME]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_LINK_NAME]
        self.events.randomize_foot_rigid_body_material = EventTerm(
            func=amp_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True),
                "static_friction_range": (1.2, 1.2),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            },
        )
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "height_offset": 0.0,
        }

        # 终止条件: mirror the PPO BSRL setup.
        self.terminations.base_height.func = amp_mdp.base_height_below_minimum
        self.terminations.base_height.params["minimum_height"] = 0.4
        # 身体基础如果碰到地面，认为跌倒
        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_LINK_NAME]


@configclass
class BSRLAmpEnvCfgPlay(BSRLAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 播放模式下，往往环境数量为 1
        # self.viewer.eye = (2.0, 2.0, 1.0)
        pass
