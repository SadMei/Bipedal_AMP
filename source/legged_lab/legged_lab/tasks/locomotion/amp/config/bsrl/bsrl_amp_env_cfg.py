import math
import os
from dataclasses import MISSING

import joblib
import isaaclab.envs.mdp as il_mdp
import numpy as np
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_lab.assets.bsrl import BSRL_ACTION_SCALE, BSRL_CFG # 引入你的机器人本体配置
from legged_lab.assets.bsrl_conventions import (
    BSRL_AMP_JOINT_NAMES,
    BSRL_AMP_KEY_BODY_NAMES,
    BSRL_GROUNDING_CONVENTION,
    BSRL_JOINT_COORDINATE_CONVENTION,
    validate_bsrl_joint_positions,
)
import legged_lab.tasks.locomotion.amp.mdp as amp_mdp

LEGGED_LAB_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

# AMP 动作序列名称及判别器使用的连续帧数。
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4

# BSRL 模型中的关键刚体、足部和关节名称。
BASE_LINK_NAME = "base_link"
FOOT_NAMES = [
    "link_left_ankle_roll",
    "link_right_ankle_roll",
]
KEY_BODY_NAMES = list(BSRL_AMP_KEY_BODY_NAMES)
FOOT_REGEX = "link_.*_ankle_roll"
JOINT_NAMES = list(BSRL_AMP_JOINT_NAMES)


def _validate_motion_files(motion_dir: str, motion_files: list[str]) -> None:
    for file_name in motion_files:
        path = os.path.join(motion_dir, file_name)
        motion = joblib.load(path)
        convention = motion.get("joint_coordinate_convention")
        if convention != BSRL_JOINT_COORDINATE_CONVENTION:
            raise ValueError(
                f"{path}: expected joint coordinate convention "
                f"{BSRL_JOINT_COORDINATE_CONVENTION!r}, got {convention!r}"
            )
        grounding = motion.get("grounding_convention")
        if grounding != BSRL_GROUNDING_CONVENTION:
            raise ValueError(
                f"{path}: expected grounding convention {BSRL_GROUNDING_CONVENTION!r}, got {grounding!r}"
            )
        if tuple(motion.get("joint_names", ())) != BSRL_AMP_JOINT_NAMES:
            raise ValueError(f"{path}: AMP joint order does not match BSRL_AMP_JOINT_NAMES")
        if tuple(motion.get("key_body_names", ())) != BSRL_AMP_KEY_BODY_NAMES:
            raise ValueError(f"{path}: AMP key-body order does not match BSRL_AMP_KEY_BODY_NAMES")

        frame_count = len(motion["root_pos"])
        expected_shapes = {
            "root_pos": (frame_count, 3),
            "root_rot": (frame_count, 4),
            "dof_pos": (frame_count, len(BSRL_AMP_JOINT_NAMES)),
            "key_body_pos": (frame_count, len(BSRL_AMP_KEY_BODY_NAMES), 3),
        }
        for key, expected_shape in expected_shapes.items():
            values = np.asarray(motion[key])
            if values.shape != expected_shape:
                raise ValueError(f"{path}: expected {key} shape {expected_shape}, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {key} contains non-finite values")

        quat_norm = np.linalg.norm(np.asarray(motion["root_rot"]), axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
            raise ValueError(f"{path}: root_rot contains non-unit quaternions")
        validate_bsrl_joint_positions(motion["dof_pos"], motion["joint_names"], path)

# 周期推扰时叠加到根节点的速度范围：x/y/z 为线速度，roll/pitch/yaw 为角速度。
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}


@configclass
class BSRLObservationsCfg:
    """BSRL AMP 的观测组配置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测：仅包含部署时需要提供给 actor 的信息。"""

        # 基座角速度，并加入 IMU 测量噪声。
        base_ang_vel = ObsTerm(func=amp_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 基座局部旋转的 6D 切向归一化表示。
        root_local_rot_tan_norm = ObsTerm(
            func=amp_mdp.root_local_rot_tan_norm,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 当前期望的前后、横向和偏航速度指令。
        velocity_commands = ObsTerm(
            func=amp_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # 12 个关节的位置和速度，顺序严格采用 JOINT_NAMES。
        joint_pos = ObsTerm(
            func=amp_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=amp_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # 上一个控制周期输出的动作。
        actions = ObsTerm(func=amp_mdp.last_action)
        # 左右脚相对基座的位置，用于描述下肢姿态。
        key_body_pos_b = ObsTerm(
            func=amp_mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.08, n_max=0.08),
        )

        def __post_init__(self):
            # 保存最近 5 个控制周期，并对策略观测启用噪声。
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """价值网络观测：可包含仿真中可得的特权信息，不输入 actor。"""

        # critic 比 policy 多使用无噪声的基座线速度。
        base_lin_vel = ObsTerm(func=amp_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=amp_mdp.base_ang_vel)
        root_local_rot_tan_norm = ObsTerm(func=amp_mdp.root_local_rot_tan_norm)
        velocity_commands = ObsTerm(
            func=amp_mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=amp_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=amp_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        actions = ObsTerm(func=amp_mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=amp_mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
            },
        )

        def __post_init__(self):
            # critic 同样使用 5 帧历史，但不添加观测噪声。
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        """AMP 判别器使用的当前机器人状态序列。"""

        root_local_rot_tan_norm = ObsTerm(func=amp_mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=amp_mdp.base_ang_vel)
        joint_pos = ObsTerm(
            func=amp_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=amp_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        key_body_pos_b = ObsTerm(
            func=amp_mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True),
            },
        )

        def __post_init__(self):
            # 保留 4 帧时间维，不展平，最终单帧维度为 39。
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = AMP_NUM_STEPS
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        """AMP 判别器使用的参考动作状态序列。"""

        ref_root_local_rot_tan_norm = ObsTerm(
            func=amp_mdp.ref_root_local_rot_tan_norm,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )
        ref_root_ang_vel_b = ObsTerm(
            func=amp_mdp.ref_root_ang_vel_b,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )
        ref_joint_pos = ObsTerm(
            func=amp_mdp.ref_joint_pos,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )
        ref_joint_vel = ObsTerm(
            func=amp_mdp.ref_joint_vel,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )
        ref_key_body_pos_b = ObsTerm(
            func=amp_mdp.ref_key_body_pos_b,
            params={"animation": ANIMATION_TERM_NAME, "flatten_steps_dim": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class BSRLEventCfg:
    """第一版成功训练使用的事件配置。"""

    # 每个 episode 随机化全身刚体材料，包括足底摩擦和恢复系数。
    physics_material = EventTerm(
        func=amp_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    # 不再额外增加 base 质量；全身质量与惯量统一由 body_mass 事件缩放。
    add_base_mass = None

    # 第一版：不持续施加外力或外力矩。
    base_external_force_torque = EventTerm(
        func=amp_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    # 与宇树 G1 一致，训练环境全部从参考动作的随机帧复位。
    reset_from_ref = EventTerm(func=amp_mdp.reset_from_ref, mode="reset", params=MISSING)

    # 仅由播放配置启用；训练使用上面的参考动作复位。
    reset_to_default = None

    # 每 5 秒只扰动 x/y 方向根速度，不扰动高度与角速度。
    push_robot = EventTerm(
        func=amp_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )

    # 全身材料事件已经包含足底，不再用固定足底材料覆盖随机结果。
    randomize_foot_rigid_body_material = None

    # -------------------------------------------------------------------------
    # 扩展随机化事件：本轮按参考配置全部启用。
    # -------------------------------------------------------------------------
    # 启动时加入关节零位误差，并同步关节位置动作的默认 offset。
    add_joint_default_pos = EventTerm(
        func=amp_mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    # 启动时随机偏移机身质心，覆盖装配和载荷分布误差。
    base_com = EventTerm(
        func=amp_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # 启动时按环境缩放关节 PD 增益，模拟电机和控制器参数误差。
    actuator_gains = EventTerm(
        func=amp_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 启动时缩放全身质量，并按新质量重算惯量。
    body_mass = EventTerm(
        func=amp_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # 启动时缩放关节 armature，模拟转子和传动等效惯量误差。
    joint_params = EventTerm(
        func=amp_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # 启动时随机化理想转动关节的摩擦参数。
    joint_friction = EventTerm(
        func=amp_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    # base_external_force_torque = EventTerm(
    #     func=amp_mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
    #         "force_range": (-15.0, 15.0),
    #         "torque_range": (-4.0, 4.0),
    #     },
    # )
    # push_robot = EventTerm(
    #     func=amp_mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(2.5, 5.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )


@configclass
class BSRLCurriculumCfg:
    """逐步增加到适合当前 BSRL 稳定性的中等速度指令。"""

    velocity_range_and_tracking_std = CurrTerm(
        func=amp_mdp.velocity_range_and_tracking_std,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            # PPO 每轮采集 24 个控制步；课程分别在第 5000 和 12000 轮切换。
            "steps_per_iteration": 24,
            "phase_boundaries": (5000, 12000),
            "lin_vel_x_ranges": ((-0.6, 0.6), (-0.9, 0.9), (-1.2, 1.2)),
            "lin_vel_y_ranges": ((-0.4, 0.4), (-0.6, 0.6), (-0.8, 0.8)),
            "ang_vel_z_ranges": ((-0.25, 0.25), (-0.375, 0.375), (-0.5, 0.5)),
            "tracking_stds": (0.5, 0.4, 0.3),
        },
    )


@configclass
class BSRLAmpEnvCfg(LocomotionAmpEnvCfg):
    """BSRL 的 AMP 训练环境配置。"""

    observations: BSRLObservationsCfg = BSRLObservationsCfg()
    events: BSRLEventCfg = BSRLEventCfg()
    curriculum: BSRLCurriculumCfg = BSRLCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # 将场景中的机器人替换为你的 BSRL 配置
        self.scene.robot = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 8192

        # 控制频率为 50 Hz：200 Hz 物理仿真每 4 步执行一次策略。
        self.decimation = 4

        # 关节位置动作使用资产中定义的逐关节 scale，并保持 SDK 关节顺序。
        self.actions.joint_pos.scale = BSRL_ACTION_SCALE
        # 不裁剪处理后的 PD 位置目标。参考动作中部分关节正好位于 URDF 硬限位；
        # 允许目标略微越过限位，才能由位置误差产生抵住机械限位所需的预载力矩。
        # 与宇树一致，runner 不裁剪原始动作；这里仅保留宽松的数值保护范围。
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.preserve_order = True

        # 加载已经重定向到 BSRL 12 个关节的走路和跑步 AMP 数据。
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "bsrl_12dof", "amp", "walk_and_run"
        )
        motion_files = sorted(
            file_name for file_name in os.listdir(self.motion_data.motion_dataset.motion_data_dir) if file_name.endswith(".pkl")
        )
        _validate_motion_files(self.motion_data.motion_dataset.motion_data_dir, motion_files)
        self.motion_data.motion_dataset.motion_data_weights = {
            os.path.splitext(file_name)[0]: 1.0 for file_name in motion_files
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.animation.animation.motion_data_components[6] = "key_body_pos_b"

        # 中等速度训练指令；最高档为 vx=1.2、vy=0.8、yaw=0.5，优先学习稳定行走。
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.02

        # 任务奖励：保持现有 PPO 任务奖励和 AMP 风格奖励组合不变。
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = amp_mdp.track_lin_vel_xy_base_frame_exp
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.3,
        }
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = amp_mdp.track_ang_vel_z_base_frame_exp
        self.rewards.track_ang_vel_z_exp.params = {
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        }
        # 指数奖励在误差很大时接近零；超额误差惩罚提供全程可见的稠密信号。
        self.rewards.lin_vel_xy_error_excess_l2 = RewTerm(
            func=amp_mdp.lin_vel_xy_base_frame_error_excess_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "threshold": 0.25,
                "max_excess": 2.0,
            },
        )
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
        # 直接采用宇树 G1 AMP 参数，减弱对长单支撑和大步幅的偏好。
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.func = amp_mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True),
            "threshold": 0.4,
        }
        # 非足部刚体接触地面时施加惩罚；足底碰撞 link 由 FOOT_REGEX 排除。
        self.rewards.undesired_contacts.weight = -0.1
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
            weight=0.0,
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

        # 指定 AMP 参考动作复位所使用的数据项、关节顺序和高度偏移。
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "height_offset": 0.0,
        }

        # 终止条件：基座过低、姿态异常或基座接触地面时结束当前 episode。
        self.terminations.base_height.func = amp_mdp.base_height_below_minimum
        # 基座明显下沉时提前结束，避免机器人跪下后继续收集无效轨迹。
        self.terminations.base_height.params["minimum_height"] = 0.5
        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_LINK_NAME]


@configclass
class BSRLAmpEnvCfgPlay(BSRLAmpEnvCfg):
    """单环境展示配置，不影响 BSRLAmpEnvCfg 的训练指令随机性。"""

    def __post_init__(self):
        super().__post_init__()

        # 播放时使用标称动力学，避免把训练期域随机化误当成策略性能。
        self.curriculum.velocity_range_and_tracking_std = None
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_foot_rigid_body_material = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # 训练期随机化在播放时全部关闭。
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.body_mass = None
        self.events.joint_params = None
        self.events.joint_friction = None

        # 播放首次启动及每次 episode 复位时，都恢复 CFG 中的默认根状态、关节位置和 PD 目标。
        self.events.reset_from_ref = None
        self.events.reset_to_default = EventTerm(
            func=il_mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )

        # 播放时关闭随机 actuator 延迟，保证不同 checkpoint 的视频可直接比较。
        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0

        # 展示时固定为 1.0 m/s 前进，不采样横向和转向指令。
        self.scene.num_envs = 1
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
