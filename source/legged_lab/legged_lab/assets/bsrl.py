import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab.assets.delayed_implicit_actuator import DelayedImplicitActuatorCfg

# 第一版原生 actuator 写法保留在下方注释区，供消融实验对照。
# from isaaclab.actuators import ImplicitActuatorCfg


LEGGED_LAB_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# GMR walk1 第一帧左右腿平均并按 0.1 rad 对称化后的贴地站姿高度。
BSRL_DEFAULT_ROOT_HEIGHT = 0.8665

# 训练时为每个环境随机采样 0~4 个物理步的执行器命令延迟。
BSRL_ACTUATOR_MIN_DELAY = 0
BSRL_ACTUATOR_MAX_DELAY = 4

BSRL_ACTION_SCALE_MULTIPLIER = {
    "joint_.*_hip_pitch": 0.25,
    "joint_.*_hip_roll": 0.25,
    "joint_.*_hip_yaw": 0.25,
    "joint_.*_knee_pitch": 0.25,
    "joint_.*_ankle_roll": 0.25,
    "joint_.*_ankle_pitch": 0.25,
}

@configclass
class BSRLArticulationCfg(ArticulationCfg):
    """Configuration for BSRL articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


BSRL_CFG = BSRLArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/BSRL_urdf/urdf/export/export.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 左右腿和左右脚必须参与自碰撞，避免策略利用穿腿动作。
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BSRL_DEFAULT_ROOT_HEIGHT),
        joint_pos={
            "joint_.*_hip_yaw": 0.0,
            "joint_.*_hip_roll": 0.0,
            "joint_.*_hip_pitch": -0.2,
            "joint_.*_knee_pitch": 0.4,
            "joint_.*_ankle_pitch": -0.2,
            "joint_.*_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 现有 delayed actuator 框架使用第一版成功训练的 PD、限位和 armature。
        "legs": DelayedImplicitActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_hip_pitch",
                "joint_.*_hip_roll",
                "joint_.*_hip_yaw",
                "joint_.*_knee_pitch",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            stiffness={
                "joint_.*_hip_pitch": 100,
                "joint_.*_hip_roll": 100,
                "joint_.*_hip_yaw": 100,
                "joint_.*_knee_pitch": 150,
            },
            damping={
                "joint_.*_hip_pitch": 2,
                "joint_.*_hip_roll": 2,
                "joint_.*_hip_yaw": 2,
                "joint_.*_knee_pitch": 4,
            },
        ),
        "feet": DelayedImplicitActuatorCfg(
            min_delay=BSRL_ACTUATOR_MIN_DELAY,
            max_delay=BSRL_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                "joint_.*_ankle_roll",
                "joint_.*_ankle_pitch",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            stiffness={
                "joint_.*_ankle_roll": 40,
                "joint_.*_ankle_pitch": 40,
            },
            damping={
                "joint_.*_ankle_roll": 2,
                "joint_.*_ankle_pitch": 2,
            },
        ),
        # 第一版原生 actuator 写法保留区：本轮不启用。
        # "hip_motors": ImplicitActuatorCfg(
        #     joint_names_expr=["joint_.*_hip_.*"],
        #     effort_limit_sim=1000.0,
        #     velocity_limit_sim=10.0,
        #     stiffness=100.0,
        #     damping=2.0,
        #     armature=0.01,
        # ),
        # "knee_motors": ImplicitActuatorCfg(
        #     joint_names_expr=["joint_.*_knee_pitch"],
        #     effort_limit_sim=1000.0,
        #     velocity_limit_sim=10.0,
        #     stiffness=150.0,
        #     damping=4.0,
        #     armature=0.01,
        # ),
        # "ankle_motors": ImplicitActuatorCfg(
        #     joint_names_expr=["joint_.*_ankle_.*"],
        #     effort_limit_sim=1000.0,
        #     velocity_limit_sim=10.0,
        #     stiffness=40.0,
        #     damping=2.0,
        #     armature=0.01,
        # ),
        # 第二版 actuator 配置保留区：当前全部停用。
        # "legs": DelayedImplicitActuatorCfg(
        #     min_delay=BSRL_ACTUATOR_MIN_DELAY,
        #     max_delay=BSRL_ACTUATOR_MAX_DELAY,
        #     joint_names_expr=[
        #         "joint_.*_hip_pitch",
        #         "joint_.*_hip_roll",
        #         "joint_.*_hip_yaw",
        #         "joint_.*_knee_pitch",
        #     ],
        #     effort_limit_sim={
        #         "joint_.*_hip_pitch": 110,
        #         "joint_.*_hip_roll": 110,
        #         "joint_.*_hip_yaw": 110,
        #         "joint_.*_knee_pitch": 200,
        #     },
        #     velocity_limit_sim={
        #         "joint_.*_hip_pitch": 12,
        #         "joint_.*_hip_roll": 12,
        #         "joint_.*_hip_yaw": 12,
        #         "joint_.*_knee_pitch": 18.0,
        #     },
        #     stiffness={
        #         "joint_.*_hip_pitch": 100,
        #         "joint_.*_hip_roll": 100,
        #         "joint_.*_hip_yaw": 100,
        #         "joint_.*_knee_pitch": 150,
        #     },
        #     damping={
        #         "joint_.*_hip_pitch": 3,
        #         "joint_.*_hip_roll": 3,
        #         "joint_.*_hip_yaw": 3,
        #         "joint_.*_knee_pitch": 5,
        #     },
        #     armature=None,
        #     friction={"joint_.*_knee_pitch": 13},
        # ),
        # "feet": DelayedImplicitActuatorCfg(
        #     min_delay=BSRL_ACTUATOR_MIN_DELAY,
        #     max_delay=BSRL_ACTUATOR_MAX_DELAY,
        #     joint_names_expr=[
        #         "joint_.*_ankle_roll",
        #         "joint_.*_ankle_pitch",
        #     ],
        #     effort_limit_sim={
        #         "joint_.*_ankle_roll": 36,
        #         "joint_.*_ankle_pitch": 100,
        #     },
        #     velocity_limit_sim={
        #         "joint_.*_ankle_roll": 15.71,
        #         "joint_.*_ankle_pitch": 16.0,
        #     },
        #     stiffness={
        #         "joint_.*_ankle_roll": 40,
        #         "joint_.*_ankle_pitch": 40,
        #     },
        #     damping={
        #         "joint_.*_ankle_roll": 2,
        #         "joint_.*_ankle_pitch": 2,
        #     },
        #     armature=None,
        # ),
    },
    joint_sdk_names=[
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
    ],
)


def _build_action_scale(robot_cfg: ArticulationCfg) -> dict[str, float]:
    action_scale = {}
    for actuator_cfg in robot_cfg.actuators.values():
        for name in actuator_cfg.joint_names_expr:
            action_scale[name] = BSRL_ACTION_SCALE_MULTIPLIER.get(name, 0.25)
    return action_scale


BSRL_ACTION_SCALE = _build_action_scale(BSRL_CFG)
