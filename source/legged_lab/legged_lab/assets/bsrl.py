from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR


@configclass
class BSRLArticulationCfg(ArticulationCfg):
    """Configuration for BSRL articulations."""
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


BSRL_CFG = BSRLArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # TODO: 请确保将你的 USD 模型文件放到这个路径下
        usd_path=f"{{LEGGED_LAB_ROOT_DIR}}/data/Robots/BSRL/export/export.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8), # TODO: 根据你的机器人修改初始高度
        # 根据 URDF 填写的初始角度
        joint_pos={
            "joint_right_hip_yaw": 0.0,
            "joint_right_hip_roll": 0.0,
            "joint_right_hip_pitch": 0.0,
            "joint_right_knee_pitch": 0.0,
            "joint_right_ankle_pitch": 0.0,
            "joint_right_ankle_roll": 0.0,
            "joint_left_hip_yaw": 0.0,
            "joint_left_hip_roll": 0.0,
            "joint_left_hip_pitch": 0.0,
            "joint_left_knee_pitch": 0.0,
            "joint_left_ankle_pitch": 0.0,
            "joint_left_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # TODO: 根据你的电机具体参数进行修改
        "base_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # 应用于所有关节
            effort_limit_sim=50,   # 最大力矩 N.m
            velocity_limit_sim=20.0, # 最大转速 rad/s
            stiffness=40.0,        # Kp 刚度
            damping=1.0,           # Kd 阻尼
            armature=0.01,
        ),
    },
    # 机器人的关节名称，须与 URDF 严格一致
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
