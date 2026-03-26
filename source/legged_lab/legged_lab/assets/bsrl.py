import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


LEGGED_LAB_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@configclass
class BSRLArticulationCfg(ArticulationCfg):
    """Configuration for BSRL articulations."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


BSRL_CFG = BSRLArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/BSRL/export/export.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            "joint_.*_hip_yaw": 0.0,
            "joint_.*_hip_roll": 0.0,
            "joint_.*_hip_pitch": -0.2,
            "joint_.*_knee_pitch": -0.6,
            "joint_.*_ankle_pitch": -0.2,
            "joint_.*_ankle_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
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
