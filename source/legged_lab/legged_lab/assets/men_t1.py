"""Men T1 (0722 new-shoe) articulation used by the AMP tasks."""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.delayed_implicit_actuator import DelayedImplicitActuatorCfg


T1_ASSET_ROOT = f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Men_T1_0722_new_shoe"
T1_URDF_PATH = f"{T1_ASSET_ROOT}/urdf/men_t1_path_25dofs_collision_feet_points.urdf"

# This order is the policy and AMP-reference contract inherited from wbt_est.
T1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "torso_pitch_joint",
    "torso_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

T1_ACTUATOR_MIN_DELAY = 0
T1_ACTUATOR_MAX_DELAY = 4


T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=T1_URDF_PATH,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.16),
        joint_pos={
            "left_hip_pitch_joint": -0.09,
            "left_hip_roll_joint": 0.03,
            "left_hip_yaw_joint": 0.03,
            "left_knee_joint": 0.22,
            "left_ankle_pitch_joint": -0.13,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.09,
            "right_hip_roll_joint": -0.03,
            "right_hip_yaw_joint": -0.03,
            "right_knee_joint": 0.22,
            "right_ankle_pitch_joint": -0.13,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "torso_pitch_joint": 0.0,
            "torso_roll_joint": 0.0,
            "torso_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.4,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            min_delay=T1_ACTUATOR_MIN_DELAY,
            max_delay=T1_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 110,
                ".*_hip_roll_joint": 110,
                ".*_hip_yaw_joint": 110,
                ".*_knee_joint": 280,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 11.94,
                ".*_hip_roll_joint": 11.94,
                ".*_hip_yaw_joint": 11.94,
                ".*_knee_joint": 18.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 100,
                ".*_hip_roll_joint": 100,
                ".*_hip_yaw_joint": 100,
                ".*_knee_joint": 600,
            },
            damping={
                ".*_hip_pitch_joint": 3,
                ".*_hip_roll_joint": 3,
                ".*_hip_yaw_joint": 3,
                ".*_knee_joint": 10,
            },
            armature={
                ".*_hip_pitch_joint": 0.04220818,
                ".*_hip_roll_joint": 0.04220818,
                ".*_hip_yaw_joint": 0.04220818,
                ".*_knee_joint": 0.0,
            },
            friction=None,
        ),
        "feet": DelayedImplicitActuatorCfg(
            min_delay=T1_ACTUATOR_MIN_DELAY,
            max_delay=T1_ACTUATOR_MAX_DELAY,
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit_sim={".*_ankle_roll_joint": 36, ".*_ankle_pitch_joint": 180},
            velocity_limit_sim={".*_ankle_roll_joint": 15.71, ".*_ankle_pitch_joint": 16.0},
            stiffness={".*_ankle_roll_joint": 100, ".*_ankle_pitch_joint": 100},
            damping={".*_ankle_roll_joint": 3, ".*_ankle_pitch_joint": 3},
            armature={".*_ankle_roll_joint": 0.005131796, ".*_ankle_pitch_joint": 0.0},
        ),
        "waist": DelayedImplicitActuatorCfg(
            min_delay=T1_ACTUATOR_MIN_DELAY,
            max_delay=T1_ACTUATOR_MAX_DELAY,
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"],
            effort_limit_sim={"waist_yaw_joint": 110, "waist_pitch_joint": 110},
            velocity_limit_sim={"waist_yaw_joint": 11.94, "waist_pitch_joint": 11.94},
            stiffness={"waist_yaw_joint": 150, "waist_pitch_joint": 150},
            damping={"waist_yaw_joint": 5, "waist_pitch_joint": 5},
            armature={"waist_yaw_joint": 0.04220818, "waist_pitch_joint": 0.04220818},
        ),
        "torso": DelayedImplicitActuatorCfg(
            min_delay=T1_ACTUATOR_MIN_DELAY,
            max_delay=T1_ACTUATOR_MAX_DELAY,
            joint_names_expr=["torso_pitch_joint", "torso_roll_joint", "torso_yaw_joint"],
            effort_limit_sim={"torso_pitch_joint": 180, "torso_roll_joint": 180, "torso_yaw_joint": 60},
            velocity_limit_sim={"torso_pitch_joint": 10.0, "torso_roll_joint": 10.0, "torso_yaw_joint": 13.613},
            stiffness={"torso_pitch_joint": 150, "torso_roll_joint": 150, "torso_yaw_joint": 100},
            damping={"torso_pitch_joint": 5, "torso_roll_joint": 5, "torso_yaw_joint": 3},
            armature={"torso_pitch_joint": 0.0, "torso_roll_joint": 0.0, "torso_yaw_joint": 0.01201035},
        ),
        "arms": DelayedImplicitActuatorCfg(
            min_delay=T1_ACTUATOR_MIN_DELAY,
            max_delay=T1_ACTUATOR_MAX_DELAY,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36,
                ".*_shoulder_roll_joint": 36,
                ".*_shoulder_yaw_joint": 21,
                ".*_elbow_joint": 21,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 14.66,
                ".*_shoulder_roll_joint": 14.66,
                ".*_shoulder_yaw_joint": 11.52,
                ".*_elbow_joint": 11.52,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40,
                ".*_shoulder_roll_joint": 40,
                ".*_shoulder_yaw_joint": 40,
                ".*_elbow_joint": 40,
            },
            damping={
                ".*_shoulder_pitch_joint": 2,
                ".*_shoulder_roll_joint": 2,
                ".*_shoulder_yaw_joint": 2,
                ".*_elbow_joint": 2,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.005131796,
                ".*_shoulder_roll_joint": 0.005131796,
                ".*_shoulder_yaw_joint": 0.004528982,
                ".*_elbow_joint": 0.004528982,
            },
        ),
    },
)

T1_ACTION_SCALE = {
    joint_pattern: 0.25
    for actuator in T1_CFG.actuators.values()
    for joint_pattern in actuator.joint_names_expr
}
