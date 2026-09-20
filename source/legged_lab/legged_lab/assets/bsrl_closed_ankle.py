"""Pipeline asset configuration for the BSRL explicit-closed-chain experiment."""
from isaaclab.actuators import IdealPDActuatorCfg
from legged_lab.assets.bsrl import (
    BSRL_ACTUATOR_MIN_DELAY, BSRL_ACTUATOR_MAX_DELAY,
    BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED, BSRL_SMALL_MOTOR_PEAK_POWER,
)
from legged_lab.assets.bsrl_closed_ankle_actuator import ClosedAnkleActuatorCfg, SubsteppedPowerLimitedPDActuatorCfg
from legged_lab.assets.bsrl_closed_ankle_articulation import ClosedAnkleArticulation
from legged_lab.assets.bsrl_closed_ankle_spawn import spawn_bsrl_closed_ankles


def with_closed_ankles(serial_cfg, control_decimation=1):
    """Keep the serial URDF/inertias intact and add the linkage at spawn time."""
    cfg = serial_cfg.copy()
    cfg.class_type = ClosedAnkleArticulation
    cfg.spawn.func = spawn_bsrl_closed_ankles
    # 2026-09-15 pipeline: TGS12/6, 1ms, BOTH solver switches False.
    # The previous PGS128/16 run is restored from its saved env.yaml in play.
    # cfg.spawn.articulation_props.solver_position_iteration_count = 128
    # cfg.spawn.articulation_props.solver_velocity_iteration_count = 16
    cfg.spawn.articulation_props.solver_position_iteration_count = 12
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 6
    # Hip yaw remains the original directly driven small motor.
    small = cfg.actuators["small_motors"]
    small.joint_names_expr = ["joint_.*_hip_yaw"]
    small.stiffness = {"joint_.*_hip_yaw": 150.}
    small.damping = {"joint_.*_hip_yaw": 2.}
    if control_decimation > 1:
        # Convert only this new asset's direct actuators. Legacy serial runs
        # and T1 do not use this class or change their control timing.
        for name in ("large_motors", "small_motors"):
            old = cfg.actuators[name]
            fields = {key: value for key, value in vars(old).items() if key != "class_type"}
            cfg.actuators[name] = SubsteppedPowerLimitedPDActuatorCfg(**fields, control_decimation=control_decimation)
    cfg.actuators["closed_ankles"] = ClosedAnkleActuatorCfg(
        joint_names_expr=["joint_.*_ankle_(pitch|roll)", "motor_.*"],
        min_delay=BSRL_ACTUATOR_MIN_DELAY, max_delay=BSRL_ACTUATOR_MAX_DELAY,
        control_decimation=control_decimation,
        stiffness={"joint_.*_ankle_(pitch|roll)": 100., "motor_.*": 0.},
        damping={"joint_.*_ankle_(pitch|roll)": 1., "motor_.*": 0.},
        # No joint-space effort clipping. Passive joints have zero drive force;
        # only the motor hinges receive a physical generalized effort.
        effort_limit=1e9,
        effort_limit_sim={"joint_.*_ankle_(pitch|roll)": 0., "motor_.*": BSRL_SMALL_MOTOR_PEAK_TORQUE},
        velocity_limit={"joint_.*_ankle_(pitch|roll)": 1000., "motor_.*": BSRL_SMALL_MOTOR_PEAK_SPEED},
        # The datasheet no-load speed is an ACTUATOR envelope, not an infinitely
        # strong kinematic brake against external impacts. Match the verified
        # MuJoCo motor hinges: only the effort model cuts accelerating torque
        # above rated speed. Hard-clamping motor velocity fights loop closure.
        velocity_limit_sim=1000.,
        # Preserve existing joint armature; do not double-add unmeasured rotor inertia.
        armature={"joint_.*_ankle_(pitch|roll)": .01, "motor_.*": 0.},
        friction=0.,
        motor_effort=BSRL_SMALL_MOTOR_PEAK_TORQUE,
        motor_speed=BSRL_SMALL_MOTOR_PEAK_SPEED,
        motor_power=BSRL_SMALL_MOTOR_PEAK_POWER,
    )
    cfg.actuators["passive_rod_balls"] = IdealPDActuatorCfg(
        joint_names_expr=["rod_ball_.*"], stiffness=0., damping=0.,
        effort_limit=0., effort_limit_sim=0., velocity_limit=1000., velocity_limit_sim=1000.,
        armature=0., friction=0.,
    )
    return cfg
