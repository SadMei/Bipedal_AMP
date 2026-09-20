"""Serial PhysX skeleton with motor-limited, analytical parallel ankles.

No auxiliary bodies/constraints or changes to the authoritative URDF. Legacy
serial and explicit-loop factories are retained for old checkpoint playback.
"""
from __future__ import annotations

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.utils.string import resolve_matching_names_values

from legged_lab.assets.bsrl_ankle_transmission import AnkleTransmission
from legged_lab.assets.bsrl import (
    BSRL_SMALL_MOTOR_PEAK_TORQUE, BSRL_SMALL_MOTOR_PEAK_SPEED, BSRL_SMALL_MOTOR_PEAK_POWER,
)


class ReducedAnkleActuator(DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.q_ids = [[self.joint_names.index(f"joint_{side}_ankle_{axis}")
                       for axis in ("pitch", "roll")] for side in ("left", "right")]
        if len(self.joint_names) != 4:
            raise ValueError("Reduced ankle actuator must control exactly the four ankle axes.")
        self.transmission = AnkleTransmission(
            self._num_envs, device=self._device, dt=cfg.physics_dt,
            effort=cfg.motor_effort, speed=cfg.motor_speed, power=cfg.motor_power)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.transmission.reset(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self.computed_effort[rows] = 0.
        self.applied_effort[rows] = 0.

    def compute(self, control_action, joint_pos, joint_vel):
        target = self.positions_delay_buffer.compute(control_action.joint_positions)
        target_velocity = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        extra_torque = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # No PD target or requested joint-torque clipping, including at URDF
        # angle limits. Actual joint limits/rewards still apply in the scene.
        self.computed_effort = (self.stiffness * (target - joint_pos)
                                + self.damping * (target_velocity - joint_vel) + extra_torque)
        self.applied_effort[:, self.q_ids] = self.transmission.compute(
            joint_pos[:, self.q_ids], joint_vel[:, self.q_ids], self.computed_effort[:, self.q_ids])
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ReducedAnkleActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = ReducedAnkleActuator
    physics_dt: float = .005
    motor_effort: float = BSRL_SMALL_MOTOR_PEAK_TORQUE
    motor_speed: float = BSRL_SMALL_MOTOR_PEAK_SPEED
    motor_power: float = BSRL_SMALL_MOTOR_PEAK_POWER


class ReducedAnkleArticulation(Articulation):
    def begin_physics_step(self):
        self._physics_step_pending = True

    def _apply_actuator_model(self):
        # Reuse the opt-in AMP clock fix: reset/forward writes must not advance
        # command delay or torque-response state in any of this asset's motors.
        if getattr(self, "_physics_step_pending", False):
            self._physics_step_pending = False
            super()._apply_actuator_model()

    def update(self, dt):
        super().update(dt)
        # IsaacLab initializes articulation caches via update(0.0); that is
        # neither a physics step nor a change to the controller time base.
        if dt == 0.:
            return
        actuator = self.actuators.get("parallel_ankles")
        if actuator is None:
            return
        tr = actuator.transmission
        if abs(tr.dt - dt) > 1e-8:
            raise RuntimeError("Reduced ankle actuator physics_dt differs from simulation dt.")
        if not hasattr(self, "_transmission_audit"):
            self._transmission_audit = torch.zeros(6, device=self.device)
            self._transmission_step = 0
        # All physical steps, including before reset; only one CPU sync/rollout.
        sample = torch.stack((tr.motor_torque.abs().max(),
                              (tr.motor_torque * tr.motor_velocity).abs().max(),
                              tr.motor_velocity.abs().max(),
                              tr.joint_torque.abs().max(), (~tr.valid).sum().float(),
                              (~torch.isfinite(self.data.joint_pos)).sum().float()
                              + (~torch.isfinite(self.data.joint_vel)).sum().float()
                              + (~torch.isfinite(tr.joint_torque)).sum().float()))
        self._transmission_audit = torch.maximum(self._transmission_audit, sample)
        self._transmission_step += 1
        if self._transmission_step % max(1, round(.48 / dt)) == 0:
            torque, power, speed, joint, invalid, nonfinite = self._transmission_audit.tolist()
            print(f"[AnkleTransmissionAudit] physics_steps={self._transmission_step} "
                  f"motor_peak_Nm={torque:.6f} motor_power_peak_W={power:.6f} "
                  f"motor_speed_peak_rad_s={speed:.6f} joint_peak_Nm={joint:.6f} "
                  f"invalid={invalid:.0f} nonfinite={nonfinite:.0f}", flush=True)
            if invalid or nonfinite or torque > tr.effort + .001 or power > tr.power + .01:
                raise RuntimeError("Reduced ankle transmission audit failed (only this run stops).")
            self._transmission_audit.zero_()


def with_reduced_ankles(serial_cfg, physics_dt=.005):
    cfg = serial_cfg.copy()
    cfg.class_type = ReducedAnkleArticulation
    # Keep hip yaw directly driven; only ankle pitch/roll change transmission.
    small = cfg.actuators["small_motors"]
    ankle_names = [f"joint_{s}_ankle_{a}" for s in ("left", "right") for a in ("pitch", "roll")]
    hip_names = [f"joint_{s}_hip_yaw" for s in ("left", "right")]

    def select(value, names):
        if not isinstance(value, dict):
            return value
        _, matched, values = resolve_matching_names_values(value, names, strict=False)
        if len(matched) != len(names):
            raise ValueError(f"Incomplete BSRL actuator parameter for {names}")
        return dict(zip(matched, values))

    # One PD/armature/friction source of truth in bsrl.py: future parameter
    # edits must not be silently shadowed by a second set of factory constants.
    ankle_fields = {}
    for key in ("stiffness", "damping", "armature", "friction", "dynamic_friction", "viscous_friction"):
        value = getattr(small, key)
        ankle_fields[key] = select(value, ankle_names)
        setattr(small, key, select(value, hip_names))
    small.joint_names_expr = ["joint_.*_hip_yaw"]
    cfg.actuators["parallel_ankles"] = ReducedAnkleActuatorCfg(
        joint_names_expr=["joint_.*_ankle_(pitch|roll)"],
        min_delay=small.min_delay, max_delay=small.max_delay,
        **ankle_fields,
        # Do not accidentally impose MOTOR ratings (112Nm/26.18rad/s) on
        # joint coordinates after backprojection. These are numerical guards,
        # NOT a second physical torque/speed envelope or PD target clipping.
        effort_limit=1e9, effort_limit_sim=1e9,
        velocity_limit=1000., velocity_limit_sim=1000., physics_dt=physics_dt,
    )
    return cfg


def randomize_ankle_transmission(env, env_ids, ratio_range=(.97, 1.03),
                                 strength_range=(.9, 1.), response_time_range=(0., .005)):
    """Optional startup event. Ranges are engineering hypotheses, not measurements.

    No new actor observations, encoder offsets, white-noise Jacobians, or
    duplicate joint friction/PD randomization are introduced.
    """
    actuator = env.scene["robot"].actuators.get("parallel_ankles")
    if actuator is None:
        raise ValueError("Transmission randomization requires the reduced ankle actuator.")
    actuator.transmission.randomize(env_ids, ratio_range, strength_range, response_time_range)
