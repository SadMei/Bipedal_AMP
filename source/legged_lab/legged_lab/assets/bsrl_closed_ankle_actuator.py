"""Roll/pitch-space delayed PD, driving ONLY the physical motor hinges."""
from __future__ import annotations

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from legged_lab.assets.bsrl_closed_ankle_geometry import CADAnkleGeometry, motor_torque_from_joint, limit_motor_torque
from legged_lab.assets.delayed_implicit_actuator import DelayedPowerLimitedPDActuator, DelayedPowerLimitedPDActuatorCfg


class SubsteppedPowerLimitedPDActuator(DelayedPowerLimitedPDActuator):
    """Refine physics without increasing PD/command-delay update frequency."""
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._control_tick = 0

    def reset(self, env_ids):
        super().reset(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self.computed_effort[rows] = 0.
        self.applied_effort[rows] = 0.

    def compute(self, control_action, joint_pos, joint_vel):
        if self._control_tick == 0:
            control_action = super().compute(control_action, joint_pos, joint_vel)
        else:
            self._joint_vel[:] = joint_vel
            self.applied_effort = self._clip_effort(self.computed_effort)
            control_action.joint_efforts = self.applied_effort
            control_action.joint_positions = None
            control_action.joint_velocities = None
        self._control_tick = (self._control_tick + 1) % self.cfg.control_decimation
        return control_action


@configclass
class SubsteppedPowerLimitedPDActuatorCfg(DelayedPowerLimitedPDActuatorCfg):
    class_type: type = SubsteppedPowerLimitedPDActuator
    control_decimation: int = 5


class ClosedAnkleActuator(DelayedPDActuator):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = self.joint_names
        self.q_ids = [[names.index(f"joint_{side}_ankle_{axis}") for axis in ("pitch", "roll")]
                      for side in ("left", "right")]
        self.m_ids = [[names.index(f"motor_{side}_{branch}") for branch in ("long", "short")]
                      for side in ("left", "right")]
        self.geometry = CADAnkleGeometry(device=self._device)
        self.requested_joint_torque = torch.zeros((self._num_envs, 2, 2), device=self._device)
        self.jacobian_valid = torch.ones((self._num_envs, 2), dtype=torch.bool, device=self._device)
        self.motor_torque = torch.zeros_like(self.requested_joint_torque)
        self.motor_velocity = torch.zeros_like(self.requested_joint_torque)
        self._requested_motor = torch.zeros_like(self.requested_joint_torque)
        self._control_tick = 0

    def reset(self, env_ids):
        super().reset(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self._requested_motor[rows] = 0.

    def compute(self, control_action, joint_pos, joint_vel):
        if self._control_tick == 0:
            target = self.positions_delay_buffer.compute(control_action.joint_positions)
            target_velocity = self.velocities_delay_buffer.compute(control_action.joint_velocities)
            extra_torque = self.efforts_delay_buffer.compute(control_action.joint_efforts)
            q, dq = joint_pos[:, self.q_ids], joint_vel[:, self.q_ids]
            self.requested_joint_torque = (
                self.stiffness[:, self.q_ids]*(target[:, self.q_ids] - q)
                + self.damping[:, self.q_ids]*(target_velocity[:, self.q_ids] - dq)
                + extra_torque[:, self.q_ids]
            )
            jac = self.geometry.jacobian(q, joint_pos[:, self.m_ids])
            self._requested_motor, self.jacobian_valid = motor_torque_from_joint(jac, self.requested_joint_torque)
        self._control_tick = (self._control_tick + 1) % self.cfg.control_decimation
        self.motor_velocity = joint_vel[:, self.m_ids]
        self.motor_torque = limit_motor_torque(
            self._requested_motor, self.motor_velocity, self.cfg.motor_effort, self.cfg.motor_speed, self.cfg.motor_power)
        # Passive pitch/roll NEVER receive an equivalent J.T torque as well.
        self.computed_effort.zero_()
        self.applied_effort.zero_()
        self.computed_effort[:, self.m_ids] = self._requested_motor
        self.applied_effort[:, self.m_ids] = self.motor_torque
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ClosedAnkleActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = ClosedAnkleActuator
    motor_effort: float = 112.0
    motor_speed: float = 26.17993877991494
    motor_power: float = 1250.0
    control_decimation: int = 1
