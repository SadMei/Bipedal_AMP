from __future__ import annotations

from collections.abc import Sequence

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit PD actuator with delayed position, velocity, and effort commands."""

    cfg: DelayedImplicitActuatorCfg

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        num_envs = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for an implicit PD actuator with randomized command delay."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    max_delay: int = 0


class DelayedPowerLimitedPDActuator(DelayedPDActuator):
    """Delayed explicit PD actuator with joint-side torque, speed, and power limits."""

    cfg: DelayedPowerLimitedPDActuatorCfg

    def __init__(self, cfg: DelayedPowerLimitedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.peak_power <= 0.0:
            raise ValueError(f"peak_power must be positive, got {cfg.peak_power}")
        self._peak_power = float(cfg.peak_power)
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # The datasheet values are gearbox-output values. Peak torque and peak
        # speed are not simultaneously available, so enforce the listed peak
        # mechanical power in addition to both independent hard limits.
        speed_abs = torch.abs(self._joint_vel)
        power_effort = self._peak_power / torch.clamp(speed_abs, min=1.0e-6)
        effort_magnitude = torch.minimum(self.effort_limit, power_effort)

        min_effort = -effort_magnitude
        max_effort = effort_magnitude
        # Beyond the speed rating, allow braking torque but no torque that
        # accelerates the joint farther outside its rated operating envelope.
        max_effort = torch.where(self._joint_vel >= self.velocity_limit, 0.0, max_effort)
        min_effort = torch.where(self._joint_vel <= -self.velocity_limit, 0.0, min_effort)
        return torch.clamp(effort, min=min_effort, max=max_effort)


@configclass
class DelayedPowerLimitedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a delayed PD motor with a peak mechanical-power envelope."""

    class_type: type = DelayedPowerLimitedPDActuator
    peak_power: float = MISSING
    """Peak joint-side mechanical output power in watts."""
