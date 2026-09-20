"""Opt-in heading-frame reward/critic state without actor heading feedback.

The reference starts at the reset heading and follows the *commanded* yaw rate.
Uniform velocity sampling and body-frame commands themselves are unchanged.
"""
from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand

__all__ = ["HeadingReferenceVelocityCommand", "heading_frame_error",
           "heading_frame_lin_vel_xy", "heading_frame_lin_vel_xy_exp"]


def _velocity_in_heading_frame(velocity_w, yaw):
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((cosine * velocity_w[:, 0] + sine * velocity_w[:, 1],
                        -sine * velocity_w[:, 0] + cosine * velocity_w[:, 1]), dim=-1)


class HeadingReferenceVelocityCommand(UniformVelocityCommand):
    """Add a privileged reference clock; inherit uniform sampling verbatim.

    Reward evaluation precedes command resampling in ManagerBasedAmpEnv.
    Advancing by the physical-step counter on first access uses the command
    that actually applied during that interval, once only. A reset marks just
    those environments at the current counter, so it never integrates the
    previous episode's last interval into the new episode reference.
    """

    def __init__(self, cfg, env):
        if cfg.heading_command:
            raise ValueError("Heading reference must not replace the actor's yaw-rate command")
        super().__init__(cfg, env)
        self.heading_target.copy_(self.robot.data.heading_w)
        self._heading_step = torch.full((self.num_envs,), env._sim_step_counter,
                                       dtype=torch.long, device=self.device)
        self.metrics["heading_reference_error_rad"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_frame_velocity_error"] = torch.zeros(self.num_envs, device=self.device)

    def get_heading_reference(self):
        step = self._env._sim_step_counter
        elapsed = step - self._heading_step
        advanced = math_utils.wrap_to_pi(
            self.heading_target + self.vel_command_b[:, 2] * (elapsed * self._env.physics_dt))
        self.heading_target.copy_(torch.where(elapsed > 0, advanced, self.heading_target))
        self._heading_step.fill_(step)
        return self.heading_target

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        extras = super().reset(ids)
        self.sync_episode_frame_after_reset(env_ids)
        return extras

    def sync_episode_frame_after_reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.heading_target[ids] = self.robot.data.heading_w[ids]
        self._heading_step[ids] = self._env._sim_step_counter

    def compute(self, dt):
        # Complete the old command's interval BEFORE UniformVelocityCommand
        # may sample a new one. No extra random calls or sampling branches.
        self.get_heading_reference()
        super().compute(dt)

    def _update_metrics(self):
        super()._update_metrics()
        reference = self.get_heading_reference()
        horizon = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["heading_reference_error_rad"] += torch.abs(
            math_utils.wrap_to_pi(reference - self.robot.data.heading_w)) / horizon
        velocity = _velocity_in_heading_frame(self.robot.data.root_lin_vel_w, reference)
        self.metrics["heading_frame_velocity_error"] += torch.linalg.vector_norm(
            self.vel_command_b[:, :2] - velocity, dim=-1) / horizon


def heading_frame_error(env, command_name: str):
    """Critic-only sin/cos error; the policy observation layout is unchanged."""
    term = env.command_manager.get_term(command_name)
    error = math_utils.wrap_to_pi(term.get_heading_reference() - term.robot.data.heading_w)
    return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


def heading_frame_lin_vel_xy(env, command_name: str):
    """True planar velocity in the reset-heading + commanded-turn frame."""
    term = env.command_manager.get_term(command_name)
    return _velocity_in_heading_frame(term.robot.data.root_lin_vel_w, term.get_heading_reference())


def heading_frame_lin_vel_xy_exp(env, command_name: str, std: float):
    """Additional velocity objective active for all unchanged uniform commands.

    Zero yaw command holds the initial heading; commanded turns rotate the
    reference, so this does not reward staying at the initial yaw during turns.
    No direct yaw-angle penalty or extra standing bonus is added.
    """
    if std <= 0:
        raise ValueError("Velocity reward std must be positive")
    command = env.command_manager.get_command(command_name)
    velocity = heading_frame_lin_vel_xy(env, command_name)
    return torch.exp(-torch.sum(torch.square(command[:, :2] - velocity), dim=-1) / std**2)
