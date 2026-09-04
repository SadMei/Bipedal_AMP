"""Command generators used by AMP locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


class StraightMixtureVelocityCommand(UniformVelocityCommand):
    """Velocity commands with explicit straight and single-axis subsets."""

    cfg: StraightMixtureVelocityCommandCfg

    def __init__(self, cfg: StraightMixtureVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # Reuse the base command's heading buffer, but interpret it as the
        # integral of the requested yaw rate rather than a random world yaw.
        self.heading_target[:] = self.robot.data.heading_w

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        resolved_ids = slice(None) if env_ids is None else env_ids
        extras = super().reset(env_ids)
        self.heading_target[resolved_ids] = self.robot.data.heading_w[resolved_ids]
        return extras

    def sync_episode_frame_after_reset(self, env_ids: Sequence[int] | None = None):
        """Synchronize the episode yaw after a reference-state reset.

        Command-conditioned AMP samples commands before ``reset_from_ref`` so
        the matching clip is available to the reset event.  This hook updates
        only the episode frame after that event and intentionally does not
        resample the command.
        """
        resolved_ids = slice(None) if env_ids is None else env_ids
        self.heading_target[resolved_ids] = self.robot.data.heading_w[resolved_ids]

    def compute(self, dt: float):
        super().compute(dt)
        self.heading_target[:] = math_utils.wrap_to_pi(self.heading_target + self.vel_command_b[:, 2] * dt)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        straight_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.straight_command_probability
        straight_ids = env_ids[straight_mask]
        if len(straight_ids) > 0:
            # Respect the active curriculum's forward-speed ceiling while
            # retaining an exact zero lateral/yaw command.
            x_min = max(self.cfg.straight_lin_vel_x_range[0], self.cfg.ranges.lin_vel_x[0])
            x_max = min(self.cfg.straight_lin_vel_x_range[1], self.cfg.ranges.lin_vel_x[1])
            supported = self._restrict_to_motion_support("forward", x_min, x_max)
            if supported is None:
                raise ValueError(f"Forward command range {(x_min, x_max)} does not overlap forward motion speeds")
            x_min, x_max = supported
            self.vel_command_b[straight_ids, 0] = torch.empty(
                len(straight_ids), device=self.device
            ).uniform_(x_min, x_max)
            self.vel_command_b[straight_ids, 1:] = 0.0

        if not self.cfg.single_axis_commands:
            return

        # The reference set contains only pure forward/backward/lateral/turn
        # clips.  Generate the remaining samples from one active axis at a time
        # instead of asking AMP to reconcile a compound command with a pure clip.
        mixed_ids = env_ids[~straight_mask]
        if len(mixed_ids) == 0:
            return

        ranges = self.cfg.ranges
        candidates: list[tuple[str, int, float, float]] = []
        if ranges.lin_vel_x[1] > 0.0:
            candidates.append(("forward", 0, max(0.0, ranges.lin_vel_x[0]), ranges.lin_vel_x[1]))
        if ranges.lin_vel_x[0] < 0.0:
            candidates.append(("backward", 0, ranges.lin_vel_x[0], min(0.0, ranges.lin_vel_x[1])))
        if ranges.lin_vel_y[1] > 0.0:
            candidates.append(("lateral_left", 1, max(0.0, ranges.lin_vel_y[0]), ranges.lin_vel_y[1]))
        if ranges.lin_vel_y[0] < 0.0:
            candidates.append(("lateral_right", 1, ranges.lin_vel_y[0], min(0.0, ranges.lin_vel_y[1])))
        if ranges.ang_vel_z[1] > 0.0:
            candidates.append(("rotate_left", 2, max(0.0, ranges.ang_vel_z[0]), ranges.ang_vel_z[1]))
        if ranges.ang_vel_z[0] < 0.0:
            candidates.append(("rotate_right", 2, ranges.ang_vel_z[0], min(0.0, ranges.ang_vel_z[1])))

        supported_candidates = []
        for category, component, lower, upper in candidates:
            supported = self._restrict_to_motion_support(category, lower, upper)
            if supported is not None:
                supported_candidates.append((category, component, *supported))
        candidates = supported_candidates
        if not candidates:
            self.vel_command_b[mixed_ids] = 0.0
            return

        # When the explicit straight subset is enabled, reserve it as the sole
        # source of forward commands whenever other directions are available.
        if self.cfg.straight_command_probability > 0.0 and len(candidates) > 1:
            candidates = [candidate for candidate in candidates if candidate[0] != "forward"]

        self.vel_command_b[mixed_ids] = 0.0
        choices = torch.randint(len(candidates), (len(mixed_ids),), device=self.device)
        random_values = torch.rand(len(mixed_ids), device=self.device)
        for candidate_index, (_, component, lower, upper) in enumerate(candidates):
            selected = choices == candidate_index
            selected_ids = mixed_ids[selected]
            if len(selected_ids) > 0:
                self.vel_command_b[selected_ids, component] = lower + random_values[selected] * (upper - lower)

    def _restrict_to_motion_support(self, category: str, lower: float, upper: float) -> tuple[float, float] | None:
        """Intersect a signed command interval with its measured motion support."""
        if self.cfg.motion_speed_animation is None:
            return lower, upper
        animation_term = self._env.animation_manager.get_term(self.cfg.motion_speed_animation)
        min_speed, max_speed = animation_term.get_command_category_speed_range(category)
        if lower >= 0.0:
            supported_lower = max(lower, min_speed)
            supported_upper = min(upper, max_speed)
        else:
            supported_lower = max(lower, -max_speed)
            supported_upper = min(upper, -min_speed)
        if supported_upper < supported_lower:
            return None
        return supported_lower, supported_upper


@configclass
class StraightMixtureVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`StraightMixtureVelocityCommand`."""

    class_type: type = StraightMixtureVelocityCommand
    straight_command_probability: float = 0.5
    straight_lin_vel_x_range: tuple[float, float] = (0.2, 0.8)
    single_axis_commands: bool = True
    """Restrict non-straight samples to one demonstrated motion axis."""

    motion_speed_animation: str | None = None
    """Animation term whose measured clip-speed support bounds command sampling."""
