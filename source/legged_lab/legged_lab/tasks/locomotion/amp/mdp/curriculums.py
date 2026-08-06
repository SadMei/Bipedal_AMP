from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_range_and_tracking_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    reward_term_name: str,
    steps_per_iteration: int,
    phase_boundaries: tuple[int, int],
    lin_vel_x_ranges: tuple[tuple[float, float], ...],
    lin_vel_y_ranges: tuple[tuple[float, float], ...],
    ang_vel_z_ranges: tuple[tuple[float, float], ...],
    tracking_stds: tuple[float, ...],
) -> dict[str, float]:
    """Synchronize command ranges and velocity-reward width over training phases."""
    del env_ids

    phase_count = len(tracking_stds)
    schedules = (lin_vel_x_ranges, lin_vel_y_ranges, ang_vel_z_ranges)
    if phase_count != 3 or any(len(schedule) != phase_count for schedule in schedules):
        raise ValueError("Velocity curriculum requires exactly three complete phases.")
    if steps_per_iteration <= 0:
        raise ValueError("steps_per_iteration must be positive.")

    iteration = env.common_step_counter // steps_per_iteration
    if iteration < phase_boundaries[0]:
        phase = 0
    elif iteration < phase_boundaries[1]:
        phase = 1
    else:
        phase = 2

    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    ranges.lin_vel_x = lin_vel_x_ranges[phase]
    ranges.lin_vel_y = lin_vel_y_ranges[phase]
    ranges.ang_vel_z = ang_vel_z_ranges[phase]

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    target_std = tracking_stds[phase]
    if reward_cfg.params["std"] != target_std:
        reward_cfg.params["std"] = target_std
        env.reward_manager.set_term_cfg(reward_term_name, reward_cfg)

    return {
        "phase": float(phase + 1),
        "iteration": float(iteration),
        "lin_vel_x_max": float(lin_vel_x_ranges[phase][1]),
        "lin_vel_y_max": float(lin_vel_y_ranges[phase][1]),
        "ang_vel_z_max": float(ang_vel_z_ranges[phase][1]),
        "tracking_std": float(target_std),
    }
