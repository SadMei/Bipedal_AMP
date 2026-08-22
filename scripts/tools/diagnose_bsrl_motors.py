"""Validate BSRL motor grouping and torque-speed-power limits in simulation."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--rollout_steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli, fast_shutdown=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import legged_lab.tasks  # noqa: F401
from legged_lab.assets.bsrl import (
    BSRL_LARGE_MOTOR_PEAK_POWER,
    BSRL_LARGE_MOTOR_PEAK_SPEED,
    BSRL_LARGE_MOTOR_PEAK_TORQUE,
    BSRL_SMALL_MOTOR_PEAK_POWER,
    BSRL_SMALL_MOTOR_PEAK_SPEED,
    BSRL_SMALL_MOTOR_PEAK_TORQUE,
)


TASK_NAME = "LeggedLab-Isaac-AMP-BSRL-v0"
EXPECTED = {
    "large_motors": {
        "suffixes": ("hip_roll", "hip_pitch", "knee_pitch"),
        "torque": BSRL_LARGE_MOTOR_PEAK_TORQUE,
        "speed": BSRL_LARGE_MOTOR_PEAK_SPEED,
        "power": BSRL_LARGE_MOTOR_PEAK_POWER,
    },
    "small_motors": {
        "suffixes": ("hip_yaw", "ankle_pitch", "ankle_roll"),
        "torque": BSRL_SMALL_MOTOR_PEAK_TORQUE,
        "speed": BSRL_SMALL_MOTOR_PEAK_SPEED,
        "power": BSRL_SMALL_MOTOR_PEAK_POWER,
    },
}


def _assert_close(name: str, actual: torch.Tensor, expected: float) -> None:
    target = torch.full_like(actual, expected)
    if not torch.allclose(actual, target, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError(f"{name}: expected {expected}, got range {actual.min().item()}..{actual.max().item()}")


def _check_grouping_and_limits(robot) -> None:
    assigned_names: list[str] = []
    for group_name, expected in EXPECTED.items():
        actuator = robot.actuators[group_name]
        joint_names = tuple(actuator.joint_names)
        expected_names = {
            f"joint_{side}_{suffix}" for side in ("left", "right") for suffix in expected["suffixes"]
        }
        if set(joint_names) != expected_names:
            raise RuntimeError(f"{group_name}: wrong joints: {joint_names}")
        assigned_names.extend(joint_names)

        _assert_close(f"{group_name}.effort_limit", actuator.effort_limit, expected["torque"])
        _assert_close(f"{group_name}.velocity_limit", actuator.velocity_limit, expected["speed"])
        if abs(actuator._peak_power - expected["power"]) > 1.0e-6:
            raise RuntimeError(f"{group_name}: wrong peak power {actuator._peak_power}")

        # Directly exercise both motoring and regenerative/braking quadrants at
        # zero, corner, rated, and over-speed operating points.
        speed_samples = torch.tensor(
            [
                -1.1 * expected["speed"],
                -expected["speed"],
                -0.5 * expected["speed"],
                0.0,
                0.5 * expected["speed"],
                expected["speed"],
                1.1 * expected["speed"],
            ],
            device=robot.device,
        )
        actuator._joint_vel.zero_()
        actuator._joint_vel[: speed_samples.numel(), :] = speed_samples[:, None]
        demanded = torch.full_like(actuator._joint_vel, 1.0e6)
        positive = actuator._clip_effort(demanded)
        negative = actuator._clip_effort(-demanded)
        for label, effort in (("positive", positive), ("negative", negative)):
            velocity = actuator._joint_vel
            if effort.abs().max() > expected["torque"] + 1.0e-4:
                raise RuntimeError(f"{group_name}/{label}: peak torque limit failed")
            if (effort.mul(velocity).abs() > expected["power"] + 1.0e-3).any():
                raise RuntimeError(f"{group_name}/{label}: peak power limit failed")
            accelerating_overspeed = (velocity.abs() >= expected["speed"]) & (effort * velocity > 1.0e-6)
            if accelerating_overspeed.any():
                raise RuntimeError(f"{group_name}/{label}: motor accelerates beyond peak speed")

        print(
            f"[motor] {group_name}: joints={joint_names} torque={expected['torque']:.3f} Nm "
            f"speed={expected['speed']:.4f} rad/s power={expected['power']:.1f} W"
        )

    if len(assigned_names) != 12 or len(set(assigned_names)) != 12 or set(assigned_names) != set(robot.joint_names):
        raise RuntimeError("the two motor groups do not cover every BSRL joint exactly once")


def _rollout(env) -> None:
    robot = env.scene["robot"]
    max_effort = {name: 0.0 for name in EXPECTED}
    max_power = {name: 0.0 for name in EXPECTED}
    terminations = 0
    timeouts = 0

    for step in range(args_cli.rollout_steps):
        actions = torch.empty(env.num_envs, env.action_manager.total_action_dim, device=env.device).uniform_(-1.0, 1.0)
        _, reward, terminated, timeout, _ = env.step(actions)
        if not torch.isfinite(reward).all() or not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError(f"non-finite state or reward at rollout step {step}")
        terminations += int(terminated.sum())
        timeouts += int(timeout.sum())

        for group_name, expected in EXPECTED.items():
            actuator = robot.actuators[group_name]
            effort = actuator.applied_effort
            velocity = actuator._joint_vel
            power = (effort * velocity).abs()
            if not torch.isfinite(effort).all() or not torch.isfinite(power).all():
                raise RuntimeError(f"{group_name}: non-finite actuator output at step {step}")
            if effort.abs().max() > expected["torque"] + 1.0e-4:
                raise RuntimeError(f"{group_name}: torque limit violated at step {step}")
            if power.max() > expected["power"] + 1.0e-3:
                raise RuntimeError(f"{group_name}: power limit violated at step {step}")
            max_effort[group_name] = max(max_effort[group_name], effort.abs().max().item())
            max_power[group_name] = max(max_power[group_name], power.max().item())

    for group_name in EXPECTED:
        actuator = robot.actuators[group_name]
        lags = actuator.positions_delay_buffer.time_lags
        if lags.min() < actuator.cfg.min_delay or lags.max() > actuator.cfg.max_delay:
            raise RuntimeError(f"{group_name}: sampled delay is outside configured limits")
        print(
            f"[rollout] {group_name}: max_torque={max_effort[group_name]:.4f} Nm "
            f"max_power={max_power[group_name]:.4f} W delay_values={lags.unique().sort().values.tolist()}"
        )
    print(f"[rollout] steps={args_cli.rollout_steps} terminations={terminations} timeouts={timeouts}")


def main() -> None:
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
    try:
        env.reset()
        _check_grouping_and_limits(env.scene["robot"])
        _rollout(env)
        print("[diagnostic] PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
