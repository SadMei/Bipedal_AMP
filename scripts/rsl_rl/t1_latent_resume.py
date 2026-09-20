"""Strict continuation helpers: total-update cap, curriculum clock and weights."""

from __future__ import annotations

import copy
import math

from t1_heading_ablation import configuration_differences, resolved_scene_config


def continuation_counts(checkpoint_iteration: int, total_iterations: int) -> tuple[int, int]:
    """AMPRunner stores the last completed zero-based iteration, not the next."""
    completed = int(checkpoint_iteration) + 1
    if checkpoint_iteration < 0 or not completed < total_iterations <= 40000:
        raise ValueError(f"Require completed={completed} < total_iterations <= 40000.")
    return completed, total_iterations - completed


def expected_configuration(saved_env, saved_agent, *, checkpoint, run_name, total_iterations,
                           num_envs, device, log_dir, completed, apply_curriculum):
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    if (agent["policy"]["class_name"] != "legged_lab.latent.models:LatentActorCritic"
            or agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"
            or agent["algorithm"]["symmetry_cfg"] is not None
            or agent["algorithm"]["state_est_loss_required"]
            or env["observations"]["policy"].get("heading_error") is not None):
        raise ValueError("Require a no-mirror/no-estimator/no-actor-heading T1 latent checkpoint.")
    if num_envs < 1 or num_envs > saved_env["scene"]["num_envs"]:
        raise ValueError("Environment count may only match the baseline or be reduced for a smoke test.")
    if not run_name or run_name == saved_agent["run_name"]:
        raise ValueError("A distinct continuation run name is required.")
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = log_dir
    agent.update(run_name=run_name, max_iterations=total_iterations, device=device,
                 resume=True, load_run=str(checkpoint.parent), load_checkpoint=checkpoint.name)
    if apply_curriculum:
        params = env["curriculum"]["velocity_range_and_tracking_std"]["params"]
        if params["steps_per_iteration"] != agent["num_steps_per_env"]:
            raise ValueError("Curriculum and PPO clocks disagree.")
        phase = sum(completed >= boundary for boundary in params["phase_boundaries"])
        ranges = env["commands"][params["command_name"]]["ranges"]
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            ranges[key] = params[key + "_ranges"][phase]
        env["rewards"][params["reward_term_name"]]["params"]["std"] = params["tracking_stds"][phase]
    return resolved_scene_config(env), agent


def check_configuration(env_cfg, agent_cfg, saved_env, saved_agent, **options):
    env, agent = expected_configuration(saved_env, saved_agent, **options)
    changes = configuration_differences(env, resolved_scene_config(env_cfg.to_dict()), "env")
    changes += configuration_differences(agent, agent_cfg.to_dict(), "agent")
    if changes:
        raise RuntimeError(f"Unrequested continuation configuration changes: {changes}")
    return changes


def restore_training_clock(env, completed: int, steps_per_iteration: int) -> None:
    """Call BEFORE wrapper.reset() so the first command uses the correct phase."""
    env.common_step_counter = completed * steps_per_iteration
    env._sim_step_counter = env.common_step_counter * env.cfg.decimation


def assert_state_equal(actual, expected, path="state") -> None:
    """Verify models AND Adam moments, not just compatible tensor shapes."""
    import torch

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0,
                                   msg=lambda message: f"{path}: {message}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError(f"{path}: keys differ")
        for key in expected:
            assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise RuntimeError(f"{path}: length differs")
        for index, value in enumerate(expected):
            assert_state_equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise RuntimeError(f"{path}: {actual!r} != {expected!r}")


def audit_loaded_runner(runner, checkpoint, completed):
    pairs = {
        "model_state_dict": runner.alg.policy.state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "amp_discriminator_state_dict": runner.alg.amp_discriminator.state_dict(),
        "amp_discriminator_normalizer_state_dict": runner.alg.amp_discriminator.disc_obs_normalizer.state_dict(),
        "amp_discriminator_optimizer_state_dict": runner.alg.disc_optimizer.state_dict(),
    }
    for key, actual in pairs.items():
        assert_state_equal(actual, checkpoint[key], key)
    if runner.current_learning_iteration != checkpoint["iter"]:
        raise RuntimeError("Runner did not load the checkpoint iteration.")
    rates = {float(group["lr"]) for group in runner.alg.optimizer.param_groups}
    if len(rates) != 1 or not all(math.isfinite(rate) and rate > 0 for rate in rates):
        raise RuntimeError(f"Unexpected restored PPO learning rates: {rates}")
    # AMPRunner.load restores Adam, but its separate adaptive-schedule scalar
    # would otherwise retain the config's initial 1e-4 instead of saved 1e-5.
    runner.alg.learning_rate = rates.pop()
    runner.current_learning_iteration = completed
    return {"exactly_restored_states": list(pairs), "next_iteration": completed,
            "ppo_learning_rate": runner.alg.learning_rate,
            "discriminator_learning_rates": [g["lr"] for g in runner.alg.disc_optimizer.param_groups],
            "replay_buffers_restored": False, "simulator_or_rng_state_restored": False}
