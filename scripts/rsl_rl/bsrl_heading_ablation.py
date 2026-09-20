"""Strict opt-in heading-reward experiment on a frozen reduced-ankle run."""
from __future__ import annotations

import copy
from pathlib import Path

from frozen_run_config import restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config

MODULE = "legged_lab.tasks.locomotion.amp.mdp.heading_frame"


def expected_heading_configuration(saved_env, saved_agent, run_name):
    """Describe the complete whitelist of intended config differences."""
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    env["commands"]["base_velocity"]["class_type"] = MODULE + ":HeadingReferenceVelocityCommand"
    env["rewards"]["heading_frame_lin_vel_xy_exp"] = {
        "func": MODULE + ":heading_frame_lin_vel_xy_exp", "params": {"command_name": "base_velocity", "std": .4},
        "weight": 1.,
    }
    for name in ("heading_frame_error", "heading_frame_lin_vel_xy"):
        env["observations"]["critic"][name] = {
            "func": MODULE + ":" + name, "params": {"command_name": "base_velocity"},
            "modifiers": None, "noise": None, "clip": None, "scale": None,
            "history_length": 0, "flatten_history_dim": True,
        }
    agent["run_name"] = run_name
    return env, agent


def validate_reduced_heading_baseline(saved_env, saved_agent):
    if saved_agent["experiment_name"] != "bsrl_amp" or saved_agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Expected estimator-free BSRL AMP")
    robot = saved_env["scene"]["robot"]
    if set(robot["actuators"]) != {"large_motors", "small_motors", "parallel_ankles"}:
        raise ValueError("Expected analytical reduced ankles, not serial-only or explicit loops")
    if not robot["actuators"]["parallel_ankles"]["class_type"].endswith(":ReducedAnkleActuator"):
        raise ValueError("Expected ReducedAnkleActuator")
    command = saved_env["commands"]["base_velocity"]
    if not command["class_type"].endswith(":UniformVelocityCommand") or command["heading_command"]:
        raise ValueError("Expected unchanged uniform body-frame velocity sampling")
    policy = saved_env["observations"]["policy"]
    if policy.get("heading_error") is not None or policy.get("heading_frame_error") is not None:
        raise ValueError("Actor must remain without heading observations")
    if saved_env["observations"].get("state_est_label") is not None:
        raise ValueError("Do not enable an estimator")
    if saved_env["rewards"].get("heading_frame_lin_vel_xy_exp") is not None:
        raise ValueError("Baseline already contains the proposed reward")
    symmetry = saved_agent["algorithm"].get("symmetry_cfg")
    if (not symmetry or not symmetry["use_mirror_loss"] or symmetry["use_data_augmentation"]
            or symmetry["mirror_loss_coeff"] != .1):
        raise ValueError("Keep baseline mirror loss 0.1, without data augmentation")
    if saved_agent["resume"]:
        raise ValueError("Fresh training required")


def check_heading_configuration(expected_env, expected_agent, actual_env, actual_agent):
    expected_env = copy.deepcopy(expected_env)
    expected_env["log_dir"] = actual_env.get("log_dir")
    differences = configuration_differences(expected_env, resolved_scene_config(actual_env), "env")
    differences += configuration_differences(expected_agent, actual_agent, "agent")
    if differences:
        raise ValueError(f"Unexpected change beyond heading reward/critic state: {differences}")


def apply_bsrl_reduced_heading(env_cfg, agent_cfg, baseline_run):
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg
    from legged_lab.assets.bsrl import BSRL_CFG
    from legged_lab.assets.bsrl_reduced_ankle import with_reduced_ankles
    from legged_lab.tasks.locomotion.amp.mdp.heading_frame import (
        HeadingReferenceVelocityCommand, heading_frame_error, heading_frame_lin_vel_xy, heading_frame_lin_vel_xy_exp,
    )

    baseline = Path(baseline_run).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    validate_reduced_heading_baseline(saved_env, saved_agent)
    run_name = agent_cfg.run_name
    if not run_name or run_name == saved_agent["run_name"]:
        raise ValueError("A distinct --run_name is required")
    for name, value, expected in (
        ("num_envs", env_cfg.scene.num_envs, saved_env["scene"]["num_envs"]),
        ("seed", agent_cfg.seed, saved_agent["seed"]),
        ("max_iterations", agent_cfg.max_iterations, saved_agent["max_iterations"]),
    ):
        if value != expected:
            raise ValueError(f"Keep baseline {name}: {value} != {expected}")
    if agent_cfg.resume:
        raise ValueError("This comparison must start from scratch")
    env_cfg.scene.robot = with_reduced_ankles(BSRL_CFG, physics_dt=saved_env["sim"]["dt"])
    audit = restore_run_config(env_cfg, agent_cfg, baseline)
    agent_cfg.run_name = run_name
    env_cfg.commands.base_velocity.class_type = HeadingReferenceVelocityCommand
    env_cfg.rewards.heading_frame_lin_vel_xy_exp = RewardTermCfg(
        func=heading_frame_lin_vel_xy_exp, weight=1., params={"command_name": "base_velocity", "std": .4})
    env_cfg.observations.critic.heading_frame_error = ObservationTermCfg(
        func=heading_frame_error, params={"command_name": "base_velocity"})
    env_cfg.observations.critic.heading_frame_lin_vel_xy = ObservationTermCfg(
        func=heading_frame_lin_vel_xy, params={"command_name": "base_velocity"})
    expected_env, expected_agent = expected_heading_configuration(saved_env, saved_agent, run_name)
    check_heading_configuration(expected_env, expected_agent, env_cfg.to_dict(), agent_cfg.to_dict())
    audit.update({
        "fresh_training": True, "topology": "serial_skeleton_analytical_parallel_ankles_no_explicit_loops",
        "actor_observation_dim": 240, "critic_observation_dim": 370,
        "actor_heading": False, "mirror_loss_coeff": .1,
        "new_reward_weight": 1., "new_reward_std": .4,
        "heading_reference": "reset yaw + integral of executed commanded yaw rate; physical-step clock",
        "uniform_command_sampling_unchanged": True,
        "only_training_changes": ["heading reference bookkeeping without command changes",
                                  "additional heading-frame velocity reward (all commands)",
                                  "critic-only heading sin/cos + reference-frame XY velocity, five-frame history"],
    })
    return audit
