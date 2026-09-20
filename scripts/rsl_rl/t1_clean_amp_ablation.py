"""Opt-in clean-AMP comparison against the saved T1 latent baseline.

Keep the latent network, motion files/weights, PD, ordinary rewards and domain
randomization. Disable the requested heading/command-conditioning additions
with None/False instead of deleting the historical implementations.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import _restore_value, restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config


UNIFORM_COMMAND = "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand"
COMMAND_RANGES = {"lin_vel_x": (-0.8, 0.8), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.5, 1.5)}
HEADING_REWARDS = ("zero_yaw_initial_heading_lin_vel_xy_exp", "heading_frame_lin_vel_xy_exp",
                   "zero_command_heading_hold_exp")
HEADING_OBSERVATIONS = ("heading_error", "heading_frame_error", "heading_frame_lin_vel_xy")


def validate_baseline(env, agent):
    if (agent["experiment_name"] != "men_t1_amp"
            or agent["policy"]["class_name"] != "legged_lab.latent.models:LatentActorCritic"
            or agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"):
        raise ValueError("Expected the frozen T1 latent training run")
    if agent["algorithm"]["state_est_loss_required"] or env["observations"].get("state_est_label"):
        raise ValueError("The comparison baseline must not have a supervised estimator")
    if env["observations"]["policy"]["history_length"] != 10:
        raise ValueError("The comparison preserves the ten-frame latent actor")


def expected_configuration(saved_env, saved_agent, *, run_name, num_envs, max_iterations, device):
    validate_baseline(saved_env, saved_agent)
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    # Disable the extra initial/integrated-heading rewards. Body-frame linear
    # and yaw-rate tracking, roll/pitch regularization, etc. remain untouched.
    for key in HEADING_REWARDS:
        if key in env["rewards"]:
            env["rewards"][key] = None
    for group_name in ("policy", *agent["obs_groups"]["critic"]):
        for key in HEADING_OBSERVATIONS:
            if key in env["observations"][group_name]:
                env["observations"][group_name][key] = None
    # Ordinary unconditional AMP: remove command features on BOTH discriminator
    # sides, retaining actual measured/reference linear and angular velocities.
    env["observations"]["disc"]["velocity_commands"] = None
    env["observations"]["disc_demo"]["ref_velocity_commands"] = None
    animation = env["animation"]["animation"]
    animation["command_name"] = None
    animation["command_motion_patterns"] = None
    animation["match_command_speed"] = False
    command = env["commands"]["base_velocity"]
    command["class_type"] = UNIFORM_COMMAND  # no inherited yaw integration
    command["straight_command_probability"] = 0.0
    command["single_axis_commands"] = False
    command["motion_speed_animation"] = None
    command["heading_command"] = False
    command["rel_heading_envs"] = 0.0
    command["ranges"].update(COMMAND_RANGES)
    command["ranges"]["heading"] = None
    # The old course begins with forward-only commands and rewrites all three
    # ranges. Disable it so the requested ranges apply from the very first step.
    course = env["curriculum"]["velocity_range_and_tracking_std"]["params"]
    if len(set(course["tracking_stds"])) != 1:
        raise ValueError("Baseline tracking width changes by phase; specify which width to preserve")
    # Saved env.yaml precedes the first curriculum reset and says std=0.5.
    # Actual baseline training uses 0.4 in ALL phases: retain that active value.
    env["rewards"][course["reward_term_name"]]["params"]["std"] = course["tracking_stds"][0]
    env["curriculum"]["velocity_range_and_tracking_std"] = None
    agent["algorithm"]["symmetry_cfg"] = None
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent["run_name"] = run_name
    agent["max_iterations"] = max_iterations
    agent["device"] = device
    return resolved_scene_config(env), agent


def check_configuration(env_cfg, agent_cfg, baseline_run):
    baseline = Path(baseline_run)
    expected_env, expected_agent = expected_configuration(
        load_saved_config(baseline / "params/env.yaml"), load_saved_config(baseline / "params/agent.yaml"),
        run_name=agent_cfg.run_name, num_envs=env_cfg.scene.num_envs,
        max_iterations=agent_cfg.max_iterations, device=env_cfg.sim.device)
    expected_env["log_dir"] = env_cfg.log_dir
    differences = configuration_differences(expected_env, resolved_scene_config(env_cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent_cfg.to_dict(), "agent")
    if differences:
        raise ValueError(f"Unexpected clean-AMP comparison changes: {differences}")
    return differences


def apply_clean_amp(env_cfg, agent_cfg, baseline_run):
    from legged_lab.latent.config import POLICY_OPTIONS

    baseline = Path(baseline_run).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    validate_baseline(saved_env, saved_agent)
    if agent_cfg.resume or agent_cfg.seed != saved_agent["seed"]:
        raise ValueError("Start this comparison from scratch with the baseline seed")
    overrides = dict(run_name=agent_cfg.run_name, num_envs=env_cfg.scene.num_envs,
                     max_iterations=agent_cfg.max_iterations, device=env_cfg.sim.device)
    if not overrides["run_name"] or overrides["run_name"] == saved_agent["run_name"]:
        raise ValueError("Use a distinct clean-AMP run name")
    if not 1 <= overrides["max_iterations"] <= 40000:
        raise ValueError("This comparison is capped at the requested 40000 iterations")
    # The default registered task has no latent fields. Add compatible typed
    # templates before exact restoration, without enabling another ablation.
    for key, value in POLICY_OPTIONS.items():
        setattr(agent_cfg.policy, key, copy.deepcopy(value))
    audit = restore_run_config(env_cfg, agent_cfg, baseline)
    expected_env, expected_agent = expected_configuration(saved_env, saved_agent, **overrides)
    _restore_value(env_cfg, expected_env, "env")
    _restore_value(agent_cfg, expected_agent, "agent")
    differences = check_configuration(env_cfg, agent_cfg, baseline)
    requested = configuration_differences(saved_env, expected_env, "env")
    requested += configuration_differences(saved_agent, expected_agent, "agent")
    audit.update(fresh_training=True, latent_preserved=True, mirror=False, estimator=False,
                 actor_heading=False, critic_heading=False, command_conditioned_amp=False,
                 command_ranges=COMMAND_RANGES, velocity_curriculum=False,
                 launch_overrides=overrides, requested_configuration_differences=requested,
                 unrequested_configuration_differences=differences)
    return audit


def runtime_audit(env, agent_cfg, baseline_run):
    import torch
    from isaaclab.envs.mdp.commands import UniformVelocityCommand

    check_configuration(env.cfg, agent_cfg, baseline_run)
    manager = env.observation_manager
    dims = manager.group_obs_dim
    critic_width = sum(dims[key][0] for key in agent_cfg.obs_groups["critic"])
    if tuple(dims["policy"]) != (870,) or critic_width != 585:
        raise RuntimeError(f"Unexpected clean-AMP Actor/Critic dimensions: {dims}, critic={critic_width}")
    for group in ("disc", "disc_demo"):
        if tuple(dims[group]) != (4, 80):
            raise RuntimeError(f"Unexpected unconditional discriminator dimensions: {dims}")
    for group in ("policy", *agent_cfg.obs_groups["critic"]):
        if set(manager.active_terms[group]) & set(HEADING_OBSERVATIONS):
            raise RuntimeError(f"Heading observation still active in {group}")
    if "velocity_commands" in manager.active_terms["disc"] or "ref_velocity_commands" in manager.active_terms["disc_demo"]:
        raise RuntimeError("Discriminator still observes commands")
    if set(env.reward_manager.active_terms) & set(HEADING_REWARDS):
        raise RuntimeError("Extra heading reward is still active")
    cmd = env.command_manager.get_term("base_velocity")
    if type(cmd) is not UniformVelocityCommand:
        raise RuntimeError("Expected standard uniform velocity commands, without a heading integrator")
    animation = env.animation_manager.get_term("animation")
    if animation._command_category_motion_ids is not None:
        raise RuntimeError("References are still selected by command category")
    commands = cmd.command
    for i, (name, (lower, upper)) in enumerate(COMMAND_RANGES.items()):
        if not bool(torch.all((commands[:, i] >= lower) & (commands[:, i] <= upper))):
            raise RuntimeError(f"Command samples exceed {name} bounds")
    exact_straight = ((commands[:, 1] == 0) & (commands[:, 2] == 0)).float().mean().item()
    if exact_straight >= 0.05:
        raise RuntimeError(f"Unexpected pure-straight atom in command sampling: {exact_straight}")
    return {"policy_dim": 870, "critic_dim": critic_width, "disc_shape": [4, 80],
            "policy_terms": manager.active_terms["policy"],
            "critic_terms": {key: manager.active_terms[key] for key in agent_cfg.obs_groups["critic"]},
            "disc_terms": manager.active_terms["disc"], "disc_demo_terms": manager.active_terms["disc_demo"],
            "command_class": UNIFORM_COMMAND, "sample_min": commands.min(0).values.tolist(),
            "sample_max": commands.max(0).values.tolist(), "sample_mean": commands.mean(0).tolist(),
            "exact_straight_fraction": exact_straight, "command_conditioned_reference": False,
            "mirror": False, "estimator": False, "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
            "unrequested_configuration_differences": []}


def snapshot_sources(log_dir):
    from t1_latent_ablation import snapshot_sources as snapshot_latent

    hashes = snapshot_latent(log_dir)
    root = Path(__file__).resolve().parents[2]
    files = ["scripts/rsl_rl/t1_clean_amp_ablation.py", "scripts/rsl_rl/launch_t1_clean_amp.sh",
             "source/legged_lab/test/test_t1_clean_amp_ablation.py",
             "source/legged_lab/legged_lab/managers/animation_manager.py",
             "source/legged_lab/legged_lab/envs/manager_based_amp_env.py",
             "source/legged_lab/legged_lab/tasks/locomotion/amp/config/t1/t1_amp_env_cfg.py",
             "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands.py",
             "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/observations.py"]
    for relative in files:
        source = root / relative
        target = Path(log_dir) / "latent_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
