"""Strict restoration of ACCAD34/no-mirror/reduced-ankle BSRL for latent training."""

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config


def validate_baseline(env, agent):
    robot = env["scene"]["robot"]
    if agent["experiment_name"] != "bsrl_amp" or agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Expected estimator-free BSRL AMP baseline")
    if agent["algorithm"]["symmetry_cfg"] is not None or agent["algorithm"]["state_est_loss_required"]:
        raise ValueError("Use no-mirror, no-estimator BSRL baseline")
    if (set(robot["actuators"]) != {"large_motors", "small_motors", "parallel_ankles"}
            or not robot["actuators"]["parallel_ankles"]["class_type"].endswith(":ReducedAnkleActuator")
            or not robot["class_type"].endswith(":ReducedAnkleArticulation")):
        raise ValueError("Expected analytical parallel ankles, not direct serial or explicit closed loops")
    policy = env["observations"]["policy"]
    if policy.get("heading_error") or policy.get("heading_frame_error") or policy["history_length"] != 5:
        raise ValueError("Expected five-frame Actor with no cumulative heading")
    terms = [key for key, value in policy.items() if isinstance(value, dict) and value.get("func")]
    if terms != ["base_ang_vel", "root_local_rot_tan_norm", "velocity_commands", "joint_pos", "joint_vel", "actions"]:
        raise ValueError("Unexpected baseline policy terms")
    if env["observations"].get("state_est_label") is not None:
        raise ValueError("No supervised estimator labels allowed")
    if (env["commands"]["base_velocity"]["heading_command"]
            or not env["commands"]["base_velocity"]["class_type"].endswith(":HeadingReferenceVelocityCommand")
            or env["rewards"]["heading_frame_lin_vel_xy_exp"]["weight"] != 1.0):
        raise ValueError("Expected existing heading-frame reward without an outer heading controller")
    if len(env["motion_data"]["motion_dataset"]["motion_data_weights"]) != 34:
        raise ValueError("Expected full ACCAD34 baseline")


def expected_configuration(saved_env, saved_agent, *, run_name, num_envs, max_iterations, device):
    from legged_lab.latent.bsrl_config import POLICY_OPTIONS, ALGORITHM_CLASS

    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    env["observations"]["policy"]["history_length"] = 10
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent["run_name"] = run_name
    agent["max_iterations"] = max_iterations
    agent["device"] = device
    agent["policy"].update(copy.deepcopy(POLICY_OPTIONS))
    agent["algorithm"]["class_name"] = ALGORITHM_CLASS
    return resolved_scene_config(env), agent


def check_configuration(env_cfg, agent_cfg, baseline_run):
    baseline = Path(baseline_run)
    env, agent = expected_configuration(
        load_saved_config(baseline / "params/env.yaml"), load_saved_config(baseline / "params/agent.yaml"),
        run_name=agent_cfg.run_name, num_envs=env_cfg.scene.num_envs,
        max_iterations=agent_cfg.max_iterations, device=env_cfg.sim.device)
    env["log_dir"] = env_cfg.log_dir
    differences = configuration_differences(env, resolved_scene_config(env_cfg.to_dict()), "env")
    differences += configuration_differences(agent, agent_cfg.to_dict(), "agent")
    if differences:
        raise ValueError(f"Unrequested BSRL latent configuration changes: {differences}")
    return differences


def apply_bsrl_latent(env_cfg, agent_cfg, baseline_run):
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg
    from legged_lab.assets.bsrl import BSRL_CFG
    from legged_lab.assets.bsrl_reduced_ankle import with_reduced_ankles
    from legged_lab.tasks.locomotion.amp.mdp.heading_frame import (
        HeadingReferenceVelocityCommand, heading_frame_error, heading_frame_lin_vel_xy, heading_frame_lin_vel_xy_exp)
    from legged_lab.latent.bsrl_config import enable_bsrl_latent
    from bsrl_accad_ablation import load_dataset, file_hash

    baseline = Path(baseline_run).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    validate_baseline(saved_env, saved_agent)
    if agent_cfg.seed != saved_agent["seed"] or agent_cfg.resume:
        raise ValueError("Start fresh with the baseline seed")
    overrides = {"run_name": agent_cfg.run_name, "num_envs": env_cfg.scene.num_envs,
                 "max_iterations": agent_cfg.max_iterations, "device": env_cfg.sim.device}
    if not overrides["run_name"] or overrides["run_name"] == saved_agent["run_name"]:
        raise ValueError("Use a distinct latent run name")
    if not 1 <= overrides["max_iterations"] <= 25000:
        raise ValueError("Non-explicit-loop latent training is capped at 25000 iterations")
    dataset = saved_env["motion_data"]["motion_dataset"]
    manifest_path = Path(dataset["motion_data_dir"]) / "manifest.json"
    manifest, weights = load_dataset(manifest_path)
    if weights != dataset["motion_data_weights"]:
        raise ValueError("Saved baseline motion weights differ from validated ACCAD manifest")
    # Supply typed templates for optional terms absent in today's task defaults.
    env_cfg.scene.robot = with_reduced_ankles(BSRL_CFG, physics_dt=saved_env["sim"]["dt"])
    env_cfg.commands.base_velocity.class_type = HeadingReferenceVelocityCommand
    env_cfg.rewards.heading_frame_lin_vel_xy_exp = RewardTermCfg(
        func=heading_frame_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.4})
    env_cfg.observations.critic.heading_frame_error = ObservationTermCfg(
        func=heading_frame_error, params={"command_name": "base_velocity"})
    env_cfg.observations.critic.heading_frame_lin_vel_xy = ObservationTermCfg(
        func=heading_frame_lin_vel_xy, params={"command_name": "base_velocity"})
    audit = restore_run_config(env_cfg, agent_cfg, baseline)
    enable_bsrl_latent(env_cfg, agent_cfg)
    agent_cfg.run_name = overrides["run_name"]
    agent_cfg.max_iterations = overrides["max_iterations"]
    agent_cfg.device = env_cfg.sim.device = overrides["device"]
    env_cfg.scene.num_envs = overrides["num_envs"]
    env_cfg.log_dir = None
    check_configuration(env_cfg, agent_cfg, baseline)
    audit.update({"fresh_training": True, "mirror": False, "estimator": False, "actor_heading": False,
        "topology": "serial_skeleton_analytical_parallel_ankles_no_explicit_loops",
        "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": file_hash(manifest_path),
        "dataset_clips": len(weights), "dataset_duration_s": manifest["duration_s"],
        "requested_changes": ["policy history 5 -> 10 frames, 480 raw features",
            "encoder512/256/128 -> deterministic mu20; latest96 + mu20 -> Actor116/2048/1024/512/128/12",
            "next proprioception33 prediction MSE + beta0.01 KL, joint PPO encoder training",
            "internal per-frame normalization, frozen within rollouts/updates", "25000 iteration cap"],
        "launch_overrides": overrides, "critic_rewards_commands_data_randomization_preserved": True})
    return audit


def snapshot_sources(log_dir):
    from t1_latent_ablation import snapshot_sources as snapshot_common

    hashes = snapshot_common(log_dir)
    root = Path(__file__).resolve().parents[2]
    for relative in ("scripts/rsl_rl/bsrl_latent_ablation.py", "scripts/rsl_rl/launch_bsrl_latent.sh",
                     "source/legged_lab/legged_lab/latent/README_bsrl.md", "source/legged_lab/test/test_bsrl_latent.py",
                     "scripts/rsl_rl/bsrl_accad_ablation.py", "source/legged_lab/legged_lab/assets/bsrl_reduced_ankle.py",
                     "source/legged_lab/legged_lab/assets/bsrl_ankle_transmission.py"):
        source = root / relative
        target = Path(log_dir) / "latent_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
