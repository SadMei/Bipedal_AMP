"""Frozen 9/15 T1 baseline -> predictive latent-only, no-mirror branch."""

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import restore_run_config
from t1_heading_ablation import load_saved_config, configuration_differences, resolved_scene_config


def expected_configuration(saved_env, saved_agent, *, run_name, num_envs, max_iterations, device):
    from legged_lab.latent.config import POLICY_OPTIONS, ALGORITHM_CLASS

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
    agent["algorithm"]["symmetry_cfg"] = None
    return resolved_scene_config(env), agent


def apply_t1_latent(env_cfg, agent_cfg, baseline_run):
    from legged_lab.latent.config import enable_latent

    baseline = Path(baseline_run).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    if saved_agent["experiment_name"] != "men_t1_amp" or saved_agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Expected the estimator-free T1 AMP run")
    if saved_env["observations"]["policy"].get("heading_error") is not None:
        raise ValueError("Use the 9/15 no-actor-heading baseline")
    if agent_cfg.seed != saved_agent["seed"] or agent_cfg.resume:
        raise ValueError("This experiment starts fresh with the baseline seed")
    overrides = {"run_name": agent_cfg.run_name, "num_envs": env_cfg.scene.num_envs,
                 "max_iterations": agent_cfg.max_iterations, "device": env_cfg.sim.device}
    if not overrides["run_name"] or overrides["run_name"] == saved_agent["run_name"]:
        raise ValueError("Use a distinct latent run name")
    if not 1 <= overrides["max_iterations"] <= 25000:
        raise ValueError("This experiment is capped at 25000 iterations")
    audit = restore_run_config(env_cfg, agent_cfg, baseline)
    enable_latent(env_cfg, agent_cfg)
    agent_cfg.run_name = overrides["run_name"]
    agent_cfg.max_iterations = overrides["max_iterations"]
    agent_cfg.device = env_cfg.sim.device = overrides["device"]
    env_cfg.scene.num_envs = overrides["num_envs"]
    env_cfg.log_dir = None
    expected_env, expected_agent = expected_configuration(saved_env, saved_agent, **overrides)
    differences = configuration_differences(expected_env, resolved_scene_config(env_cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent_cfg.to_dict(), "agent")
    if differences:
        raise ValueError(f"Unrequested latent comparison differences: {differences}")
    audit.update({"fresh_training": True, "requested_changes": [
        "policy history 5 -> 10 frames; latest two + deterministic 20D latent",
        "internal per-frame empirical normalization; updated only between rollouts",
        "encoder512/256/128; Actor2048/1024/512/128; next proprioception MSE + beta0.01 KL",
        "mirror loss and augmentation disabled as requested",
        "25000 iteration cap (shorter runs permitted for smoke tests)"],
        "launch_overrides": overrides, "unrequested_configuration_differences": differences,
        "critic_rewards_commands_data_randomization_preserved": True})
    return audit


def check_runtime_configuration(env_cfg, agent_cfg, baseline_run):
    """Repeat the full guard after Isaac manager/scene initialization."""
    baseline = Path(baseline_run)
    expected_env, expected_agent = expected_configuration(
        load_saved_config(baseline / "params/env.yaml"),
        load_saved_config(baseline / "params/agent.yaml"),
        run_name=agent_cfg.run_name, num_envs=env_cfg.scene.num_envs,
        max_iterations=agent_cfg.max_iterations, device=env_cfg.sim.device)
    expected_env["log_dir"] = env_cfg.log_dir
    differences = configuration_differences(expected_env, env_cfg.to_dict(), "env")
    differences += configuration_differences(expected_agent, agent_cfg.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Unexpected post-initialization configuration changes: {differences}")
    return differences


def snapshot_sources(log_dir):
    """Include opt-in/untracked experiment code that git diff cannot capture."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "source/legged_lab/legged_lab/latent").glob("*.py"))
    paths += [root / path for path in (
        "source/legged_lab/legged_lab/latent/README.md",
        "source/legged_lab/test/test_t1_latent.py", "scripts/rsl_rl/t1_latent_ablation.py",
        "scripts/rsl_rl/frozen_run_config.py", "scripts/rsl_rl/t1_heading_ablation.py",
        "scripts/rsl_rl/train.py", "scripts/rsl_rl/launch_t1_latent.sh",
        "external/rsl_rl/rsl_rl/algorithms/ppo_amp.py", "external/rsl_rl/rsl_rl/runners/amp_runner.py")]
    hashes = {}
    for source in paths:
        relative = source.relative_to(root)
        target = Path(log_dir) / "latent_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
