"""T1 latent AMP baseline with deployment-like default-pose reset coverage."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import _restore_value, restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config


DEFAULT_START_PROBABILITY = 0.2
MIXED_RESET = "legged_lab.tasks.locomotion.amp.mdp.events:reset_from_ref_with_default_mixture"


def expected_configuration(saved_env, saved_agent, *, run_name, num_envs, max_iterations, device):
    if (saved_agent["experiment_name"] != "men_t1_amp"
            or saved_agent["policy"]["class_name"] != "legged_lab.latent.models:LatentActorCritic"
            or saved_agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"):
        raise ValueError("Expected the completed no-mirror T1 latent AMP configuration")
    if (saved_agent["algorithm"]["symmetry_cfg"] is not None
            or saved_agent["algorithm"]["state_est_loss_required"]
            or saved_env["observations"]["policy"].get("heading_error") is not None):
        raise ValueError("Require no mirror, estimator, or Actor heading")
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    reset = env["events"]["reset_from_ref"]
    reset["func"] = MIXED_RESET
    reset["params"]["default_start_probability"] = DEFAULT_START_PROBABILITY
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device,
                 resume=False, load_run=".*", load_checkpoint="model_.*.pt")
    return resolved_scene_config(env), agent


def apply_startup_ablation(cfg, agent, baseline, *, run_name, num_envs, max_iterations, device):
    baseline = Path(baseline).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    audit = restore_run_config(cfg, agent, baseline)
    expected_env, expected_agent = expected_configuration(
        saved_env, saved_agent, run_name=run_name, num_envs=num_envs,
        max_iterations=max_iterations, device=device)
    _restore_value(cfg, expected_env, "env")
    _restore_value(agent, expected_agent, "agent")
    differences = configuration_differences(expected_env, resolved_scene_config(cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Unexpected T1 startup-ablation differences: {differences}")
    audit.update(fresh_training=True, only_training_change={
        "events.reset_from_ref.func": MIXED_RESET,
        "events.reset_from_ref.params.default_start_probability": DEFAULT_START_PROBABILITY,
    }, mirror=False, estimator=False, actor_heading=False,
        steady_state_network_rewards_commands_pd_scale_preserved=True,
        requested_configuration_differences=(
            configuration_differences(saved_env, expected_env, "env")
            + configuration_differences(saved_agent, expected_agent, "agent")),
        unrequested_configuration_differences=differences)
    return audit


def runtime_audit(env, agent):
    dims = env.observation_manager.group_obs_dim
    critic_width = sum(dims[key][0] for key in agent.obs_groups["critic"])
    reset_cfg = env.cfg.events.reset_from_ref
    if tuple(dims["policy"]) != (870,) or critic_width != 595:
        raise RuntimeError(f"Unexpected T1 observation contract: {dims}, critic={critic_width}")
    if reset_cfg.func.__name__ != "reset_from_ref_with_default_mixture":
        raise RuntimeError("Mixed reset was not installed")
    count, total = env._amp_default_start_last_count, env._amp_default_start_last_total
    fraction = count / total
    if not 0.15 <= fraction <= 0.25:
        raise RuntimeError(f"Initial default-start fraction outside statistical guard: {fraction}")
    return dict(policy_width=870, critic_width=critic_width, history_frames=10,
                latent_dim=20, default_start_probability=DEFAULT_START_PROBABILITY,
                observed_initial_default_starts=count, observed_initial_total=total,
                observed_initial_fraction=fraction, mirror=False, estimator=False,
                actor_heading=False, physics_dt=env.physics_dt, policy_dt=env.step_dt)


def snapshot_sources(log_dir):
    root = Path(__file__).resolve().parents[2]
    files = ["scripts/rsl_rl/t1_startup_ablation.py",
             "scripts/rsl_rl/train_t1_startup_ablation.py",
             "scripts/rsl_rl/reconstruct_recorded_baseline.py",
             "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py"]
    hashes = {}
    for relative in files:
        source = root / relative
        target = Path(log_dir) / "startup_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
