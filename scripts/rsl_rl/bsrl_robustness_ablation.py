"""Cumulative sim-to-real robustness ablations from the deploy-PD BSRL AMP run."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config
from bsrl_scale01_ablation import COMMAND_REF_DEPLOY_PD, validate_baseline


VARIANTS = ("r1_reset_push_task", "r2_ankle_transmission", "r3_observation")
BASELINE_RUN = (
    "logs/rsl_rl/bsrl_amp/"
    "2026-09-23_23-06-52_accad34_scale01_command_ref_deploy_pd_vxneg1pos1_40000it_8192_"
    "20260923_commandref_deploypd200_300_ankle120_40"
)

DEFAULT_START_PROBABILITY = 0.2
PUSH_INTERVAL_S = (2.5, 5.0)
PUSH_VELOCITY_RANGE = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.2, 0.2)}
TASK_STYLE_LERP = 0.6
TRANSMISSION_RANGES = {
    "ratio_range": (0.97, 1.03),
    "strength_range": (0.9, 1.0),
    "response_time_range": (0.0, 0.005),
}
# Delay all policy-side proprioceptive sensor channels by a per-episode 0/1
# policy frame. Add conservative constant biases only where units are clear.
OBSERVATION_PERTURBATIONS = {
    "base_ang_vel": {"bias_range": (-0.03, 0.03), "max_delay_steps": 1},
    "root_local_rot_tan_norm": {"bias_range": (0.0, 0.0), "max_delay_steps": 1},
    "joint_pos": {"bias_range": (-0.0075, 0.0075), "max_delay_steps": 1},
    "joint_vel": {"bias_range": (0.0, 0.0), "max_delay_steps": 1},
}


def _validate_deploy_pd_baseline(env: dict, agent: dict) -> None:
    validate_baseline(env, agent)
    if env["animation"]["animation"].get("command_name") != "base_velocity":
        raise ValueError("Expected the command-conditioned reference baseline")
    if env["events"].get("ankle_transmission") is not None:
        raise ValueError("Baseline already has ankle transmission randomization")
    if agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"] != 0.4:
        raise ValueError("Expected the 0.4 deploy-PD reward blend")
    motors = env["scene"]["robot"]["actuators"]
    for name, expected in COMMAND_REF_DEPLOY_PD.items():
        if motors[name]["stiffness"] != expected["stiffness"] or motors[name]["damping"] != expected["damping"]:
            raise ValueError(f"Unexpected deploy-PD gains for {name}")
    for term in OBSERVATION_PERTURBATIONS:
        if env["observations"]["policy"][term].get("modifiers") is not None:
            raise ValueError(f"Baseline already modifies policy observation {term}")


def expected_configuration(saved_env, saved_agent, *, variant, run_name, num_envs,
                           max_iterations, device):
    _validate_deploy_pd_baseline(saved_env, saved_agent)
    if variant not in VARIANTS or not 1 <= max_iterations <= 40000 or num_envs < 7:
        raise ValueError("Invalid variant, environment count, or iteration count")
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)

    reset = env["events"]["reset_from_ref"]
    reset["func"] = (
        "legged_lab.tasks.locomotion.amp.mdp.events:reset_from_ref_with_default_mixture"
    )
    reset["params"]["default_start_probability"] = DEFAULT_START_PROBABILITY
    push = env["events"]["push_robot"]
    push["interval_range_s"] = PUSH_INTERVAL_S
    push["params"]["velocity_range"] = copy.deepcopy(PUSH_VELOCITY_RANGE)
    agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"] = TASK_STYLE_LERP

    if variant in VARIANTS[1:]:
        env["events"]["ankle_transmission"] = {
            "func": "legged_lab.assets.bsrl_reduced_ankle:randomize_ankle_transmission",
            "params": copy.deepcopy(TRANSMISSION_RANGES),
            "mode": "startup",
            "interval_range_s": None,
            "is_global_time": False,
            "min_step_count_between_reset": 0,
        }
    if variant == VARIANTS[2]:
        for term, perturbation in OBSERVATION_PERTURBATIONS.items():
            env["observations"]["policy"][term]["modifiers"] = [{
                "func": (
                    "legged_lab.tasks.locomotion.amp.mdp.observation_modifiers:EpisodeBiasDelay"
                ),
                "params": {},
                **copy.deepcopy(perturbation),
            }]

    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device, resume=False)
    return resolved_scene_config(env), agent


def apply_ablation(cfg, agent, baseline, variant, *, run_name, num_envs,
                   max_iterations, device):
    from isaaclab.managers import EventTermCfg
    from legged_lab.assets.bsrl_reduced_ankle import randomize_ankle_transmission
    from legged_lab.tasks.locomotion.amp.mdp.events import reset_from_ref_with_default_mixture
    from legged_lab.tasks.locomotion.amp.mdp.observation_modifiers import EpisodeBiasDelayCfg

    baseline = Path(baseline).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    expected_env, expected_agent = expected_configuration(
        saved_env, saved_agent, variant=variant, run_name=run_name, num_envs=num_envs,
        max_iterations=max_iterations, device=device)
    audit = restore_run_config(cfg, agent, baseline)

    cfg.events.reset_from_ref.func = reset_from_ref_with_default_mixture
    cfg.events.reset_from_ref.params["default_start_probability"] = DEFAULT_START_PROBABILITY
    cfg.events.push_robot.interval_range_s = PUSH_INTERVAL_S
    cfg.events.push_robot.params["velocity_range"] = copy.deepcopy(PUSH_VELOCITY_RANGE)
    agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp = TASK_STYLE_LERP
    if variant in VARIANTS[1:]:
        cfg.events.ankle_transmission = EventTermCfg(
            func=randomize_ankle_transmission, mode="startup", params=copy.deepcopy(TRANSMISSION_RANGES)
        )
    if variant == VARIANTS[2]:
        for term, perturbation in OBSERVATION_PERTURBATIONS.items():
            getattr(cfg.observations.policy, term).modifiers = [
                EpisodeBiasDelayCfg(**copy.deepcopy(perturbation))
            ]
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.log_dir = None
    agent.run_name = run_name
    agent.max_iterations = max_iterations
    agent.device = device
    agent.resume = False

    differences = configuration_differences(expected_env, resolved_scene_config(cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Incomplete robustness ablation restoration: {differences}")
    requested = configuration_differences(saved_env, expected_env, "env")
    requested += configuration_differences(saved_agent, expected_agent, "agent")
    audit.update(
        variant=variant, fresh_training=True, resume=False, max_iterations=max_iterations,
        num_envs=num_envs, task_style_lerp=TASK_STYLE_LERP,
        default_start_probability=DEFAULT_START_PROBABILITY,
        push_interval_s=PUSH_INTERVAL_S, push_velocity_range=PUSH_VELOCITY_RANGE,
        ankle_transmission=(copy.deepcopy(TRANSMISSION_RANGES) if variant in VARIANTS[1:] else None),
        policy_observation_perturbations=(
            copy.deepcopy(OBSERVATION_PERTURBATIONS) if variant == VARIANTS[2] else None
        ),
        critic_observation_perturbations=None, discriminator_observation_perturbations=None,
        requested_configuration_differences=requested,
        unrequested_configuration_differences=differences,
    )
    return audit


def runtime_audit(env, agent, variant):
    from legged_lab.tasks.locomotion.amp.mdp.observation_modifiers import EpisodeBiasDelay

    dims = env.observation_manager.group_obs_dim
    critic_width = sum(dims[name][0] for name in agent.obs_groups["critic"])
    if tuple(dims["policy"]) != (480,) or critic_width != 370:
        raise RuntimeError(f"Unexpected observation contract: policy={dims['policy']}, critic={critic_width}")
    if agent.resume or agent.algorithm.amp_cfg.amp_discriminator.task_style_lerp != TASK_STYLE_LERP:
        raise RuntimeError("Training is not fresh or reward blend is incorrect")
    startup = env.event_manager.active_terms.get("startup", [])
    has_transmission = "ankle_transmission" in startup
    if has_transmission != (variant in VARIANTS[1:]):
        raise RuntimeError(f"Wrong transmission event state: {startup}")
    tr = env.scene["robot"].actuators["parallel_ankles"].transmission
    if has_transmission:
        if not (tr.ratio.min() >= 0.97 and tr.ratio.max() <= 1.03
                and tr.strength.min() >= 0.9 and tr.strength.max() <= 1.0
                and tr.response_time.min() >= 0.0 and tr.response_time.max() <= 0.005):
            raise RuntimeError("Runtime ankle transmission draw is outside requested ranges")
    modifiers = {}
    for term, cfg_term in zip(
        env.observation_manager._group_obs_term_names["policy"],
        env.observation_manager._group_obs_term_cfgs["policy"],
    ):
        instances = [m.func for m in (cfg_term.modifiers or []) if isinstance(m.func, EpisodeBiasDelay)]
        if instances:
            modifiers[term] = {
                "bias_range": tuple(instances[0]._cfg.bias_range),
                "max_delay_steps": instances[0]._cfg.max_delay_steps,
            }
    expected_modifiers = OBSERVATION_PERTURBATIONS if variant == VARIANTS[2] else {}
    if modifiers != expected_modifiers:
        raise RuntimeError(f"Wrong policy observation modifiers: {modifiers}")
    reset_cfg = next(cfg for name, cfg in zip(
        env.event_manager.active_terms["reset"], env.event_manager._mode_term_cfgs["reset"]
    ) if name == "reset_from_ref")
    default_count = getattr(env, "_amp_default_start_last_count", None)
    default_total = getattr(env, "_amp_default_start_last_total", None)
    if default_count is None or default_total is None or not 0 <= default_count <= default_total:
        raise RuntimeError("Default/reference reset mixture did not report a valid runtime sample")
    return {
        "variant": variant, "policy_width": 480, "critic_width": critic_width,
        "fresh_training": True, "task_style_lerp": TASK_STYLE_LERP,
        "default_start_probability": reset_cfg.params["default_start_probability"],
        "observed_default_start_count": default_count,
        "observed_default_start_total": default_total,
        "observed_default_start_fraction": default_count / default_total,
        "push_interval_s": PUSH_INTERVAL_S, "push_velocity_range": PUSH_VELOCITY_RANGE,
        "ankle_transmission": has_transmission, "policy_observation_modifiers": modifiers,
        "critic_observation_modifiers": False, "discriminator_observation_modifiers": False,
    }


def snapshot_sources(log_dir):
    root = Path(__file__).resolve().parents[2]
    paths = [
        "scripts/rsl_rl/bsrl_robustness_ablation.py",
        "scripts/rsl_rl/train_bsrl_robustness_ablation.py",
        "scripts/rsl_rl/launch_bsrl_robustness_ablation.py",
        "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/observation_modifiers.py",
        "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/events.py",
        "source/legged_lab/legged_lab/assets/bsrl_reduced_ankle.py",
    ]
    hashes = {}
    for relative in paths:
        source = root / relative
        target = Path(log_dir) / "ablation_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
