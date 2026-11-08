"""T1 AMP ablation using Motion-QJ deployment PD gains and action scale 0.1.

The numeric gains are frozen from wbt_est's MEN_T1_QJ_PD_CFG.  This module
contains its own copy deliberately: Bipedal_AMP training must not import assets
or code from another repository at runtime.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import shutil

from frozen_run_config import _restore_value, restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config


ACTION_SCALE = 0.1
WBT_QJPD_SOURCE = "wbt_est/whole_body_tracking.robots.men_t1:MEN_T1_QJ_PD_CFG"
WBT_QJPD_SOURCE_SHA256 = "18fdb71ae77112e3503a9131c8b7c84eee01746c876908e3c2dafd08e5d853f3"
QJ_PD = {
    "legs": {
        "stiffness": {
            ".*_hip_pitch_joint": 300.0, ".*_hip_roll_joint": 400.0,
            ".*_hip_yaw_joint": 200.0, ".*_knee_joint": 300.0,
        },
        "damping": {
            ".*_hip_pitch_joint": 12.0, ".*_hip_roll_joint": 15.0,
            ".*_hip_yaw_joint": 10.0, ".*_knee_joint": 12.0,
        },
    },
    "feet": {
        "stiffness": {".*_ankle_roll_joint": 150.0, ".*_ankle_pitch_joint": 300.0},
        "damping": {".*_ankle_roll_joint": 6.0, ".*_ankle_pitch_joint": 12.0},
    },
    "waist": {
        "stiffness": {"waist_yaw_joint": 250.0, "waist_pitch_joint": 300.0},
        "damping": {"waist_yaw_joint": 12.0, "waist_pitch_joint": 12.0},
    },
    "torso": {
        "stiffness": {
            "torso_pitch_joint": 300.0, "torso_roll_joint": 300.0,
            "torso_yaw_joint": 250.0,
        },
        "damping": {
            "torso_pitch_joint": 12.0, "torso_roll_joint": 12.0,
            "torso_yaw_joint": 12.0,
        },
    },
    "arms": {
        "stiffness": {
            ".*_shoulder_pitch_joint": 80.0, ".*_shoulder_roll_joint": 80.0,
            ".*_shoulder_yaw_joint": 60.0, ".*_elbow_joint": 60.0,
        },
        "damping": {
            ".*_shoulder_pitch_joint": 8.0, ".*_shoulder_roll_joint": 8.0,
            ".*_shoulder_yaw_joint": 6.0, ".*_elbow_joint": 6.0,
        },
    },
}


def _validate_baseline(env, agent):
    if (agent["experiment_name"] != "men_t1_amp"
            or agent["policy"]["class_name"] != "legged_lab.latent.models:LatentActorCritic"
            or agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"):
        raise ValueError("Expected the latest T1 latent AMP baseline")
    if (agent["algorithm"]["symmetry_cfg"] is not None
            or agent["algorithm"]["state_est_loss_required"]
            or env["observations"]["policy"].get("heading_error") is not None):
        raise ValueError("Require no mirror, estimator, or Actor heading")
    reset = env["events"].get("reset_from_ref")
    if (not reset or not reset["func"].endswith(":reset_from_ref_with_default_mixture")
            or reset["params"].get("default_start_probability") != 0.2):
        raise ValueError("Expected the completed default-start20 baseline")
    scales = env["actions"]["joint_pos"]["scale"]
    values = scales.values() if isinstance(scales, dict) else (scales,)
    if any(float(value) != 0.25 for value in values):
        raise ValueError("Expected the latest scale-0.25 baseline")


def expected_configuration(saved_env, saved_agent, *, run_name, num_envs, max_iterations, device):
    _validate_baseline(saved_env, saved_agent)
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    for group, gains in QJ_PD.items():
        actuator = env["scene"]["robot"]["actuators"][group]
        actuator["stiffness"] = copy.deepcopy(gains["stiffness"])
        actuator["damping"] = copy.deepcopy(gains["damping"])
    old_scale = env["actions"]["joint_pos"]["scale"]
    env["actions"]["joint_pos"]["scale"] = (
        {key: ACTION_SCALE for key in old_scale} if isinstance(old_scale, dict) else ACTION_SCALE)
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device,
                 resume=False, load_run=".*", load_checkpoint="model_.*.pt")
    return resolved_scene_config(env), agent


def apply_ablation(cfg, agent, baseline, *, run_name, num_envs, max_iterations, device):
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
        raise RuntimeError(f"Unexpected T1 QJPD-scale01 differences: {differences}")
    requested = configuration_differences(saved_env, expected_env, "env")
    requested += configuration_differences(saved_agent, expected_agent, "agent")
    audit.update(
        fresh_training=True, qjpd_source=WBT_QJPD_SOURCE,
        qjpd_source_sha256=WBT_QJPD_SOURCE_SHA256, qjpd=QJ_PD,
        action_scale=ACTION_SCALE, mirror=False, estimator=False, actor_heading=False,
        default_start_probability=0.2,
        network_rewards_commands_resets_randomization_plant_preserved=True,
        requested_configuration_differences=requested,
        unrequested_configuration_differences=differences,
    )
    return audit


def _joint_value(mapping, joint):
    matches = [float(value) for pattern, value in mapping.items() if re.fullmatch(pattern, joint)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one QJPD mapping for {joint}, got {matches}")
    return matches[0]


def runtime_audit(env, agent):
    import torch

    dims = env.observation_manager.group_obs_dim
    critic_width = sum(dims[key][0] for key in agent.obs_groups["critic"])
    if tuple(dims["policy"]) != (870,) or critic_width != 595:
        raise RuntimeError(f"Unexpected T1 observation contract: {dims}, critic={critic_width}")
    if env.cfg.events.reset_from_ref.func.__name__ != "reset_from_ref_with_default_mixture":
        raise RuntimeError("Latest mixed reset was not preserved")
    count, total = env._amp_default_start_last_count, env._amp_default_start_last_total
    fraction = count / total
    if not 0.15 <= fraction <= 0.25:
        raise RuntimeError(f"Initial default-start fraction outside statistical guard: {fraction}")
    action = env.action_manager.get_term("joint_pos")
    if not torch.allclose(action._scale, torch.full_like(action._scale, ACTION_SCALE)):
        raise RuntimeError("Runtime action scale is not uniformly 0.1")
    robot = env.scene["robot"]
    nominal, randomized = {}, {}
    for group, gains in QJ_PD.items():
        actuator = robot.actuators[group]
        expected_kp = torch.tensor([_joint_value(gains["stiffness"], joint)
                                    for joint in actuator.joint_names], device=env.device)
        expected_kd = torch.tensor([_joint_value(gains["damping"], joint)
                                    for joint in actuator.joint_names], device=env.device)
        cfg_actuator = env.cfg.scene.robot.actuators[group]
        if cfg_actuator.stiffness != gains["stiffness"] or cfg_actuator.damping != gains["damping"]:
            raise RuntimeError(f"Nominal QJPD config mismatch in {group}")
        kp_ratio = actuator.stiffness / expected_kp
        kd_ratio = actuator.damping / expected_kd
        if (float(kp_ratio.min()) < 0.7999 or float(kp_ratio.max()) > 1.2001
                or float(kd_ratio.min()) < 0.7999 or float(kd_ratio.max()) > 1.2001):
            raise RuntimeError(f"Runtime gain randomization outside saved 0.8..1.2 range in {group}")
        nominal[group] = {"stiffness": gains["stiffness"], "damping": gains["damping"]}
        randomized[group] = {
            "kp_ratio_min": float(kp_ratio.min()), "kp_ratio_max": float(kp_ratio.max()),
            "kd_ratio_min": float(kd_ratio.min()), "kd_ratio_max": float(kd_ratio.max()),
        }
    return {
        "policy_width": 870, "critic_width": critic_width, "history_frames": 10,
        "latent_dim": 20, "action_scale_min": float(action._scale.min()),
        "action_scale_max": float(action._scale.max()), "nominal_qjpd": nominal,
        "runtime_gain_randomization": randomized,
        "default_start_probability": 0.2, "observed_initial_default_starts": count,
        "observed_initial_total": total, "observed_initial_fraction": fraction,
        "mirror": False, "estimator": False, "actor_heading": False,
        "physics_dt": env.physics_dt, "policy_dt": env.step_dt,
    }


def snapshot_sources(log_dir):
    root = Path(__file__).resolve().parents[2]
    files = [
        "scripts/rsl_rl/t1_qjpd_scale01_ablation.py",
        "scripts/rsl_rl/train_t1_qjpd_scale01.py",
        "scripts/rsl_rl/launch_t1_qjpd_scale01.py",
    ]
    hashes = {}
    for relative in files:
        source = root / relative
        target = Path(log_dir) / "qjpd_scale01_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
