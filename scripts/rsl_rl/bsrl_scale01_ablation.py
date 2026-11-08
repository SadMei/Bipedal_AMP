"""Independent BSRL ablations from the formal scale=0.1 latent baseline."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import _restore_value, restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config
from bsrl_training_matrix import command_motion_patterns


VARIANTS = ("command_ref", "scale025", "pd2x", "custom_pd", "command_ref_deploy_pd")

CUSTOM_PD = {
    "large_motors": {
        "stiffness": {"joint_.*_hip_pitch": 600.0, "joint_.*_hip_roll": 600.0,
                      "joint_.*_knee_pitch": 600.0},
        "damping": {"joint_.*_hip_pitch": 2.0, "joint_.*_hip_roll": 2.0,
                    "joint_.*_knee_pitch": 2.0},
    },
    "small_motors": {
        "stiffness": {"joint_.*_hip_yaw": 600.0},
        "damping": {"joint_.*_hip_yaw": 2.0},
    },
    "parallel_ankles": {
        "stiffness": {"joint_.*_ankle_pitch": 350.0, "joint_.*_ankle_roll": 80.0},
        "damping": {"joint_.*_ankle_pitch": 0.0, "joint_.*_ankle_roll": 0.0},
    },
}

COMMAND_REF_DEPLOY_PD = {
    "large_motors": {
        "stiffness": {"joint_.*_hip_pitch": 300.0, "joint_.*_hip_roll": 200.0,
                      "joint_.*_knee_pitch": 300.0},
        "damping": {"joint_.*_hip_pitch": 2.0, "joint_.*_hip_roll": 2.0,
                    "joint_.*_knee_pitch": 2.0},
    },
    "small_motors": {
        "stiffness": {"joint_.*_hip_yaw": 200.0},
        "damping": {"joint_.*_hip_yaw": 2.0},
    },
    "parallel_ankles": {
        "stiffness": {"joint_.*_ankle_pitch": 120.0, "joint_.*_ankle_roll": 40.0},
        "damping": {"joint_.*_ankle_pitch": 0.0, "joint_.*_ankle_roll": 0.0},
    },
}


def _scale_values(mapping, factor):
    if isinstance(mapping, dict):
        return {key: value * factor for key, value in mapping.items()}
    return mapping * factor


def _set_vx_range(env):
    """Use exactly [-1, 1] for both command sampling and curriculum phases."""
    env["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = (-1.0, 1.0)
    curriculum = env["curriculum"].get("velocity_range_and_tracking_std")
    if curriculum is not None:
        curriculum["params"]["lin_vel_x_ranges"] = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))


def validate_baseline(env, agent):
    if (agent["experiment_name"] != "bsrl_amp"
            or agent["policy"]["class_name"] != "legged_lab.latent.models:BSRLLatentActorCritic"
            or agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"):
        raise ValueError("Expected the formal estimator-free BSRL latent baseline")
    if agent["algorithm"]["state_est_loss_required"] or agent["algorithm"]["symmetry_cfg"] is not None:
        raise ValueError("Baseline must have neither estimator loss nor mirror")
    if env["observations"]["policy"]["history_length"] != 10:
        raise ValueError("Expected the ten-frame latent Actor")
    if env["observations"]["policy"].get("heading_error") is not None:
        raise ValueError("Actor heading must remain disabled")
    robot = env["scene"]["robot"]
    if (set(robot["actuators"]) != {"large_motors", "small_motors", "parallel_ankles"}
            or not robot["class_type"].endswith(":ReducedAnkleArticulation")):
        raise ValueError("Expected analytical parallel ankles")
    if len(env["motion_data"]["motion_dataset"]["motion_data_weights"]) != 34:
        raise ValueError("Expected the full ACCAD34 dataset")
    scales = set(env["actions"]["joint_pos"]["scale"].values())
    if scales != {0.1}:
        raise ValueError(f"Expected scale01 baseline, got action scales {scales}")


def expected_configuration(saved_env, saved_agent, *, variant, run_name, num_envs,
                           max_iterations, device):
    validate_baseline(saved_env, saved_agent)
    if variant not in VARIANTS or not 1 <= max_iterations <= 40000:
        raise ValueError("Unknown variant or iterations outside 1..40000")
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    _set_vx_range(env)

    if variant == "command_ref":
        animation = env["animation"]["animation"]
        animation["command_name"] = "base_velocity"
        animation["command_motion_patterns"] = command_motion_patterns(
            env["motion_data"]["motion_dataset"]["motion_data_weights"])
        animation["command_component_scales"] = (1.2, 0.8, 0.5)
        animation["match_command_speed"] = False
    elif variant == "scale025":
        env["actions"]["joint_pos"]["scale"] = {
            pattern: 0.25 for pattern in env["actions"]["joint_pos"]["scale"]
        }
    elif variant == "pd2x":
        motors = env["scene"]["robot"]["actuators"]
        for motor in motors.values():
            motor["stiffness"] = _scale_values(motor["stiffness"], 2.0)
            motor["damping"] = _scale_values(motor["damping"], 2.0)
    elif variant == "custom_pd":
        motors = env["scene"]["robot"]["actuators"]
        for name, values in CUSTOM_PD.items():
            motors[name]["stiffness"] = copy.deepcopy(values["stiffness"])
            motors[name]["damping"] = copy.deepcopy(values["damping"])
    elif variant == "command_ref_deploy_pd":
        if env["animation"]["animation"].get("command_name") != "base_velocity":
            raise ValueError("command_ref_deploy_pd requires the selected command-ref baseline")
        motors = env["scene"]["robot"]["actuators"]
        for name, values in COMMAND_REF_DEPLOY_PD.items():
            motors[name]["stiffness"] = copy.deepcopy(values["stiffness"])
            motors[name]["damping"] = copy.deepcopy(values["damping"])

    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device, resume=False)
    return resolved_scene_config(env), agent


def apply_ablation(cfg, agent, baseline, variant, *, run_name, num_envs,
                   max_iterations, device):
    baseline = Path(baseline).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    audit = restore_run_config(cfg, agent, baseline)
    expected_env, expected_agent = expected_configuration(
        saved_env, saved_agent, variant=variant, run_name=run_name,
        num_envs=num_envs, max_iterations=max_iterations, device=device)
    if variant in ("custom_pd", "command_ref_deploy_pd"):
        # The saved baseline serializes the uniform ankle gains as scalars,
        # while this comparison needs distinct pitch/roll gains. Restore into
        # mapping containers accepted by ReducedAnkleActuatorCfg.
        ankle = cfg.scene.robot.actuators["parallel_ankles"]
        ankle.stiffness = {}
        ankle.damping = {}
    _restore_value(cfg, expected_env, "env")
    _restore_value(agent, expected_agent, "agent")
    differences = configuration_differences(expected_env, resolved_scene_config(cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Incomplete ablation restoration: {differences}")
    dataset = saved_env["motion_data"]["motion_dataset"]
    manifest_path = Path(dataset["motion_data_dir"]) / "manifest.json"
    audit.update(
        variant=variant,
        fresh_training=True,
        max_iterations=max_iterations,
        mirror=False,
        estimator=False,
        latent_preserved=True,
        action_scale=0.25 if variant == "scale025" else 0.1,
        pd_multiplier=2.0 if variant == "pd2x" else None,
        custom_pd=(copy.deepcopy(CUSTOM_PD) if variant == "custom_pd" else
                   copy.deepcopy(COMMAND_REF_DEPLOY_PD)
                   if variant == "command_ref_deploy_pd" else None),
        straight_heading_reward=True,
        critic_heading_observations=True,
        command_conditioned_reference=variant in ("command_ref", "command_ref_deploy_pd"),
        vx_command_range=[-1.0, 1.0],
        dataset_manifest=str(manifest_path),
        baseline_scale01=str(baseline),
        requested_configuration_differences=configuration_differences(saved_env, expected_env, "env")
        + configuration_differences(saved_agent, expected_agent, "agent"),
        unrequested_configuration_differences=differences,
    )
    return audit


def runtime_audit(env, agent, variant):
    import torch
    from legged_lab.assets.bsrl_reduced_ankle import ReducedAnkleArticulation

    dims = env.observation_manager.group_obs_dim
    critic_width = sum(dims[name][0] for name in agent.obs_groups["critic"])
    if tuple(dims["policy"]) != (480,) or critic_width != 370:
        raise RuntimeError(f"Unexpected latent observation contract: {dims}, critic={critic_width}")
    if agent.algorithm.state_est_loss_required or agent.algorithm.symmetry_cfg is not None:
        raise RuntimeError("Estimator or mirror unexpectedly enabled")
    if not isinstance(env.scene["robot"], ReducedAnkleArticulation):
        raise RuntimeError("Wrong BSRL robot plant")
    scale = env.action_manager.get_term("joint_pos")._scale
    expected_scale = 0.25 if variant == "scale025" else 0.1
    torch.testing.assert_close(scale, torch.full_like(scale, expected_scale))
    expected_multiplier = 2.0 if variant == "pd2x" else 1.0
    expected_gains = None
    if variant == "custom_pd":
        expected_gains = CUSTOM_PD
    elif variant == "command_ref_deploy_pd":
        expected_gains = COMMAND_REF_DEPLOY_PD
    gains = {}
    for key, motor in env.scene["robot"].actuators.items():
        gains[key] = dict(names=motor.joint_names,
                          kp_min=motor.stiffness.min(0).values.tolist(),
                          kp_max=motor.stiffness.max(0).values.tolist(),
                          kd_min=motor.damping.min(0).values.tolist(),
                          kd_max=motor.damping.max(0).values.tolist())
        for j, name in enumerate(motor.joint_names):
            ankle = "ankle" in name
            if expected_gains is not None:
                def gain_value(field):
                    return next(value for pattern, value in expected_gains[key][field].items()
                                if __import__("re").fullmatch(pattern, name))
                kp, kd = gain_value("stiffness"), gain_value("damping")
            else:
                nominal_kp = 100 if ankle else (200 if "knee" in name else 150)
                nominal_kd = 1 if ankle else 2
                kp, kd = nominal_kp * expected_multiplier, nominal_kd * expected_multiplier
            if not (bool(torch.all((motor.stiffness[:, j] >= .8 * kp - 1e-4)
                               & (motor.stiffness[:, j] <= 1.2 * kp + 1e-4)))
                    and bool(torch.all((motor.damping[:, j] >= .8 * kd - 1e-4)
                                       & (motor.damping[:, j] <= 1.2 * kd + 1e-4)))):
                raise RuntimeError(f"Unexpected randomized gains: {name}")
    animation = env.animation_manager.get_term("animation")
    conditioned = animation._command_category_motion_ids is not None
    if conditioned != (variant in ("command_ref", "command_ref_deploy_pd")):
        raise RuntimeError("Wrong reference sampling mode")
    return dict(variant=variant, policy_width=480, critic_width=critic_width,
                history_frames=10, latent_dim=20, estimator=False, mirror=False,
                action_scale=float(scale.flatten()[0]), pd_multiplier=expected_multiplier,
                custom_pd=(copy.deepcopy(CUSTOM_PD) if variant == "custom_pd" else
                           copy.deepcopy(COMMAND_REF_DEPLOY_PD)
                           if variant == "command_ref_deploy_pd" else None),
                straight_heading_reward="heading_frame_lin_vel_xy_exp" in env.reward_manager.active_terms,
                critic_heading_observations=True,
                vx_command_range=[-1.0, 1.0], physics_dt=env.physics_dt,
                policy_dt=env.step_dt, runtime_gain_ranges=gains,
                command_conditioned_reference=conditioned,
                category_counts={key: len(value) for key, value in
                                 (animation._command_category_motion_ids or {}).items()},
                heading_compensation=False)


def snapshot_sources(log_dir):
    root = Path(__file__).resolve().parents[2]
    hashes = {}
    paths = ["scripts/rsl_rl/bsrl_scale01_ablation.py",
             "scripts/rsl_rl/train_bsrl_scale01_ablation.py",
             "scripts/rsl_rl/launch_bsrl_scale01_ablation.py",
             "source/legged_lab/legged_lab/managers/animation_manager.py",
             "source/legged_lab/legged_lab/managers/motion_command_categories.py"]
    for relative in paths:
        source = root / relative
        target = Path(log_dir) / "ablation_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
