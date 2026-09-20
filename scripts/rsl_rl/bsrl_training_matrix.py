"""Three fresh BSRL latent runs from the latest formal ACCAD34 baseline."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil

from frozen_run_config import _restore_value, restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config

VARIANTS = ("scale01", "matched_bigpd", "clean_bigpd")
UNIFORM_COMMAND = "isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand"
HEADING_REWARDS = ("heading_frame_lin_vel_xy_exp", "zero_yaw_initial_heading_lin_vel_xy_exp",
                   "zero_command_heading_hold_exp", "zero_command_yaw_rate_l2")
HEADING_OBSERVATIONS = ("heading_error", "heading_frame_error", "heading_frame_lin_vel_xy")


def command_motion_patterns(names):
    """Use source action labels; net yaw audited for unnamed turnaround clips.

    No cropping, speed relabeling, mirroring or modification of source motions.
    Every existing clip belongs to one category; purity warnings remain valid.
    """
    result = {key: [] for key in ("forward", "backward", "lateral_left", "lateral_right",
                                 "rotate_left", "rotate_right", "standing")}
    for name in names:
        label = name.lower()
        if label.startswith("stand_"):
            category = "standing"
        elif "side_step_left" in label:
            category = "lateral_left"
        elif "side_step_right" in label:
            category = "lateral_right"
        elif "turn_left" in label:
            category = "rotate_left"
        elif "turn_right" in label:
            category = "rotate_right"
        elif "turn_around" in label:
            category = "rotate_left" if label.startswith("male2") else "rotate_right"
        elif "backwards_turn_run_forward" in label:
            category = "rotate_left"  # measured total yaw +202 degrees
        elif "change_direction" in label or "turn_change" in label:
            category = "rotate_right"  # measured total yaw -70 / -98 degrees
        elif "back" in label:
            category = "backward"
        elif label.startswith("straight_") or ("running" in label and "turn" not in label):
            category = "forward"
        else:
            raise ValueError(f"Unclassified reference motion: {name}")
        result[category].append(name)
    if any(not values for values in result.values()):
        raise ValueError("Each command category must have a reference clip")
    return {key: tuple(values) for key, values in result.items()}


def validate_baseline(env, agent):
    if (agent["experiment_name"] != "bsrl_amp"
            or agent["policy"]["class_name"] != "legged_lab.latent.models:BSRLLatentActorCritic"
            or agent["algorithm"]["class_name"] != "legged_lab.latent.algorithm:LatentPPOAMP"):
        raise ValueError("Expected latest formal BSRL latent baseline, not estimator smoke runs")
    if (agent["algorithm"]["state_est_loss_required"] or agent["algorithm"]["symmetry_cfg"] is not None
            or env["observations"].get("state_est_label") is not None):
        raise ValueError("Baseline must be no-mirror and no-supervised-estimator")
    if env["observations"]["policy"]["history_length"] != 10:
        raise ValueError("Ten-frame actor required")
    if any(env["observations"]["policy"].get(key) for key in HEADING_OBSERVATIONS):
        raise ValueError("Actor must not observe accumulated heading")
    robot = env["scene"]["robot"]
    if (set(robot["actuators"]) != {"large_motors", "small_motors", "parallel_ankles"}
            or not robot["class_type"].endswith(":ReducedAnkleArticulation")):
        raise ValueError("Require analytical parallel ankles, not explicit loops")
    if len(env["motion_data"]["motion_dataset"]["motion_data_weights"]) != 34:
        raise ValueError("Full ACCAD34 dataset required")


def expected_configuration(saved_env, saved_agent, *, variant, run_name, num_envs, max_iterations, device):
    validate_baseline(saved_env, saved_agent)
    if variant not in VARIANTS or not 1 <= max_iterations <= 40000:
        raise ValueError("Unknown variant or iterations outside 1..40000")
    env, agent = copy.deepcopy(saved_env), copy.deepcopy(saved_agent)
    env["actions"]["joint_pos"]["scale"] = {
        pattern: .1 if variant == "scale01" else .25 for pattern in env["actions"]["joint_pos"]["scale"]}
    if variant != "scale01":
        motors = env["scene"]["robot"]["actuators"]
        for key in ("large_motors", "small_motors"):
            motors[key]["stiffness"] = {pattern: 2 * gain for pattern, gain in motors[key]["stiffness"].items()}
            motors[key]["damping"] = {pattern: 3 * gain for pattern, gain in motors[key]["damping"].items()}
        # Deliberately do NOT copy stiff2damped3's ankle 200/3. User hardware constraint.
        assert motors["parallel_ankles"]["stiffness"] == 100
        assert motors["parallel_ankles"]["damping"] == 1
    if variant == "matched_bigpd":
        animation = env["animation"]["animation"]
        animation["command_name"] = "base_velocity"
        animation["command_motion_patterns"] = command_motion_patterns(
            env["motion_data"]["motion_dataset"]["motion_data_weights"])
        animation["command_component_scales"] = (1.2, .8, .5)
        animation["match_command_speed"] = False
    if variant == "clean_bigpd":
        for key in HEADING_REWARDS:
            if key in env["rewards"]:
                env["rewards"][key] = None
        for group in ("policy", *agent["obs_groups"]["critic"]):
            for key in HEADING_OBSERVATIONS:
                if key in env["observations"][group]:
                    env["observations"][group][key] = None
        for group, key in (("disc", "velocity_commands"), ("disc_demo", "ref_velocity_commands")):
            if key in env["observations"][group]:
                env["observations"][group][key] = None
        animation = env["animation"]["animation"]
        animation["command_name"] = animation["command_motion_patterns"] = None
        animation["match_command_speed"] = False
        command = env["commands"]["base_velocity"]
        command["class_type"] = UNIFORM_COMMAND
        command["heading_command"] = False
        command["rel_heading_envs"] = 0.
        for key, value in (("straight_command_probability", 0.), ("single_axis_commands", False),
                           ("motion_speed_animation", None)):
            if key in command:
                command[key] = value
    env["scene"]["num_envs"] = num_envs
    env["sim"]["device"] = device
    env["log_dir"] = None
    agent.update(run_name=run_name, max_iterations=max_iterations, device=device, resume=False)
    return resolved_scene_config(env), agent


def apply_matrix(cfg, agent, baseline, variant, *, run_name, num_envs, max_iterations, device):
    from bsrl_accad_ablation import load_dataset, file_hash

    baseline = Path(baseline).resolve(strict=True)
    old_env = load_saved_config(baseline / "params/env.yaml")
    old_agent = load_saved_config(baseline / "params/agent.yaml")
    audit = restore_run_config(cfg, agent, baseline)
    expected_env, expected_agent = expected_configuration(old_env, old_agent, variant=variant,
        run_name=run_name, num_envs=num_envs, max_iterations=max_iterations, device=device)
    _restore_value(cfg, expected_env, "env")
    _restore_value(agent, expected_agent, "agent")
    differences = configuration_differences(expected_env, resolved_scene_config(cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent.to_dict(), "agent")
    if differences:
        raise RuntimeError(f"Incomplete matrix config restoration: {differences}")
    dataset = old_env["motion_data"]["motion_dataset"]
    manifest_path = Path(dataset["motion_data_dir"]) / "manifest.json"
    manifest, weights = load_dataset(manifest_path)
    if weights != dataset["motion_data_weights"]:
        raise ValueError("Saved weights differ from ACCAD manifest")
    audit.update(variant=variant, fresh_training=True, max_iterations=max_iterations,
        mirror=False, estimator=False, latent_preserved=True, ankle_pd=[100, 1],
        command_conditioned_reference=variant == "matched_bigpd", conditional_discriminator=False,
        initial_heading_reward=variant != "clean_bigpd", velocity_curriculum_preserved=True,
        dataset_manifest=str(manifest_path), dataset_manifest_sha256=file_hash(manifest_path),
        dataset_clips=len(weights), dataset_quality_warnings=manifest["warnings"],
        requested_configuration_differences=configuration_differences(old_env, expected_env, "env")
          + configuration_differences(old_agent, expected_agent, "agent"),
        unrequested_configuration_differences=differences)
    return audit


def runtime_audit(env, agent, variant):
    import torch
    from legged_lab.assets.bsrl_reduced_ankle import ReducedAnkleArticulation
    manager = env.observation_manager
    dims = manager.group_obs_dim
    critic_width = sum(dims[name][0] for name in agent.obs_groups["critic"])
    expected_critic = 350 if variant == "clean_bigpd" else 370
    if (tuple(dims["policy"]) != (480,) or critic_width != expected_critic
            or tuple(dims["disc"]) != (4, 39) or tuple(dims["disc_demo"]) != (4, 39)):
        raise RuntimeError(f"Wrong observation contract: {dims}, critic={critic_width}")
    if (agent.algorithm.state_est_loss_required or agent.algorithm.symmetry_cfg is not None
            or "state_est_label" in manager.active_terms):
        raise RuntimeError("Estimator or mirror unexpectedly enabled")
    if not isinstance(env.scene["robot"], ReducedAnkleArticulation):
        raise RuntimeError("Wrong robot plant")
    animation = env.animation_manager.get_term("animation")
    conditioned = animation._command_category_motion_ids is not None
    if conditioned != (variant == "matched_bigpd"):
        raise RuntimeError("Wrong reference sampling mode")
    if ("heading_frame_lin_vel_xy_exp" in env.reward_manager.active_terms) != (variant != "clean_bigpd"):
        raise RuntimeError("Wrong heading reward mode")
    scale = env.action_manager.get_term("joint_pos")._scale
    torch.testing.assert_close(scale, torch.full_like(scale, .1 if variant == "scale01" else .25))
    gains = {}
    for key, motor in env.scene["robot"].actuators.items():
        gains[key] = dict(names=motor.joint_names, kp_min=motor.stiffness.min(0).values.tolist(),
            kp_max=motor.stiffness.max(0).values.tolist(), kd_min=motor.damping.min(0).values.tolist(),
            kd_max=motor.damping.max(0).values.tolist())
        # Existing startup DR remains +/-20%; nominal gains are saved with the policy.
        for j, name in enumerate(motor.joint_names):
            ankle = "ankle" in name
            kp = 100 if ankle else (200 if "knee" in name else 150) * (1 if variant == "scale01" else 2)
            kd = 1 if ankle else (2 if variant == "scale01" else 6)
            if not (bool(torch.all((motor.stiffness[:, j] >= .8*kp-1e-4) & (motor.stiffness[:, j] <= 1.2*kp+1e-4)))
                    and bool(torch.all((motor.damping[:, j] >= .8*kd-1e-4) & (motor.damping[:, j] <= 1.2*kd+1e-4)))):
                raise RuntimeError(f"Unexpected randomized gains: {name}")
    if conditioned:
        # Audit actual seven-way sampler, including zero commands. Save/restore
        # RNG and term state so this check does not alter training trajectories.
        from legged_lab.managers.motion_command_categories import command_category_ids
        command = env.command_manager.get_command("base_velocity")
        saved = command.clone(), animation.motion_ids.clone(), animation.motion_durations.clone()
        devices = [command.device.index] if command.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            try:
                probes = torch.tensor([[.6,0,0],[-.6,0,0],[0,.4,0],[0,-.4,0],
                                       [0,0,.3],[0,0,-.3],[0,0,0]], device=command.device)
                command[:7] = probes
                animation._sample_motions_for_commands(torch.arange(7, device=command.device))
                for i, category in enumerate(animation._command_category_motion_ids):
                    if not bool(torch.isin(animation.motion_ids[i], animation._command_category_motion_ids[category])):
                        raise RuntimeError(f"Reference selector failed {category}")
            finally:
                command[:] = saved[0]
                animation.motion_ids[:] = saved[1]
                animation.motion_durations[:] = saved[2]
        category_ids, _ = command_category_ids(command, animation.cfg.command_component_scales, standing=True)
        for i, category in enumerate(animation._command_category_motion_ids):
            mask = category_ids == i
            if not bool(torch.isin(animation.motion_ids[mask], animation._command_category_motion_ids[category]).all()):
                raise RuntimeError("Actual reset references do not match commands")
    return dict(variant=variant, policy_width=480, critic_width=critic_width, disc_shape=[4,39],
        history_frames=10, latent_dim=20, estimator=False, mirror=False, action_scale=float(scale.flatten()[0]),
        physics_dt=env.physics_dt, policy_dt=env.step_dt, runtime_gain_ranges=gains,
        command_conditioned_reference=conditioned,
        category_counts={key: len(value) for key, value in (animation._command_category_motion_ids or {}).items()},
        seven_way_command_probes_passed=conditioned, heading_compensation=False)


def snapshot_sources(log_dir):
    from bsrl_latent_ablation import snapshot_sources as snapshot_latent
    hashes = snapshot_latent(log_dir)
    root = Path(__file__).resolve().parents[2]
    paths = ["scripts/rsl_rl/bsrl_training_matrix.py", "scripts/rsl_rl/train_bsrl_matrix.py",
        "scripts/rsl_rl/launch_bsrl_matrix.py", "source/legged_lab/test/test_bsrl_training_matrix.py",
        "source/legged_lab/legged_lab/managers/animation_manager.py",
        "source/legged_lab/legged_lab/managers/motion_command_categories.py",
        "source/legged_lab/legged_lab/envs/manager_based_animation_env.py",
        "source/legged_lab/legged_lab/envs/manager_based_amp_env.py",
        "source/legged_lab/legged_lab/assets/bsrl.py",
        "source/legged_lab/legged_lab/assets/delayed_implicit_actuator.py"]
    for relative in paths:
        source = root / relative
        target = Path(log_dir) / "latent_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes
