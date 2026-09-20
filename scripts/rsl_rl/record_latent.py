"""Frozen T1/BSRL latent checkpoint: nominal ONNX replay in Isaac, two views.

Does not touch training/export directories. No heading compensation or automatic
resets. Writes both videos, the raw trajectory, and exact evaluation provenance.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

from local_source import prefer_local_source_tree, prepare_agent_cfg_for_local_rsl_rl
from bsrl_pd_profiles import PROFILES, apply_bsrl_pd_profile


def main():
    _, version = prefer_local_source_tree(__file__)
    original_import = builtins.__import__

    def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "isaaclab.utils.version":
            version_module = sys.modules.get(name)
            if version_module is not None and not hasattr(version_module, "get_isaac_sim_version"):
                version_module.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
        return module

    builtins.__import__ = compatible_import
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--command", type=float, nargs=3, default=(0.6, 0.0, 0.0))
    parser.add_argument("--label", default="", help="Short run label shown in the video")
    parser.add_argument(
        "--pd-profile",
        choices=tuple(PROFILES),
        default="nominal",
        help="BSRL actuator gains for simulator-only evaluation; action scale stays unchanged",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.checkpoint.is_file() or args.output_dir.exists():
        parser.error("Require valid checkpoint, positive seconds, and a NEW output directory")
    args.enable_cameras = True
    try:
        launcher = AppLauncher(args, multi_gpu=False)
    finally:
        builtins.__import__ = original_import
    try:
        record(args, version, launcher.app)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def restore_templates(saved_env, saved_agent, version):
    """Provide typed objects for strict saved-run restoration, including new fields."""
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg

    policy_class = saved_agent["policy"]["class_name"]
    if policy_class == "legged_lab.latent.models:LatentActorCritic":
        from legged_lab.tasks.locomotion.amp.config.t1.t1_amp_env_cfg import T1AmpEnvCfg
        from legged_lab.tasks.locomotion.amp.config.t1.agents.rsl_rl_ppo_cfg import T1RslRlOnPolicyRunnerAmpCfg
        cfg, agent = T1AmpEnvCfg(), T1RslRlOnPolicyRunnerAmpCfg()
        robot_name, width, action_dim = "T1", 870, 25
    elif policy_class == "legged_lab.latent.models:BSRLLatentActorCritic":
        from legged_lab.tasks.locomotion.amp.config.bsrl.bsrl_amp_env_cfg import BSRLAmpEnvCfg
        from legged_lab.tasks.locomotion.amp.config.bsrl.agents.rsl_rl_ppo_cfg import BSRLRslRlOnPolicyRunnerAmpCfg
        from legged_lab.assets.bsrl import BSRL_CFG
        from legged_lab.assets.bsrl_reduced_ankle import with_reduced_ankles
        from legged_lab.tasks.locomotion.amp.mdp.heading_frame import (
            HeadingReferenceVelocityCommand, heading_frame_error, heading_frame_lin_vel_xy, heading_frame_lin_vel_xy_exp)
        cfg, agent = BSRLAmpEnvCfg(), BSRLRslRlOnPolicyRunnerAmpCfg()
        cfg.scene.robot = with_reduced_ankles(BSRL_CFG, physics_dt=saved_env["sim"]["dt"])
        cfg.commands.base_velocity.class_type = HeadingReferenceVelocityCommand
        cfg.rewards.heading_frame_lin_vel_xy_exp = RewardTermCfg(
            func=heading_frame_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.4})
        cfg.observations.critic.heading_frame_error = ObservationTermCfg(
            func=heading_frame_error, params={"command_name": "base_velocity"})
        cfg.observations.critic.heading_frame_lin_vel_xy = ObservationTermCfg(
            func=heading_frame_lin_vel_xy, params={"command_name": "base_velocity"})
        robot_name, width, action_dim = "BSRL", 480, 12
    else:
        raise ValueError(f"Unsupported checkpoint policy: {policy_class}")
    agent = prepare_agent_cfg_for_local_rsl_rl(agent, version)
    # Current base policy config has no latent fields; supply inert templates.
    from legged_lab.latent.config import POLICY_OPTIONS
    for key, value in POLICY_OPTIONS.items():
        setattr(agent.policy, key, value.copy() if isinstance(value, list) else value)
    return cfg, agent, robot_name, width, action_dim


def record(args, version, app):
    import imageio.v2 as imageio
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from tensordict import TensorDict
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.utils import string_to_callable
    from legged_lab.envs import ManagerBasedAmpEnv
    from legged_lab.latent.export import export_checkpoint
    from frozen_run_config import restore_run_config, disable_training_command_sampling
    from t1_heading_ablation import load_saved_config, configuration_differences, resolved_scene_config

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve(strict=True)
    run = checkpoint.parent
    saved_env = load_saved_config(run / "params/env.yaml")
    saved_agent = load_saved_config(run / "params/agent.yaml")
    cfg, agent_cfg, name, width, action_dim = restore_templates(saved_env, saved_agent, version)
    restore_audit = restore_run_config(cfg, agent_cfg, run)
    if (saved_agent["algorithm"]["symmetry_cfg"] is not None
            or saved_agent["algorithm"]["state_est_loss_required"]
            or saved_env["observations"]["policy"].get("heading_error")):
        raise ValueError("Expected no-mirror, no-estimator, no-actor-heading latent run")
    apply_bsrl_pd_profile(cfg, name, args.pd_profile)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = saved_agent["seed"]
    cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.scene.robot.spawn.usd_dir = tempfile.mkdtemp(prefix="latent_video_usd_")
    cfg.sim.render_interval = cfg.decimation
    cfg.log_dir = str(output)
    cfg.viewer.resolution = (1280, 720)
    cfg.viewer.eye = (-2.0, 3.0, 1.6)
    cfg.viewer.lookat = (0.25, 0.0, 0.6)
    cfg.observations.policy.enable_corruption = False
    for event_name in vars(cfg.events):
        setattr(cfg.events, event_name, None)
    cfg.events.reset_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = actuator.max_delay = 0
    for group in (cfg.curriculum, cfg.terminations):
        for key in vars(group):
            setattr(group, key, None)
    cfg.terminal_obs_groups = ()
    cfg.episode_length_s = args.seconds + 1.0
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (args.command[0], args.command[0])
    cmd.ranges.lin_vel_y = (args.command[1], args.command[1])
    cmd.ranges.ang_vel_z = (args.command[2], args.command[2])
    cmd.ranges.heading = None
    cmd.heading_command = False
    cmd.rel_heading_envs = cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1e9, 1e9)
    cmd.debug_vis = False
    disable_training_command_sampling(cmd)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during load")
    for value in loaded["model_state_dict"].values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint")
    contract = export_checkpoint(checkpoint, output / "export")
    dump_yaml(str(output / "effective_env.yaml"), cfg)
    dump_yaml(str(output / "saved_agent.yaml"), agent_cfg)
    shutil.copy2(__file__, output / "record_latent.py")
    shutil.copy2(Path(__file__).with_name("bsrl_pd_profiles.py"), output / "bsrl_pd_profiles.py")
    env = ManagerBasedAmpEnv(cfg=cfg, render_mode="rgb_array")
    writers = {}
    try:
        obs, _ = env.reset(seed=cfg.seed)
        obs = TensorDict(obs, batch_size=[1])
        kwargs = dict(saved_agent["policy"])
        policy_class = string_to_callable(kwargs.pop("class_name"))
        policy = policy_class(obs.to("cpu"), saved_agent["obs_groups"], action_dim, **kwargs).cpu()
        policy.load_state_dict(loaded["model_state_dict"], strict=True)
        policy.eval()
        if obs["policy"].shape != (1, width) or env.action_manager.total_action_dim != action_dim:
            raise RuntimeError("Unexpected actual observation/action dimensions")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        engine = ort.InferenceSession(str(output / "export/policy.onnx"), options, providers=["CPUExecutionProvider"])
        input_name = engine.get_inputs()[0].name
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        actual_controller = {"actuators": {}, "action_joint_names": action_term._joint_names,
                             "action_scale": action_term._scale[0].cpu().tolist()}

        def scalar_for_joint(value, joint):
            if not isinstance(value, dict):
                return float(value)
            matches = [gain for pattern, gain in value.items() if re.fullmatch(pattern, joint)]
            if len(matches) != 1:
                raise RuntimeError(f"Ambiguous saved actuator/action setting for {joint}: {value}")
            return float(matches[0])

        for actuator_name, actuator in robot.actuators.items():
            actual_controller["actuators"][actuator_name] = {
                "joint_names": actuator.joint_names, "kp": actuator.stiffness[0].cpu().tolist(),
                "kd": actuator.damping[0].cpu().tolist(), "implicit": actuator.is_implicit_model}
            if args.pd_profile == "nominal":
                saved_actuator = saved_env["scene"]["robot"]["actuators"][actuator_name]
                for field, actual in (("stiffness", actuator.stiffness), ("damping", actuator.damping)):
                    expected = [scalar_for_joint(saved_actuator[field], joint) for joint in actuator.joint_names]
                    np.testing.assert_allclose(actual[0].cpu().numpy(), expected, atol=1e-5, rtol=1e-6)
        expected_scale = [scalar_for_joint(saved_env["actions"]["joint_pos"]["scale"], joint)
                          for joint in action_term._joint_names]
        np.testing.assert_allclose(actual_controller["action_scale"], expected_scale, atol=1e-7, rtol=1e-6)
        print(f"[CONTROLLER] Verified saved PD/scale: {actual_controller}", flush=True)
        cmd_tensor = torch.tensor(args.command, device=env.device).reshape(1, 3)
        torch.testing.assert_close(env.command_manager.get_command("base_velocity"), cmd_tensor)
        if name == "BSRL":
            from legged_lab.assets.bsrl_reduced_ankle import ReducedAnkleArticulation
            if not isinstance(robot, ReducedAnkleArticulation):
                raise RuntimeError("BSRL recording must retain analytical parallel ankles")
        dt, fps = env.step_dt, round(1.0 / env.step_dt)
        steps = round(args.seconds / dt)
        trajectory = {key: [] for key in ("time", "obs", "action", "command", "root_pos", "root_quat",
                     "root_lin_vel_b", "joint_pos", "joint_vel", "applied_torque", "fall")}
        initial_yaw = float(robot.data.heading_w[0])
        initial_xy = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
        prev_yaw, unwrapped_yaw, live_error, first_fall = initial_yaw, initial_yaw, 0.0, None
        sensor = env.scene["contact_forces"]
        base_names = saved_env["terminations"]["base_contact"]["params"]["sensor_cfg"]["body_names"]
        base_ids, _ = sensor.find_bodies(base_names)
        min_height = saved_env["terminations"]["base_height"]["params"]["minimum_height"]
        limit_angle = saved_env["terminations"]["bad_orientation"]["params"]["limit_angle"]
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        env.render()
        for _ in range(20):
            env.sim.render()
        command_tag = (f"forward{round(args.command[0] * 10):02d}"
                       if args.command[1:] == [0.0, 0.0] and np.isclose(args.command[0] * 10, round(args.command[0] * 10))
                       else "custom_command")
        for view in ("whole", "feet"):
            writers[view] = imageio.get_writer(str(output / f"{name.lower()}_model{loaded['iter']}_{command_tag}_{view}.mp4"),
                fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error",
                ffmpeg_params=["-threads", "2"])
        for step in range(steps):
            if not app.is_running():
                raise RuntimeError("Isaac stopped before requested video duration")
            raw = obs["policy"].cpu().numpy()
            actions = engine.run(None, {input_name: raw})[0]
            if step % 50 == 0:
                with torch.inference_mode():
                    expected = policy.actor(torch.from_numpy(raw)).numpy()
                np.testing.assert_allclose(actions, expected, atol=3e-5, rtol=1e-4)
                live_error = max(live_error, float(np.abs(actions - expected).max()))
            if not np.isfinite(actions).all():
                raise RuntimeError("Nonfinite ONNX action")
            with torch.inference_mode():
                new_obs, _, terminated, truncated, _ = env.step(torch.from_numpy(actions).to(env.device))
            if bool((terminated | truncated).any()):
                raise RuntimeError("Unexpected automatic reset during continuous recording")
            obs = TensorDict(new_obs, batch_size=[1])
            torch.testing.assert_close(env.command_manager.get_command("base_velocity"), cmd_tensor)
            pos = robot.data.root_pos_w[0].cpu().numpy().copy()
            vel = robot.data.root_lin_vel_b[0].cpu().numpy().copy()
            quat = robot.data.root_quat_w[0].cpu().numpy().copy()
            yaw = float(robot.data.heading_w[0])
            unwrapped_yaw += np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
            prev_yaw = yaw
            q = robot.data.joint_pos[0, action_term._joint_ids].cpu().numpy().copy()
            dq = robot.data.joint_vel[0, action_term._joint_ids].cpu().numpy().copy()
            torque = robot.data.applied_torque[0, action_term._joint_ids].cpu().numpy().copy()
            if not all(np.isfinite(v).all() for v in (pos, vel, q, dq, torque)):
                raise RuntimeError("Nonfinite simulator state")
            base_force = float(sensor.data.net_forces_w_history[:, :, base_ids].norm(dim=-1).max())
            fall = bool(pos[2] < min_height or float(robot.data.projected_gravity_b[0, 2]) > -np.cos(limit_angle) or base_force > 1.0)
            t = (step + 1) * dt
            if fall and first_fall is None:
                first_fall = t
            values = (t, raw[0].copy(), actions[0].copy(), np.array(args.command), pos, quat, vel, q, dq, torque, fall)
            for key, value in zip(trajectory, values):
                trajectory[key].append(value)
            # World-fixed camera azimuth: it follows translation, never hides
            # yaw drift by rotating with the robot. Close-up is a separate file.
            for view in ("whole", "feet"):
                if view == "whole":
                    eye, target = [pos[0]-2.0, pos[1]+3.0, 1.65], [pos[0]+0.2, pos[1], 0.65]
                else:
                    eye, target = [pos[0]-0.8, pos[1]+1.25, 0.48], [pos[0]+0.1, pos[1], 0.19]
                env.sim.set_camera_view(eye=eye, target=target)
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((8, 8, 1265, 122), fill=(18, 22, 28))
                draw.text((18, 13), f"{name} {args.label or 'latent20'} | model {loaded['iter']} | {view} | ONNX -> Isaac", font=font, fill="white")
                draw.text((18, 40), f"cmd=({args.command[0]:.2f}, {args.command[1]:.2f}, {args.command[2]:.2f}) | no mirror / no yaw compensation | t={t:5.2f}s", font=font, fill="white")
                draw.text((18, 67), f"body vx={vel[0]:+.3f} m/s | yaw drift={np.degrees(unwrapped_yaw-initial_yaw):+.1f} deg | first fall={first_fall}", font=font, fill="white")
                scale_values = actual_controller["action_scale"]
                pd_tag = "saved-run" if args.pd_profile == "nominal" else args.pd_profile
                draw.text((18, 93), f"PD={pd_tag} | scale={min(scale_values):.2f}..{max(scale_values):.2f} | physics={cfg.sim.dt*1000:g}ms / policy={dt*1000:g}ms | no reset", font=font, fill="white")
                writers[view].append_data(np.asarray(frame))
                if step in (0, 249, 749, steps - 1):
                    frame.save(output / f"preview_{view}_{step:04d}.png")
            if (step + 1) % 250 == 0:
                print(f"[RECORD] {name} {step+1}/{steps} t={t:.1f}s vx={vel[0]:.3f} yaw={np.degrees(unwrapped_yaw-initial_yaw):.1f} fall={first_fall}", flush=True)
        for writer in writers.values():
            writer.close()
        writers.clear()
        arrays = {key: np.asarray(values) for key, values in trajectory.items()}
        np.savez_compressed(output / "trajectory.npz", **arrays)
        dump_yaml(str(output / "evaluation_overrides.yaml"),
                  configuration_differences(saved_env, resolved_scene_config(cfg.to_dict())))
        manifest = {"robot": name, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "iteration": int(loaded["iter"]), "seconds": steps*dt, "frames_per_view": steps, "fps": fps,
            "command": list(args.command), "pd_profile": args.pd_profile, "label": args.label,
            "actual_controller": actual_controller,
            "physics_dt": cfg.sim.dt, "policy_dt": dt,
            "source_restoration": restore_audit, "nominal_eval": True, "automatic_resets": 0,
            "actor_heading": False, "yaw_compensation": False, "mirror": False,
            "topology": "analytical_parallel_ankles_no_explicit_loops" if name == "BSRL" else "T1_training_plant",
            "first_fall_time": first_fall, "final_yaw_drift_degrees": float(np.degrees(unwrapped_yaw-initial_yaw)),
            "displacement_xy": (pos[:2] - initial_xy).tolist(), "mean_body_vx": float(arrays["root_lin_vel_b"][:, 0].mean()),
            "live_onnx_max_abs_error": live_error, "export": contract,
            "videos": [str(p) for p in sorted(output.glob("*.mp4"))],
            "evaluation_overrides": str(output / "evaluation_overrides.yaml")}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("[COMPLETE] " + json.dumps({k:v for k,v in manifest.items() if k not in ("evaluation_overrides", "export")}), flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
