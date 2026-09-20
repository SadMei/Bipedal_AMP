"""Export a saved serial BSRL checkpoint and record actual ONNX inference in Isaac.

This is an isolated nominal evaluation: no training changes, no randomization,
no automatic episode resets, and no overwrite of an existing policy export.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from local_source import prefer_local_source_tree


def main():
    repo_root, _ = prefer_local_source_tree(__file__)
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
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--command", type=float, nargs=3, default=(0.6, 0.0, 0.0))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.checkpoint.is_file():
        parser.error("A valid checkpoint and positive duration are required.")
    if args.output_dir.exists() or args.video_path.exists():
        parser.error("Output directory and video must be new; refusing to overwrite.")
    args.enable_cameras = True
    try:
        launcher = AppLauncher(args, multi_gpu=False)
    finally:
        builtins.__import__ = original_import
    try:
        record(args, repo_root, launcher.app)
    except BaseException:
        # Kit may exit the interpreter during close(), before Python otherwise
        # prints an exception. Preserve the real failure in the launch log.
        import traceback

        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def record(args, repo_root, app):
    import imageio.v2 as imageio
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageFont
    from tensordict import TensorDict

    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.modules import ActorCritic

    from legged_lab.assets.bsrl import BSRL_CFG
    from legged_lab.envs import ManagerBasedAmpEnv
    from legged_lab.rsl_rl import export_policy_as_onnx
    from legged_lab.tasks.locomotion.amp.config.bsrl.bsrl_amp_env_cfg import BSRLAmpEnvCfg

    class ConfigLoader(yaml.SafeLoader):
        pass

    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
    )
    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:builtins.slice",
        lambda loader, node: slice(*loader.construct_sequence(node)),
    )

    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parent
    with (run / "params/env.yaml").open() as stream:
        saved_env = yaml.load(stream, Loader=ConfigLoader)
    with (run / "params/agent.yaml").open() as stream:
        saved_agent = yaml.load(stream, Loader=ConfigLoader)
    if set(saved_env["scene"]["robot"]["actuators"]) != {"large_motors", "small_motors"}:
        raise ValueError("This recording entry point is restricted to legacy direct-PD serial BSRL.")
    policy_kwargs = dict(saved_agent["policy"])
    if policy_kwargs.pop("class_name") != "ActorCritic" or policy_kwargs.get("state_dependent_std"):
        raise ValueError("Expected the ordinary deterministic BSRL ActorCritic contract.")

    cfg = BSRLAmpEnvCfg()
    # Restore the complete saved asset (including gains, envelopes and armature),
    # not the current default reduced-transmission factory.
    cfg.scene.robot = BSRL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.from_dict(saved_env["scene"]["robot"])
    cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.scene.robot.spawn.usd_dir = tempfile.mkdtemp(prefix="bsrl_onnx_record_usd_")
    cfg.sim.from_dict(saved_env["sim"])
    cfg.sim.device = args.device
    cfg.decimation = saved_env["decimation"]
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = saved_env["scene"]["env_spacing"]
    cfg.seed = saved_env["seed"]
    cfg.viewer.resolution = (1280, 720)
    cfg.viewer.eye = (-2.0, 3.0, 1.6)
    cfg.viewer.lookat = (0.25, 0.0, 0.6)
    cfg.log_dir = str(args.output_dir)
    cfg.observations.from_dict(saved_env["observations"])
    cfg.observations.policy.enable_corruption = False
    saved_action = dict(saved_env["actions"]["joint_pos"])
    cfg.actions.joint_pos.clip = saved_action.pop("clip")
    cfg.actions.joint_pos.from_dict(saved_action)
    for name in vars(cfg.events):
        setattr(cfg.events, name, None)
    cfg.events.reset_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = 0
        actuator.max_delay = 0
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    # Record a continuous trajectory; falls are reported below, never hidden by resets.
    for name in vars(cfg.terminations):
        setattr(cfg.terminations, name, None)
    cfg.terminal_obs_groups = ()
    cfg.episode_length_s = args.seconds + 1.0
    command_cfg = cfg.commands.base_velocity
    command_cfg.ranges.lin_vel_x = (args.command[0], args.command[0])
    command_cfg.ranges.lin_vel_y = (args.command[1], args.command[1])
    command_cfg.ranges.ang_vel_z = (args.command[2], args.command[2])
    command_cfg.ranges.heading = None
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1e9, 1e9)
    command_cfg.debug_vis = False
    args.output_dir.mkdir(parents=True)
    args.video_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(args.output_dir / "effective_env.yaml"), cfg)
    print("[CONFIG] Saved serial asset, PD, observation and action contracts restored.", flush=True)

    env = ManagerBasedAmpEnv(cfg=cfg, render_mode="rgb_array")
    writer = None
    try:
        obs, _ = env.reset(seed=cfg.seed)
        obs = TensorDict(obs, batch_size=[1])
        policy = ActorCritic(obs, saved_agent["obs_groups"], env.action_manager.total_action_dim, **policy_kwargs).cpu()
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(loaded["model_state_dict"], strict=True)
        policy.eval()
        if obs["policy"].shape != (1, 240) or env.action_manager.total_action_dim != 12:
            raise RuntimeError("Unexpected observation/action ordering or shape.")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        export_policy_as_onnx(policy, str(args.output_dir), normalizer=policy.actor_obs_normalizer)
        onnx_path = args.output_dir / "policy.onnx"
        model = onnx.load(onnx_path)
        onnx.helper.set_model_props(model, {
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "iteration": str(loaded["iter"]), "run": run.name,
            "topology": "legacy_direct_pd_serial", "policy_input": "240D term-major 5-step history",
        })
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        generator = np.random.default_rng(42)
        parity_max = 0.0
        for sample in generator.normal(size=(64, 1, 240)).astype(np.float32):
            output = session.run(None, {input_name: sample})[0]
            with torch.no_grad():
                expected = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(sample))).numpy()
            np.testing.assert_allclose(output, expected, rtol=1e-4, atol=2e-5)
            parity_max = max(parity_max, float(np.abs(output - expected).max()))
        print(f"[EXPORT] iteration={loaded['iter']} shape=240->12 synthetic_parity_max={parity_max:.3g}", flush=True)

        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        dt = env.step_dt
        steps = round(args.seconds / dt)
        fps = round(1.0 / dt)
        trajectory = {name: [] for name in ("time", "obs", "action", "command", "root_pos", "root_quat", "root_lin_vel_b", "joint_pos", "joint_vel", "applied_torque", "fall")}
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 20) if Path(font_path).is_file() else ImageFont.load_default()
        # Initialize the render product, then warm it without advancing physics.
        env.render()
        for _ in range(20):
            env.sim.render()
        writer = imageio.get_writer(str(args.video_path), fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error")
        initial_yaw = float(robot.data.heading_w[0].item())
        live_parity_max = 0.0
        first_fall_time = None
        base_ids, _ = robot.find_bodies("base_link")
        for step in range(steps):
            if not app.is_running():
                raise RuntimeError("Isaac closed before the recording completed.")
            policy_obs = obs["policy"].detach().cpu().numpy()
            action = session.run(None, {input_name: policy_obs})[0]
            if step % 50 == 0:
                with torch.no_grad():
                    reference = policy.actor(policy.actor_obs_normalizer(torch.from_numpy(policy_obs))).numpy()
                np.testing.assert_allclose(action, reference, rtol=1e-4, atol=2e-5)
                live_parity_max = max(live_parity_max, float(np.abs(action - reference).max()))
            if not np.isfinite(action).all():
                raise RuntimeError("ONNX produced a nonfinite action.")
            with torch.inference_mode():
                new_obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
            if bool((terminated | truncated).any()):
                raise RuntimeError("An unexpected automatic reset interrupted the continuous recording.")
            obs = TensorDict(new_obs, batch_size=[1])
            position = robot.data.root_pos_w[0].cpu().numpy().copy()
            velocity = robot.data.root_lin_vel_b[0].cpu().numpy().copy()
            yaw = float(robot.data.heading_w[0].item())
            joint_pos = robot.data.joint_pos[0, action_term._joint_ids].cpu().numpy().copy()
            joint_vel = robot.data.joint_vel[0, action_term._joint_ids].cpu().numpy().copy()
            if not all(np.isfinite(value).all() for value in (position, velocity, joint_pos, joint_vel)):
                raise RuntimeError("Simulation produced a nonfinite state.")
            base_force = env.scene["contact_forces"].data.net_forces_w_history[:, :, base_ids].norm(dim=-1).max()
            fell = bool(position[2] < 0.5 or robot.data.projected_gravity_b[0, 2] > -0.5 or base_force > 1.0)
            if fell and first_fall_time is None:
                first_fall_time = (step + 1) * dt
            values = ((step + 1) * dt, policy_obs[0].copy(), action[0].copy(),
                      env.command_manager.get_command("base_velocity")[0].cpu().numpy().copy(),
                      position, robot.data.root_quat_w[0].cpu().numpy().copy(), velocity,
                      joint_pos, joint_vel, robot.data.applied_torque[0, action_term._joint_ids].cpu().numpy().copy(), fell)
            for name, value in zip(trajectory, values):
                trajectory[name].append(value)
            env.sim.set_camera_view(eye=[position[0]-2.0, position[1]+3.0, 1.6], target=[position[0]+0.25, position[1], 0.6])
            frame = Image.fromarray(env.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((12, 12, 1120, 105), fill=(18, 22, 28))
            draw.text((24, 20), f"BSRL Sep14 serial | checkpoint {loaded['iter']} | ONNX Runtime -> Isaac Sim", font=font, fill="white")
            draw.text((24, 49), f"cmd vx={args.command[0]:.2f} vy={args.command[1]:.2f} wz={args.command[2]:.2f} | nominal | PD 200Hz / policy {fps}Hz", font=font, fill="white")
            draw.text((24, 78), f"t={(step+1)*dt:5.2f}s | vx={velocity[0]:+.3f} m/s | yaw={np.degrees(yaw-initial_yaw):+.1f} deg | no auto-reset", font=font, fill="white")
            writer.append_data(np.asarray(frame))
            if (step + 1) % 250 == 0:
                print(f"[RECORD] {step+1}/{steps} frames t={(step+1)*dt:.1f}s vx={velocity[0]:.3f} z={position[2]:.3f} fall={first_fall_time}", flush=True)
        writer.close()
        writer = None
        arrays = {name: np.asarray(values) for name, values in trajectory.items()}
        np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
        quats = arrays["root_quat"]
        w, x, y, z = quats.T
        yaw_series = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        manifest = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "iteration": int(loaded["iter"]), "onnx": str(onnx_path.resolve()),
            "video": str(args.video_path.resolve()), "inference_backend": session.get_providers(),
            "topology": "legacy_direct_pd_serial", "seconds": steps*dt, "frames": steps,
            "fps": fps, "physics_dt": cfg.sim.dt, "policy_dt": dt, "command": args.command,
            "nominal_eval": True, "automatic_resets": 0, "first_fall_time": first_fall_time,
            "mean_body_vx": float(arrays["root_lin_vel_b"][:, 0].mean()),
            "min_base_height": float(arrays["root_pos"][:, 2].min()),
            "final_yaw_degrees": float(np.degrees(yaw_series[-1]-initial_yaw)),
            "synthetic_onnx_max_abs_error": parity_max, "rollout_onnx_max_abs_error": live_parity_max,
            "action_clip": saved_env["actions"]["joint_pos"]["clip"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print("[COMPLETE] " + json.dumps(manifest), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
