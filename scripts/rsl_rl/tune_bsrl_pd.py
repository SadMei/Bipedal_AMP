"""Isolated Isaac PD experiments on the saved analytical-parallel BSRL plant.

Bench: fixed base, suspended legs, gravity off, individual +/-0.05 rad steps.
Walk: frozen latent policy, multiple speeds/start perturbations, first-fall metrics.
Training processes, assets, checkpoints and action scaling are never changed.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

from local_source import prefer_local_source_tree
from bsrl_pd_profiles import DEFAULT_PROFILES, PROFILES, gain_vectors


def main():
    _, version = prefer_local_source_tree(__file__)
    original = builtins.__import__

    def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original(name, globals, locals, fromlist, level)
        if name == "isaaclab.utils.version" and not hasattr(module, "get_isaac_sim_version"):
            module.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
        return module

    builtins.__import__ = compatible_import
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("bench", "walk"), required=True)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(DEFAULT_PROFILES))
    parser.add_argument("--seconds", type=float, default=30.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.checkpoint.is_file():
        parser.error("Require a checkpoint and a NEW output directory")
    try:
        launcher = AppLauncher(args, multi_gpu=False)
    finally:
        builtins.__import__ = original
    try:
        run(args, version)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def run(args, version):
    import numpy as np
    import torch
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import quat_from_euler_xyz
    from legged_lab.envs import ManagerBasedAmpEnv
    from record_latent import restore_templates
    from frozen_run_config import restore_run_config, disable_training_command_sampling
    from t1_heading_ablation import load_saved_config

    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    saved_env = load_saved_config(checkpoint.parent / "params/env.yaml")
    saved_agent = load_saved_config(checkpoint.parent / "params/agent.yaml")
    cfg, agent, name, width, action_dim = restore_templates(saved_env, saved_agent, version)
    if name != "BSRL":
        raise ValueError("Only analytical-parallel BSRL latent checkpoints are supported")
    audit = restore_run_config(cfg, agent, checkpoint.parent)
    joint_names = saved_env["actions"]["joint_pos"]["joint_names"]
    conditions = []
    for profile in args.profiles:
        if args.mode == "walk":
            for speed in (0.4, 0.6, 0.8, 1.0):
                for case, roll, delay in (("nominal", 0., 0), ("roll_plus", .01, 0),
                                          ("roll_minus", -.01, 0), ("delay20ms", 0., 4)):
                    conditions.append(dict(profile=profile, speed=speed, case=case, roll=roll, delay=delay))
        else:
            for joint in joint_names:
                for sign in (-1, 1):
                    conditions.append(dict(profile=profile, joint=joint, sign=sign, delay=0))
    cfg.scene.num_envs = len(conditions)
    cfg.scene.env_spacing = 40.0 if args.mode == "walk" else 3.0
    cfg.sim.device = args.device
    cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.scene.robot.spawn.usd_dir = tempfile.mkdtemp(prefix="bsrl_pd_tune_usd_")
    cfg.log_dir = str(output)
    cfg.observations.policy.enable_corruption = False
    for key in vars(cfg.events):
        setattr(cfg.events, key, None)
    cfg.events.reset_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = actuator.max_delay = 0
    for key in vars(cfg.curriculum):
        setattr(cfg.curriculum, key, None)
    cfg.terminations.time_out = None
    cfg.terminal_obs_groups = ()
    cfg.episode_length_s = args.seconds + 5
    cmd = cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (0., 0.)
    cmd.ranges.lin_vel_y = (0., 0.)
    cmd.ranges.ang_vel_z = (0., 0.)
    cmd.ranges.heading = None
    cmd.heading_command = False
    cmd.rel_heading_envs = cmd.rel_standing_envs = 0.
    cmd.resampling_time_range = (1e9, 1e9)
    cmd.debug_vis = False
    disable_training_command_sampling(cmd)
    if args.mode == "bench":
        cfg.scene.robot.spawn.fix_base = True
        cfg.scene.robot.init_state.pos = (0., 0., 1.5)
        cfg.sim.gravity = (0., 0., 0.)
        for key in vars(cfg.terminations):
            setattr(cfg.terminations, key, None)
    dump_yaml(str(output / "effective_env.yaml"), cfg)
    for filename in ("tune_bsrl_pd.py", "bsrl_pd_profiles.py", "record_latent.py"):
        shutil.copy2(Path(__file__).parent / filename, output / filename)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_restoration=audit, mode=args.mode, conditions=conditions, profiles=PROFILES,
        physics_dt=cfg.sim.dt, policy_dt=cfg.sim.dt*cfg.decimation,
        topology="analytical_parallel_no_explicit_loops", action_scale=saved_env["actions"]["joint_pos"]["scale"],
        no_yaw_compensation=True, joint_names=joint_names,
        failures="Walking metrics exclude the first failed step and everything after it; reset counts retained.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    env = ManagerBasedAmpEnv(cfg=cfg)
    try:
        robot = env.scene["robot"]
        obs, _ = env.reset(seed=42)
        assert robot.is_fixed_base == (args.mode == "bench")
        for actuator in robot.actuators.values():
            assert not actuator.is_implicit_model
            kp, kd = zip(*(gain_vectors(c["profile"], actuator.joint_names) for c in conditions))
            actuator.stiffness[:] = torch.tensor(kp, device=env.device)
            actuator.damping[:] = torch.tensor(kd, device=env.device)
            delay = torch.tensor([c["delay"] for c in conditions], device=env.device)
            # Existing delay buffers retain capacity from the zero-delay cfg;
            # enlarge only this evaluation instance for the 20 ms stress case.
            from isaaclab.utils import DelayBuffer
            for attr in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
                buffer = DelayBuffer(4, len(conditions), device=env.device)
                buffer.set_time_lag(delay)
                setattr(actuator, attr, buffer)
            manifest.setdefault("actual_actuator_gains", {})[type(actuator).__name__+str(actuator.joint_names)] = {
                "joint_names": actuator.joint_names, "kp": actuator.stiffness.cpu().tolist(),
                "kd": actuator.damping.cpu().tolist()}
        ids = env.action_manager.get_term("joint_pos")._joint_ids
        if [robot.joint_names[j] for j in ids] != joint_names:
            raise RuntimeError("Action joint ordering mismatch")
        scale = env.action_manager.get_term("joint_pos")._scale
        torch.testing.assert_close(scale, torch.full_like(scale, .25))
        n = len(conditions)
        device = env.device
        # Apply documented perturbations before collecting the first policy observation.
        if args.mode == "walk":
            state = robot.data.root_state_w.clone()
            roll = torch.tensor([c["roll"] for c in conditions], device=device)
            state[:, 3:7] = quat_from_euler_xyz(roll, roll*0, roll*0)
            robot.write_root_state_to_sim(state)
            env.sim.forward()
        command = env.command_manager.get_command("base_velocity")
        speeds = torch.tensor([c.get("speed", 0.) for c in conditions], device=device)
        command[:, 0] = speeds
        env.observation_manager.reset()
        obs = env.observation_manager.compute(update_history=True)
        policy = torch.jit.load(str(args.export_dir / "policy.pt"), map_location=device).eval()
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        onnx = ort.InferenceSession(str(args.export_dir / "policy.onnx"), opts, providers=["CPUExecutionProvider"])
        max_policy_error = 0.
        first_fall = torch.full((n,), float("nan"), device=device)
        resets = torch.zeros(n, dtype=torch.int, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        torque_sum = torch.zeros(n, device=device)
        torque_peak = torch.zeros(n, device=device)
        sat_sum = torch.zeros(n, device=device)
        motor_count = torch.zeros(n, device=device)
        ankle_peak = torch.zeros(n, device=device)
        ankle_power_peak = torch.zeros(n, device=device)
        update = env.scene.update

        def audited_update(dt):
            update(dt)
            if dt == 0:
                return
            motors, requested, powers = [], [], []
            for actuator in robot.actuators.values():
                if hasattr(actuator, "transmission"):
                    tr = actuator.transmission
                    tau, req = tr.motor_torque.flatten(1), tr.requested_motor_torque.flatten(1)
                    ankle_peak[:] = torch.maximum(ankle_peak, tau.abs().amax(1)*alive)
                    ankle_power_peak[:] = torch.maximum(ankle_power_peak,
                        (tr.motor_torque*tr.motor_velocity).abs().flatten(1).amax(1)*alive)
                else:
                    tau, req = actuator.applied_effort, actuator.computed_effort
                motors.append(tau)
                requested.append(req)
            tau, req = torch.cat(motors, 1), torch.cat(requested, 1)
            torque_sum[:] += tau.square().sum(1)*alive
            torque_peak[:] = torch.maximum(torque_peak, tau.abs().amax(1)*alive)
            sat_sum[:] += ((req-tau).abs() > .01).sum(1)*alive
            motor_count[:] += tau.shape[1]*alive

        env.scene.update = audited_update
        keys = ("time", "q", "target", "dq", "root_pos", "root_quat", "body_vel", "yaw", "valid", "action", "foot_force")
        data = {key: [] for key in keys}
        sensor = env.scene["contact_forces"]
        foot_ids, _ = sensor.find_bodies(["link_left_ankle_roll", "link_right_ankle_roll"], preserve_order=True)
        default_q = robot.data.default_joint_pos[:, ids].clone()
        steps = round(args.seconds / env.step_dt)
        for step in range(steps):
            t = step*env.step_dt
            with torch.inference_mode():
                if args.mode == "walk":
                    actions = policy(obs["policy"])
                    if step % 250 == 0:
                        raw = obs["policy"][:1].cpu().numpy()
                        actual = onnx.run(None, {onnx.get_inputs()[0].name: raw})[0]
                        expected = actions[:1].cpu().numpy()
                        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-4)
                        max_policy_error = max(max_policy_error, float(abs(actual-expected).max()))
                else:
                    actions = torch.zeros(n, action_dim, device=device)
                    # Two second pulse, then return to rest, for each axis independently.
                    if 1. <= t < 3.:
                        for i, condition in enumerate(conditions):
                            actions[i, joint_names.index(condition["joint"])] = condition["sign"]*.05/.25
                obs, _, terminated, truncated, _ = env.step(actions)
            failed = terminated | truncated
            first_fall[failed & alive] = (step+1)*env.step_dt
            resets += failed.int()
            alive &= ~failed
            command[:, 0] = speeds
            command[:, 1:] = 0
            # Reset command terms resample zeros; overwrite before any next policy inference.
            if bool(failed.any()):
                obs = env.observation_manager.compute(update_history=False)
            values = ((step+1)*env.step_dt, robot.data.joint_pos[:, ids],
                env.action_manager.get_term("joint_pos").processed_actions,
                robot.data.joint_vel[:, ids], robot.data.root_pos_w-env.scene.env_origins,
                robot.data.root_quat_w, robot.data.root_lin_vel_b, robot.data.heading_w, alive,
                actions, sensor.data.net_forces_w[:, foot_ids, :])
            for key, value in zip(keys, values):
                data[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
            if (step+1) % 250 == 0:
                print(f"[PD_TUNE] mode={args.mode} {step+1}/{steps} survivors={alive.sum().item()}/{n}", flush=True)
        arrays = {key: np.asarray(value) for key, value in data.items()}
        np.savez_compressed(output / "trajectory.npz", **arrays)
        summaries = []
        for i, condition in enumerate(conditions):
            valid = arrays["valid"][:, i]
            index = np.where(valid)[0]
            row = dict(condition, first_fall_s=None if torch.isnan(first_fall[i]) else float(first_fall[i]),
                resets=int(resets[i]), valid_seconds=float(valid.sum()*env.step_dt),
                motor_torque_rms_nm=float((torque_sum[i]/motor_count[i].clamp_min(1)).sqrt()),
                motor_peak_nm=float(torque_peak[i]), saturation_fraction=float(sat_sum[i]/motor_count[i].clamp_min(1)),
                ankle_motor_peak_nm=float(ankle_peak[i]), ankle_motor_power_peak_w=float(ankle_power_peak[i]))
            if index.size:
                q = arrays["q"][valid, i]
                target = arrays["target"][valid, i]
                row["joint_tracking_rmse_rad"] = float(np.sqrt(np.mean((q-target)**2)))
                row["joint_tracking_rmse_per_axis"] = np.sqrt(np.mean((q-target)**2, axis=0)).tolist()
                row["yaw_drift_deg"] = float(np.degrees(np.unwrap(arrays["yaw"][valid, i])[-1]-arrays["yaw"][0, i]))
                vel = arrays["body_vel"][valid, i]
                row["mean_vx"] = float(vel[:, 0].mean())
                row["vx_rmse"] = float(np.sqrt(np.mean((vel[:, 0]-condition.get("speed", 0))**2)))
                row["vy_rms"] = float(np.sqrt(np.mean(vel[:, 1]**2)))
                quat = arrays["root_quat"][valid, i]
                tilt = np.arccos(np.clip(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1, 1))
                row["tilt_rms_deg"] = float(np.degrees(np.sqrt(np.mean(tilt**2))))
                row["tilt_peak_deg"] = float(np.degrees(tilt.max()))
                stationary = valid & (arrays["time"] >= 3.)
                contact = np.linalg.norm(arrays["foot_force"][stationary, i], axis=-1)>5.
                row["contact_fraction_lr"] = contact.mean(0).tolist() if len(contact) else []
                if args.mode == "bench":
                    j = joint_names.index(condition["joint"])
                    steady = (arrays["time"] >= 2.5) & (arrays["time"] <= 3.)
                    phase = (arrays["time"] > 1.) & (arrays["time"] <= 3.)
                    response = (arrays["q"][phase, i, j]-float(default_q[i, j]))*condition["sign"]
                    row["overshoot_percent"] = float(max(0., response.max()/.05-1.)*100)
                    row["steady_error_rad"] = float(np.mean(abs(arrays["q"][steady, i, j]-arrays["target"][steady, i, j])))
                    row["steady_velocity_rms"] = float(np.sqrt(np.mean(arrays["dq"][steady, i, j]**2)))
                    outside = np.where(abs(response-.05)>.0025)[0]
                    row["settling_5pct_s"] = (float((outside[-1]+1)*env.step_dt) if len(outside) and outside[-1] < len(response)-1
                        else (None if len(outside) else 0.))
            summaries.append(row)
        manifest.update(live_onnx_max_abs_error=max_policy_error, completed=True,
            actual_total_mass_kg=float(robot.root_physx_view.get_masses()[0].sum()),
            actual_seconds=steps*env.step_dt, summary_file="summary.json")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False)+"\n")
        print("[COMPLETE] "+str(output), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
