#!/usr/bin/env python3
"""Record delivered door AMP policies without editing their deployment sources.

Extract the original numerical configuration, PD, observation and reset functions
with AST; do not start its viewer/gamepad or any real robot interface. The simulation
loop preserves the original first-inference-at-5ms / subsequent-20ms schedule.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_command(values, limits, stand_threshold, allow_outside_gamepad_range=False):
    """Validate direct physical commands without silently clipping the experiment."""
    command = np.asarray(values, dtype=np.float32)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("Command must contain three finite physical velocities")
    outside = bool(np.any(np.abs(command) > np.asarray(limits) + 1e-7))
    if outside and not allow_outside_gamepad_range:
        raise ValueError("Command outside delivered gamepad range; explicitly opt in with --allow-outside-gamepad-range")
    if not np.linalg.norm(command) > stand_threshold:
        raise ValueError("This audit expects an unambiguous nonzero walking command")
    return command, outside


def command_tag(command):
    vx10 = float(command[0]) * 10
    return (f"forward{round(vx10):02d}"
            if np.all(command[1:] == 0) and np.isclose(vx10, round(vx10))
            else "custom_command")


def original_namespace(source, model=None):
    import mujoco

    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JOINT_NAMES" for t in n.targets)]
    for node in main.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names and names[0] == "m":
                break
            keep.append(node)
    keep += [n for n in main.body if isinstance(n, ast.FunctionDef)
             and n.name in ("compute_single_obs", "reset_robot")]
    ns = dict(np=np, mujoco=mujoco, deque=deque)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source), "exec"), ns)
    if model is not None:
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ns["JOINT_NAMES"]])
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        if np.any(ids < 0) or base < 0:
            raise ValueError("Missing policy joint")
        if model.nu != len(ids) or not np.array_equal(model.actuator_trnid[:, 0], ids):
            raise ValueError("Actuator order differs from delivered direct ctrl assignment")
        ns.update(m=model, d=mujoco.MjData(model), joint_qadr=model.jnt_qposadr[ids],
                  joint_vadr=model.jnt_dofadr[ids], base_qadr=model.jnt_qposadr[base],
                  base_vadr=model.jnt_dofadr[base], obs_history=deque(maxlen=ns["actor_obs_hist_steps"]))
        ns["reset_robot"]()
    return ns


def contract_from_original(headless, ns):
    mapping = dict(simulation_dt="simulation_dt", control_decimation="control_decimation",
                   history_steps="actor_obs_hist_steps", base_init_height="base_init_height",
                   command_limit="cmd_limit", command_scale="command_scale", joystick_deadzone="joy_deadzone",
                   stand_threshold="stand_com_threshold", dof_velocity_scale="obs_scale_dof_vel",
                   angular_velocity_scale="obs_scale_ang_vel", stiffness="stiffness_array",
                   damping="damping_array", default_joint_pos="default_joint_pos", action_scale="action_scale")
    options = {key: tuple(ns[value].tolist()) if isinstance(ns[value], np.ndarray) else ns[value]
               for key, value in mapping.items()}
    cfg = headless.Contract(**options)
    cfg.validate()
    return cfg


def physics_step(ns):
    import mujoco

    d = ns["d"]
    tau = ns["pd_control"](ns["target_dof_pos"], d.qpos[ns["joint_qadr"]], ns["stiffness_array"],
                           np.zeros(ns["num_actions"]), d.qvel[ns["joint_vadr"]], ns["damping_array"])
    d.ctrl[:] = tau
    before = d.time
    mujoco.mj_step(ns["m"], d)
    if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            and math.isclose(d.time - before, ns["simulation_dt"], abs_tol=1e-9)):
        raise RuntimeError("Nonfinite state or unexpected MuJoCo reset")


def infer(ns, engine, command):
    ns["commands"] = command.copy()
    ns["obs_history"].append(ns["compute_single_obs"]())
    obs = np.concatenate(ns["obs_history"]).reshape(1, -1).astype(np.float32)
    action = engine.policy.run([engine.output_name], {engine.input_name: obs})[0].reshape(-1)
    if not np.isfinite(action).all():
        raise RuntimeError("Nonfinite action")
    ns["action_buffer"] = action.copy()
    ns["target_dof_pos"] = action * ns["action_scale"] + ns["default_joint_pos"]
    return obs[0].copy(), action.copy()


def validate_parity(headless, source, xml, policy, cfg):
    engine = headless.Engine(xml, policy, cfg, threads=1)
    ns = original_namespace(source, engine.model)
    errors = dict(obs=0.0, action=0.0, qpos=0.0)
    for i in range(200):
        cmd = np.array([0, 0, 0] if i < 30 else [0.6, 0, 0] if i < 130 else [0.3, 0.1, 0.2], dtype=np.float32)
        engine.tick(cmd)
        for _ in range(1 if i == 0 else cfg.control_decimation):
            physics_step(ns)
        obs, action = infer(ns, engine, cmd)
        for key, a, b in (("obs", obs, engine.last_input[0]), ("action", action, engine.action),
                          ("qpos", ns["d"].qpos, engine.data.qpos)):
            errors[key] = max(errors[key], float(np.max(np.abs(a - b))))
            np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)
    engine.reset()
    return engine, dict(policy_steps=200, max_abs_errors=errors)


def record(args):
    import imageio.v2 as imageio
    import mujoco
    import onnx
    from PIL import Image, ImageDraw, ImageFont

    folder = args.door / "document" / ("mujoco_sim_0824" if args.variant == "0827" else "mujoco_sim_0916")
    source = folder / "zy_25dof_mujoco_sim.py"
    policy_name = "0827-27-91000.onnx" if args.variant == "0827" else "2026-09-16_14-26-34.onnx"
    # 0916 contains stale 'mujoco_sim/' prefixes; resolve its bundled files only.
    xml, policy = folder / "xml/zy_25dof_0814.xml", folder / "policy" / policy_name
    engine_source = args.door / "document/mujoco_sim/headless_sim2sim.py"
    spec = importlib.util.spec_from_file_location("door_headless_audit", engine_source)
    headless = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = headless
    spec.loader.exec_module(headless)
    cfg = contract_from_original(headless, original_namespace(source))
    command, outside_gamepad_range = checked_command(
        args.command, cfg.command_limit, cfg.stand_threshold, args.allow_outside_gamepad_range)
    if outside_gamepad_range:
        print(f"[COMMAND] direct injection {command.tolist()} exceeds delivered gamepad limit {cfg.command_limit}; NOT clipped", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph = onnx.load(policy)
    graph_info = dict(inputs=[(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in graph.graph.input],
                      outputs=[(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in graph.graph.output],
                      operators=dict(Counter(n.op_type for n in graph.graph.node)),
                      weight_shapes={w.name: list(w.dims) for w in graph.graph.initializer if len(w.dims) == 2})
    for path, name in ((source, "original_sim.py"), (xml, "original_model.xml"),
                       (Path(__file__), "recorder.py"), (engine_source, "headless_engine.py")):
        shutil.copy2(path, output / name)
    hashes = {str(p): digest(p) for p in (source, xml, policy, engine_source, Path(__file__))}
    for p in sorted((folder / "meshes").glob("*")):
        if p.is_file():
            hashes[str(p)] = digest(p)
    engine, parity = validate_parity(headless, source, xml, policy, cfg)
    model = engine.model
    ns = original_namespace(source, model)
    d = ns["d"]
    model.opt.timestep = cfg.simulation_dt
    # Check that rotating world heading alone cannot change the current frame.
    old_quat = d.qpos[ns["base_qadr"]+3:ns["base_qadr"]+7].copy()
    obs0 = ns["compute_single_obs"]()
    d.qpos[ns["base_qadr"]+3:ns["base_qadr"]+7] = [math.cos(.6), 0, 0, math.sin(.6)]
    obs_rotated = ns["compute_single_obs"]()
    np.testing.assert_allclose(obs0, obs_rotated, atol=1e-7, rtol=0)
    d.qpos[ns["base_qadr"]+3:ns["base_qadr"]+7] = old_quat
    mujoco.mj_forward(model, d)
    steps = round(args.seconds / cfg.simulation_dt)
    if steps % cfg.control_decimation:
        raise ValueError("Duration must contain an integral number of video frames")
    fps = round(1 / cfg.control_dt)
    physical = {k: [] for k in ("time", "qpos", "qvel", "ctrl", "actuator_force")}
    inference = {k: [] for k in ("time", "obs", "action", "target", "command")}
    frames = {k: [] for k in ("time", "qpos", "qvel", "body_velocity", "yaw_drift_deg", "height", "tilt_deg")}
    previous_yaw = unwrapped_yaw = 0.0
    first_fall = None
    print(f"[PARITY] {args.variant}: {parity}", flush=True)
    for step in range(steps):
        physics_step(ns)
        ns["counter"] += 1
        if ns["counter"] % cfg.control_decimation == 0:
            obs, action = infer(ns, engine, command)
            for k, value in zip(inference, (d.time, obs, action, ns["target_dof_pos"].copy(), command.copy())):
                inference[k].append(value)
            np.testing.assert_allclose(obs[-84:-81], command * cfg.command_scale, atol=1e-7, rtol=0)
        quat = d.qpos[ns["base_qadr"]+3:ns["base_qadr"]+7]
        yaw = math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
        unwrapped_yaw += math.atan2(math.sin(yaw-previous_yaw), math.cos(yaw-previous_yaw))
        previous_yaw = yaw
        height = float(d.qpos[ns["base_qadr"]+2])
        tilt = math.degrees(math.acos(float(np.clip(-ns["compute_single_obs"]()[-1], -1, 1))))
        if first_fall is None and (height < .65 or tilt > 60.):
            first_fall = float(d.time)
        for k, value in zip(physical, (d.time, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.actuator_force.copy())):
            physical[k].append(value)
        if (step+1) % cfg.control_decimation == 0:
            body_vel = ns["quat_rotate_inverse_wxyz"](quat, d.qvel[ns["base_vadr"]:ns["base_vadr"]+3])
            for k, value in zip(frames, (d.time, d.qpos.copy(), d.qvel.copy(), body_vel, math.degrees(unwrapped_yaw), height, tilt)):
                frames[k].append(value)
        if (step+1) % 1000 == 0:
            print(f"[SIM] {args.variant} t={d.time:.2f}s yaw={math.degrees(unwrapped_yaw):+.2f} fall={first_fall}", flush=True)
    physical, inference, frames = [{k: np.asarray(v) for k, v in group.items()} for group in (physical, inference, frames)]
    for group in (physical, inference, frames):
        if not all(np.isfinite(v).all() for v in group.values()):
            raise RuntimeError("Nonfinite saved trajectory")
    np.savez_compressed(output / "physics.npz", **physical)
    np.savez_compressed(output / "policy.npz", **inference)
    np.savez_compressed(output / "frames.npz", **frames)
    manifest = dict(variant=args.variant, policy=str(policy), xml=str(xml), original_script=str(source),
                    hashes=hashes, contract=asdict(cfg), graph=graph_info, parity=parity,
                    changes_from_delivered="fixed physical command replacing joystick; headless; video and telemetry only; resolve bundled paths",
                    heading_controller=False, automatic_resets=0, command=command.tolist(),
                    delivered_gamepad_limit=list(cfg.command_limit),
                    outside_delivered_gamepad_range=outside_gamepad_range,
                    command_clipped=False,
                    seconds=args.seconds, simulation_seconds=float(d.time), fps=fps, frames_per_view=len(frames["time"]),
                    first_fall_time=first_fall, fall_criteria="root height <0.65m or tilt >60deg; never resets/stops rollout",
                    final_yaw_drift_deg=float(frames["yaw_drift_deg"][-1]),
                    displacement_xy=(d.qpos[ns["base_qadr"]:ns["base_qadr"]+2]).tolist(),
                    mean_body_vx=float(frames["body_velocity"][:, 0].mean()),
                    mean_body_vx_after_5s=(float(frames["body_velocity"][frames["time"]>=5, 0].mean())
                                           if np.any(frames["time"]>=5) else None),
                    mean_abs_body_vy=float(np.abs(frames["body_velocity"][:, 1]).mean()),
                    world_yaw_rotation_observation_max_error=float(np.max(np.abs(obs0-obs_rotated))),
                    versions=dict(mujoco=mujoco.__version__),
                    videos=[])
    import onnxruntime
    manifest["versions"]["onnxruntime"] = onnxruntime.__version__
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[RESULT] {args.variant} vx={manifest['mean_body_vx']:.4f} yaw={manifest['final_yaw_drift_deg']:.2f}", flush=True)
    if args.no_video:
        return
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    try:
        for view in ("whole", "feet"):
            path = output / f"door_{args.variant}_{command_tag(command)}_{view}.mp4"
            camera.azimuth, camera.elevation = 130, -15
            camera.distance = 3.7 if view == "whole" else 1.55
            if view == "feet":
                camera.elevation = -10
            with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_log_level="error", output_params=["-threads", "2", "-movflags", "+faststart"]) as writer:
                for i, t in enumerate(frames["time"]):
                    d.qpos[:] = frames["qpos"][i]
                    d.qvel[:] = frames["qvel"][i]
                    d.time = t
                    mujoco.mj_forward(model, d)
                    camera.lookat[:] = [d.qpos[ns["base_qadr"]], d.qpos[ns["base_qadr"]+1], .85 if view == "whole" else .25]
                    renderer.update_scene(d, camera=camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((8, 8, args.width-8, 98), fill=(18, 22, 28))
                    lines = [f"DOOR {args.variant} | {policy_name} | {view} | original XML / PD / ONNX",
                             f"cmd=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}) | NO heading compensation / NO resets | t={t:.2f}s",
                             f"body vx={frames['body_velocity'][i,0]:+.3f} m/s | yaw={frames['yaw_drift_deg'][i]:+.1f} deg | fall={first_fall if first_fall is not None and first_fall <= t else None}"]
                    for row, line in enumerate(lines):
                        draw.text((18, 12+row*28), line, font=font, fill="white")
                    writer.append_data(np.asarray(frame))
                    if i in (0, 249, 749, len(frames["time"])-1):
                        frame.save(output / f"preview_{view}_{i:04d}.png")
                    if (i+1) % 500 == 0:
                        print(f"[VIDEO] {args.variant} {view} {i+1}/{len(frames['time'])}", flush=True)
            manifest["videos"].append(str(path))
    finally:
        renderer.close()
    for path, old in hashes.items():
        if digest(path) != old:
            raise RuntimeError(f"Source changed during recording: {path}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[COMPLETE] {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--door", type=Path, default=Path("/root/mjw/door"))
    parser.add_argument("--variant", choices=("0827", "0916"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--command", type=float, nargs=3, default=[.6, 0., 0.])
    parser.add_argument("--allow-outside-gamepad-range", action="store_true",
                        help="Explicitly inject commands beyond the original joystick limit; no clipping")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Require a new output directory and positive finite seconds")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    record(args)


if __name__ == "__main__":
    main()
