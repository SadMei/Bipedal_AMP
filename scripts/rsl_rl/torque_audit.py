"""Opt-in, read-only physics-step torque logging for checkpoint evaluation.

The hook copies tensors AFTER the existing actuator writes its effort to PhysX.
It does not change gains, limits, torques, targets, policy parameters or physics.
Nominal conditions and episode duration are configured by play.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_torque_audit(env, policy, policy_nn, checkpoint, output_dir, seconds, simulation_app):
    sim_env = env.unwrapped
    if sim_env.num_envs != 1:
        raise ValueError("Torque audit currently requires --num_envs 1 (one continuous trajectory per command).")
    robot = sim_env.scene["robot"]
    command = sim_env.command_manager.get_term("base_velocity")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit: {output}")
    control_steps = int(round(seconds / sim_env.step_dt))
    capacity = control_steps * sim_env.cfg.decimation
    fields = ("joint_pos", "joint_vel", "joint_pos_target", "computed_torque", "applied_torque",
              "root_pos_w", "root_quat_w", "root_lin_vel_b", "root_ang_vel_b")
    metadata = {
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
        "joint_names": robot.joint_names, "physics_dt": sim_env.physics_dt,
        "control_dt": sim_env.step_dt, "seconds_requested_per_command": seconds,
        "nominal_eval": True, "seed": sim_env.cfg.seed,
        "asset_path": sim_env.cfg.scene.robot.spawn.asset_path,
        "asset_sha256": sha256(sim_env.cfg.scene.robot.spawn.asset_path),
        "normalization": type(getattr(policy_nn, "actor_obs_normalizer", None)).__name__,
        "actuators": {}, "profiles": [],
        "sample_convention": "q, qdot and torque at actuator evaluation BEFORE each physics integration step; torque is drive effort, not contact reaction",
        "conditions": "No pushes, reference resets, observation noise, mass/COM/gain/material/friction randomization or actuator delay. Real fall terminations retained; audit stops on first termination.",
    }
    for name, actuator in robot.actuators.items():
        inds = actuator.joint_indices
        indices = list(range(robot.num_joints))[inds] if isinstance(inds, slice) else [int(i) for i in inds]
        metadata["actuators"][name] = {
            "joint_indices": indices,
            **{key: getattr(actuator, key)[0].detach().cpu().tolist()
               for key in ("stiffness", "damping", "effort_limit", "velocity_limit")},
            "peak_power": getattr(actuator.cfg, "peak_power", None),
            "min_delay": getattr(actuator.cfg, "min_delay", None),
            "max_delay": getattr(actuator.cfg, "max_delay", None),
        }
    # Validate metadata before spending time on rollout; tensor-valued indices
    # from Isaac actuator groups must be converted to ordinary Python integers.
    json.dumps(metadata)
    profiles = [("zero", (0., 0., 0.)), ("forward06", (.6, 0., 0.)), ("forward10", (1., 0., 0.))]
    original_write = robot.write_data_to_sim
    for label, velocity in profiles:
        buffers = {key: torch.empty((capacity,) + getattr(robot.data, key)[0].shape,
                                    dtype=getattr(robot.data, key).dtype, device=robot.device)
                   for key in fields}
        command_buffer = torch.empty((capacity, 3), device=robot.device)
        cursor = 0
        last_counter = -1

        def record_write():
            nonlocal cursor, last_counter
            original_write()
            counter = sim_env._sim_step_counter
            # Reset can perform a second write at the same physics step. That
            # write is not an integrated transition, and must not be counted.
            if counter == last_counter:
                return
            if cursor >= capacity:
                raise RuntimeError("Unexpected extra physics step in torque capture")
            last_counter = counter
            for key in fields:
                buffers[key][cursor].copy_(getattr(robot.data, key)[0])
            command_buffer[cursor].copy_(command.command[0])
            cursor += 1

        command.cfg.ranges.lin_vel_x = (velocity[0], velocity[0])
        command.cfg.ranges.lin_vel_y = (velocity[1], velocity[1])
        command.cfg.ranges.ang_vel_z = (velocity[2], velocity[2])
        with torch.inference_mode():
            obs, _ = env.reset()
            command.command[:] = command.command.new_tensor(velocity)
            obs = env.get_observations()
            policy_nn.reset(torch.ones(1, dtype=torch.bool, device=robot.device))
        robot.write_data_to_sim = record_write
        termination = None
        print(f"[TORQUE_AUDIT] Starting {label}: command={velocity}, duration={seconds}s", flush=True)
        try:
            with torch.inference_mode():
                for step in range(control_steps):
                    if not simulation_app.is_running():
                        termination = {"reason": "simulation_app_closed"}
                        break
                    actions = policy(obs)
                    obs, _, dones, extras = env.step(actions)
                    if bool(dones[0]):
                        termination = {"control_step": step+1,
                                       "terms": {name: bool(sim_env.termination_manager.get_term(name)[0])
                                                 for name in sim_env.termination_manager.active_terms}}
                        print(f"[TORQUE_AUDIT] Termination: {termination}", flush=True)
                        break
                    if (step+1) % 500 == 0:
                        print(f"[TORQUE_AUDIT] {label}: {(step+1)*sim_env.step_dt:.1f}s, samples={cursor}, "
                              f"vx={float(robot.data.root_lin_vel_b[0,0]):.3f}", flush=True)
        finally:
            robot.write_data_to_sim = original_write
            arrays = {key: tensor[:cursor].detach().cpu().numpy() for key, tensor in buffers.items()}
            arrays["command"] = command_buffer[:cursor].detach().cpu().numpy()
            arrays["time_s"] = np.arange(cursor) * sim_env.physics_dt
            np.savez_compressed(output / f"{label}.npz", **arrays)
            metadata["profiles"].append({"name": label, "command": velocity, "samples": cursor,
                                         "duration_s": cursor*sim_env.physics_dt, "termination": termination})
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[TORQUE_AUDIT] Saved {output / (label + '.npz')}", flush=True)
    print(f"[TORQUE_AUDIT] COMPLETE: {output}", flush=True)
