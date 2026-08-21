"""Run numerical diagnostics for the T1 AMP environment and uploaded motions."""

from __future__ import annotations

import argparse
from collections import defaultdict

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rollout_steps", type=int, default=320)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli, fast_shutdown=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import legged_lab.tasks  # noqa: F401
from legged_lab.assets.men_t1 import T1_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.config.t1.t1_amp_env_cfg import T1_KEY_BODY_NAMES


TASK_NAME = "LeggedLab-Isaac-AMP-T1-v0"


def _range(name: str, values: torch.Tensor) -> None:
    values = values.detach().float().cpu()
    finite = bool(torch.isfinite(values).all())
    env_flat = values.reshape(values.shape[0], -1)
    env_spread = (env_flat.max(dim=0).values - env_flat.min(dim=0).values).abs().max().item()
    print(
        f"[randomization] {name}: finite={finite} min={values.min().item():.6g} "
        f"max={values.max().item():.6g} max_env_spread={env_spread:.6g}"
    )


def _check_motion_and_fk(env) -> None:
    robot = env.scene["robot"]
    motion = env.motion_data_manager.get_term("motion_dataset")
    joint_ids, resolved_joint_names = robot.find_joints(list(T1_JOINT_NAMES), preserve_order=True)
    body_ids, resolved_body_names = robot.find_bodies(list(T1_KEY_BODY_NAMES), preserve_order=True)
    assert tuple(resolved_joint_names) == T1_JOINT_NAMES
    assert tuple(resolved_body_names) == T1_KEY_BODY_NAMES
    assert motion.num_dofs == len(T1_JOINT_NAMES)
    assert motion.num_key_bodies == len(T1_KEY_BODY_NAMES)

    tensors = {
        "root_pos_w": motion.root_pos_w,
        "root_quat": motion.root_quat,
        "root_vel_w": motion.root_vel_w,
        "root_ang_vel_w": motion.root_ang_vel_w,
        "dof_pos": motion.dof_pos,
        "dof_vel": motion.dof_vel,
        "key_body_pos_w": motion.key_body_pos_w,
    }
    for name, values in tensors.items():
        if not torch.isfinite(values).all():
            raise RuntimeError(f"motion tensor {name} contains NaN or Inf")
        print(f"[motion] {name}: shape={tuple(values.shape)} abs_max={values.abs().max().item():.6g}")

    sample_count = min(env.num_envs, 32)
    frame_ids = torch.linspace(0, motion.root_pos_w.shape[0] - 1, sample_count, device=env.device).long()
    env_ids = torch.arange(sample_count, device=env.device)
    root_pose = torch.cat((motion.root_pos_w[frame_ids], motion.root_quat[frame_ids]), dim=-1).clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    full_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    full_joint_vel = torch.zeros_like(full_joint_pos)
    full_joint_pos[:, joint_ids] = motion.dof_pos[frame_ids]
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(sample_count, 6, device=env.device), env_ids=env_ids)
    robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)

    simulated = robot.data.body_pos_w[env_ids[:, None], torch.tensor(body_ids, device=env.device)]
    simulated = simulated - env.scene.env_origins[env_ids].unsqueeze(1)
    expected = motion.key_body_pos_w[frame_ids]
    error = torch.linalg.vector_norm(simulated - expected, dim=-1)
    print(
        f"[fk] samples={sample_count} mean_error_m={error.mean().item():.8f} "
        f"max_error_m={error.max().item():.8f}"
    )
    if error.max() > 2.0e-3:
        raise RuntimeError(f"T1 reference/0722 URDF FK mismatch: max key-body error {error.max().item():.6f} m")

    motion_ids = motion.sample_motions(4096)
    times = motion.sample_times(motion_ids)
    sampled = motion.get_motion_state(motion_ids, times)
    for name, values in sampled.items():
        if not torch.isfinite(values).all():
            raise RuntimeError(f"interpolated motion state {name} contains NaN or Inf")
        print(f"[interpolation] {name}: abs_max={values.abs().max().item():.6g}")


def _check_randomization(env) -> None:
    robot = env.scene["robot"]
    view = robot.root_physx_view
    _range("joint_default_offset", robot.data.default_joint_pos - robot.data.default_joint_pos[0:1])
    _range("material_static", view.get_material_properties()[..., 0])
    _range("material_dynamic", view.get_material_properties()[..., 1])
    _range("material_restitution", view.get_material_properties()[..., 2])
    _range("mass_ratio", view.get_masses().to(env.device) / robot.data.default_mass.to(env.device))
    pelvis_id = robot.body_names.index("pelvis_Link")
    pelvis_com = view.get_coms()[:, pelvis_id, :3].to(env.device)
    _range("pelvis_com_centered", pelvis_com - pelvis_com.mean(dim=0, keepdim=True))
    _range(
        "stiffness_ratio",
        view.get_dof_stiffnesses().to(env.device)
        / robot.data.default_joint_stiffness.to(env.device).clamp_min(1.0e-8),
    )
    _range(
        "damping_ratio",
        view.get_dof_dampings().to(env.device)
        / robot.data.default_joint_damping.to(env.device).clamp_min(1.0e-8),
    )
    default_armature = robot.data.default_joint_armature.to(env.device)
    nonzero_armature = default_armature.abs() > 1.0e-8
    armature_ratio = view.get_dof_armatures().to(env.device)[nonzero_armature] / default_armature[nonzero_armature]
    _range("nonzero_armature_ratio", armature_ratio.reshape(env.num_envs, -1))
    friction = view.get_dof_friction_properties().to(env.device)
    _range("joint_static_friction", friction[..., 0])
    _range("joint_dynamic_friction", friction[..., 1])
    _range("joint_viscous_friction", friction[..., 2])

    wrench = robot.permanent_wrench_composer
    pelvis_force = wrench.composed_force_as_torch[:, pelvis_id]
    pelvis_torque = wrench.composed_torque_as_torch[:, pelvis_id]
    _range("pelvis_external_force", pelvis_force)
    _range("pelvis_external_torque", pelvis_torque)
    if not wrench.active or pelvis_force.abs().max() == 0.0 or pelvis_torque.abs().max() == 0.0:
        raise RuntimeError("base_external_force_torque reset event did not populate the PhysX wrench buffer")

    for actuator_name, actuator in robot.actuators.items():
        lags = actuator.positions_delay_buffer.time_lags
        print(
            f"[delay] {actuator_name}: min={lags.min().item()} max={lags.max().item()} "
            f"unique={lags.unique().sort().values.tolist()}"
        )
        if lags.min() < actuator.cfg.min_delay or lags.max() > actuator.cfg.max_delay:
            raise RuntimeError(f"actuator {actuator_name} has an invalid command delay")


def _rollout_rewards_and_interval_event(env) -> None:
    reward_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    terminated_count = 0
    timeout_count = 0
    push_count = 0
    interval_index = env.event_manager._mode_term_names["interval"].index("push_robot")
    previous_timer = env.event_manager._interval_term_time_left[interval_index].clone()
    action_dim = env.action_manager.total_action_dim

    for step in range(args_cli.rollout_steps):
        actions = torch.empty(env.num_envs, action_dim, device=env.device).uniform_(-1.0, 1.0)
        _, reward, terminated, timeout, _ = env.step(actions)
        if not torch.isfinite(reward).all():
            raise RuntimeError(f"non-finite total reward at rollout step {step}")
        step_terms = env.reward_manager._step_reward.detach()
        if not torch.isfinite(step_terms).all():
            raise RuntimeError(f"non-finite reward term at rollout step {step}")
        for index, name in enumerate(env.reward_manager.active_terms):
            reward_values[name].append(step_terms[:, index].cpu())
        terminated_count += int(terminated.sum().item())
        timeout_count += int(timeout.sum().item())

        timer = env.event_manager._interval_term_time_left[interval_index]
        push_count += int((timer > previous_timer).sum().item())
        previous_timer = timer.clone()

    print(
        f"[rollout] steps={args_cli.rollout_steps} terminated={terminated_count} "
        f"timeouts={timeout_count} interval_pushes={push_count}"
    )
    if push_count == 0 and args_cli.rollout_steps * env.step_dt > 5.0:
        raise RuntimeError("push_robot interval event did not trigger")
    for name in env.reward_manager.active_terms:
        values = torch.cat(reward_values[name])
        nonzero_fraction = (values.abs() > 1.0e-8).float().mean().item()
        print(
            f"[reward] {name}: min={values.min().item():.6g} max={values.max().item():.6g} "
            f"mean={values.mean().item():.6g} nonzero={nonzero_fraction:.4f}"
        )


def main() -> None:
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
    try:
        _check_motion_and_fk(env)
        env.reset()
        _check_randomization(env)
        _rollout_rewards_and_interval_event(env)
        print("[diagnostic] PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
