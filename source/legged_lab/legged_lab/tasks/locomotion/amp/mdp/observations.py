from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


class zero_command_heading_error(ManagerTermBase):
    """Encode error from the episode's initial yaw for zero-yaw commands.

    The two outputs are ``sin(error)`` and ``cos(error)``.  For commanded
    turns the term emits the neutral value ``[0, 1]`` so it does not impose an
    episode-initial heading target on turning behavior.
    """

    def __init__(self, env: ManagerBasedEnv, cfg: ObservationTermCfg):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        robot: Articulation = env.scene[asset_cfg.name]
        self.target_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2].clone()

    def reset(self, env_ids=None):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        robot: Articulation = self._env.scene[asset_cfg.name]
        if env_ids is None:
            env_ids = slice(None)
        self.target_yaw[env_ids] = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[env_ids])[2]

    def __call__(
        self,
        env: ManagerBasedEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[asset_cfg.name]
        yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2]
        error = math_utils.wrap_to_pi(self.target_yaw - yaw)
        zero_turn = torch.abs(env.command_manager.get_command(command_name)[:, 2]) < command_threshold
        sin_error = torch.where(zero_turn, torch.sin(error), torch.zeros_like(error))
        cos_error = torch.where(zero_turn, torch.cos(error), torch.ones_like(error))
        return torch.stack((sin_error, cos_error), dim=-1)


def integrated_heading_error(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encode error from the yaw-rate-integrated command heading."""
    robot: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        raise AttributeError(f"Command term {command_name!r} does not expose an integrated heading target.")
    error = math_utils.wrap_to_pi(command_term.heading_target - robot.data.heading_w)
    return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


def root_local_rot_tan_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local)  # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs


def ref_root_lin_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    """Reference root linear velocity expressed in the reference body frame."""
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_root_vel_w = animation_term.get_root_vel_w()
    ref_root_quat = animation_term.get_root_quat()
    ref_root_vel_b = math_utils.quat_apply_inverse(ref_root_quat, ref_root_vel_w)

    if flatten_steps_dim:
        return ref_root_vel_b.reshape(env.num_envs, -1)
    return ref_root_vel_b


def ref_velocity_commands(
    env: ManagerBasedAnimationEnv,
    animation: str,
    command_name: str,
) -> torch.Tensor:
    """Repeat the current velocity command across all reference AMP steps."""
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    command = env.command_manager.get_command(command_name)
    return command.unsqueeze(1).expand(-1, animation_term.num_steps, -1)
