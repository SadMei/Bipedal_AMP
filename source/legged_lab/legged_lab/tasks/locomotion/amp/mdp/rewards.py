from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def base_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the robot base drops below a minimum world-frame height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def lin_vel_xy_base_frame_error_excess_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    max_excess: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excess XY velocity error in the command's base frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    error = torch.linalg.vector_norm(command - asset.data.root_lin_vel_b[:, :2], dim=1)
    excess = torch.clamp(error - threshold, min=0.0, max=max_excess)
    return torch.square(excess)


def lin_vel_xy_yaw_frame_error_excess_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    max_excess: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Backward-compatible name for base-frame excess velocity error."""
    return lin_vel_xy_base_frame_error_excess_l2(env, command_name, threshold, max_excess, asset_cfg)


def zero_yaw_initial_heading_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float = 1.0e-3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track XY velocity in the episode-initial yaw frame for straight commands.

    ``StraightMixtureVelocityCommand.heading_target`` is initialized from the
    robot heading after every reset and remains unchanged when the commanded yaw
    rate is exactly zero.  Rotating world velocity into that frame prevents a
    policy from preserving its forward reward by slowly turning its body.  The
    term is disabled for turning commands, which continue to use base-frame
    velocity tracking.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        raise AttributeError(f"Command term {command_name!r} does not expose an episode heading target.")

    target_yaw = command_term.heading_target
    zeros = torch.zeros_like(target_yaw)
    target_yaw_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, target_yaw)
    lin_vel_initial_heading = math_utils.quat_apply_inverse(target_yaw_quat_w, asset.data.root_lin_vel_w)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - lin_vel_initial_heading[:, :2]), dim=1)
    zero_turn = torch.abs(command[:, 2]) < command_threshold
    return torch.exp(-lin_vel_error / std**2) * zero_turn


def zero_command_yaw_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw rate strongly when the requested yaw rate is near zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    zero_turn = torch.abs(yaw_command) < command_threshold
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * zero_turn


class zero_command_heading_hold_exp(ManagerTermBase):
    """Reward holding the episode's initial world heading for zero-turn commands.

    The target is captured after every environment reset, including reference-
    motion resets.  It therefore does not assume that all reference clips start
    at world yaw zero.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: RigidObject = env.scene[asset_cfg.name]
        self.target_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2].clone()

    def reset(self, env_ids):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: RigidObject = self._env.scene[asset_cfg.name]
        yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w[env_ids])[2]
        self.target_yaw[env_ids] = yaw

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
        heading_error = math_utils.wrap_to_pi(yaw - self.target_yaw)
        yaw_command = env.command_manager.get_command(command_name)[:, 2]
        zero_turn = torch.abs(yaw_command) < command_threshold
        return torch.exp(-torch.square(heading_error) / std**2) * zero_turn


def integrated_heading_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking the heading obtained by integrating the yaw-rate command."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        raise AttributeError(f"Command term {command_name!r} does not expose an integrated heading target.")
    heading_error = math_utils.wrap_to_pi(asset.data.heading_w - command_term.heading_target)
    return torch.exp(-torch.square(heading_error) / std**2)
