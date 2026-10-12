from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch

from .config import WorldModelStateSpec
from .state import pack_frame

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor


ReferenceJointStateFn = Callable[[Any], tuple[torch.Tensor, torch.Tensor]]


def world_model_frame(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    force_threshold: float = 10.0,
    spec: WorldModelStateSpec | None = None,
    reference_joint_state_fn: ReferenceJointStateFn | None = None,
) -> torch.Tensor:
    """Build the unnormalized LIFT-style state from an Isaac environment.

    This is an opt-in data adapter.  Merely importing it does not add an
    observation group or alter PPO training.  The legacy WBT path resolves
    ``joint_pos``/``joint_vel`` from a command term.  Other projects must pass
    an explicit callback rather than relying on a project-specific command
    class or silently substituting zero reference context.
    """

    spec = spec or WorldModelStateSpec()
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if reference_joint_state_fn is None:
        if not hasattr(env, "command_manager"):
            raise TypeError(
                "reference_joint_state_fn is required when the environment has no command manager."
            )
        command = env.command_manager.get_term(command_name)
        if not hasattr(command, "joint_pos") or not hasattr(command, "joint_vel"):
            raise TypeError(
                f"Command term {command_name!r} does not expose joint_pos/joint_vel; "
                "pass reference_joint_state_fn explicitly."
            )
        reference_joint_pos, reference_joint_vel = command.joint_pos, command.joint_vel
    else:
        reference_joint_pos, reference_joint_vel = reference_joint_state_fn(env)

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    previous_action = env.action_manager.action
    foot_force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    foot_contact = (foot_force.norm(dim=-1).max(dim=1).values > force_threshold).to(joint_pos.dtype)
    foot_height = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        - env.scene.env_origins[:, 2].unsqueeze(-1)
    )
    root_height = asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]

    return pack_frame(
        spec,
        {
            "projected_gravity": asset.data.projected_gravity_b,
            "base_ang_vel": asset.data.root_ang_vel_b,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_quat": asset.data.root_quat_w,
            "base_lin_vel": asset.data.root_lin_vel_b,
            "root_height": root_height,
            "foot_contact": foot_contact,
            "foot_height": foot_height,
            "reference_joint_pos": reference_joint_pos,
            "reference_joint_vel": reference_joint_vel,
            "previous_action": previous_action,
        },
    )
