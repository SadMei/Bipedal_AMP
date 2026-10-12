"""Strict IsaacLab/AMP adapter for world-model replay collection.

This module is opt-in.  It never registers an observation term and therefore
cannot change an existing AMP policy or training run merely by being imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .config import WorldModelStateSpec
from .isaac_adapter import world_model_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


BSRL_12_JOINT_NAMES = (
    "joint_left_hip_yaw",
    "joint_right_hip_yaw",
    "joint_left_hip_roll",
    "joint_right_hip_roll",
    "joint_left_hip_pitch",
    "joint_right_hip_pitch",
    "joint_left_knee_pitch",
    "joint_right_knee_pitch",
    "joint_left_ankle_pitch",
    "joint_right_ankle_pitch",
    "joint_left_ankle_roll",
    "joint_right_ankle_roll",
)


def bsrl_world_model_state_spec(history_length: int = 3) -> WorldModelStateSpec:
    """Return the BSRL 12-DoF WM shape contract.

    Resulting dimensions are dynamic=42, context=36, frame=78,
    model-input=47, action=12 and generalized-force=18.
    """

    spec = WorldModelStateSpec(num_joints=12, num_feet=2, history_length=history_length)
    spec.validate()
    return spec


def bsrl_estimator_actor_indices(history_length: int = 3) -> tuple[int, ...]:
    """Indices for AMP BSRL's term-major history -> current-frame actor view."""

    term_dims = (3, 6, 3, 12, 12, 12)
    if history_length <= 0:
        raise ValueError("history_length must be positive.")
    indices: list[int] = []
    offset = 0
    for width in term_dims:
        indices.extend(
            range(
                offset + (history_length - 1) * width,
                offset + history_length * width,
            )
        )
        offset += history_length * width
    return tuple(indices)


def amp_reference_joint_state(
    env: ManagerBasedEnv,
    animation_name: str = "animation",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the current reference q/dq from AMP's animation clock."""

    if not hasattr(env, "animation_manager"):
        raise TypeError("AMP world-model collection requires env.animation_manager.")
    animation = env.animation_manager.get_term(animation_name)
    for name in ("dof_pos_buffer", "dof_vel_buffer"):
        if not hasattr(animation, name):
            raise TypeError(f"Animation term {animation_name!r} has no {name}.")
    joint_pos = animation.dof_pos_buffer[:, 0, :]
    joint_vel = animation.dof_vel_buffer[:, 0, :]
    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"Invalid AMP reference shapes: q={tuple(joint_pos.shape)}, dq={tuple(joint_vel.shape)}."
        )
    return joint_pos, joint_vel


def amp_world_model_frame(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    *,
    animation_name: str = "animation",
    force_threshold: float = 10.0,
    spec: WorldModelStateSpec,
) -> torch.Tensor:
    """Build one WM frame using AMP's reference animation buffers."""

    return world_model_frame(
        env,
        asset_cfg,
        sensor_cfg,
        force_threshold=force_threshold,
        spec=spec,
        reference_joint_state_fn=lambda current: amp_reference_joint_state(
            current, animation_name
        ),
    )


@dataclass(frozen=True)
class AmpWorldModelContractAudit:
    num_envs: int
    joint_names: tuple[str, ...]
    action_dim: int
    frame_dim: int
    policy_dt: float


def validate_amp_world_model_contract(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    *,
    animation_name: str = "animation",
    force_threshold: float = 10.0,
    spec: WorldModelStateSpec,
    expected_joint_names: tuple[str, ...] | None = None,
) -> AmpWorldModelContractAudit:
    """Fail fast on ordering or dimensional drift before replay is written."""

    spec.validate()
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        indices = list(range(asset.num_joints))[joint_ids]
    else:
        indices = torch.as_tensor(joint_ids, device="cpu", dtype=torch.long).tolist()
    joint_names = tuple(asset.joint_names[index] for index in indices)
    if len(joint_names) != spec.num_joints:
        raise ValueError(
            f"Asset selector exposes {len(joint_names)} joints, expected {spec.num_joints}."
        )
    if expected_joint_names is not None and joint_names != expected_joint_names:
        raise ValueError(
            "AMP/WM joint order mismatch:\n"
            f"actual={joint_names}\nexpected={expected_joint_names}"
        )
    action_dim = int(env.action_manager.total_action_dim)
    if action_dim != spec.action_dim:
        raise ValueError(f"AMP action dim {action_dim} != WM action dim {spec.action_dim}.")
    frame = amp_world_model_frame(
        env,
        asset_cfg,
        sensor_cfg,
        animation_name=animation_name,
        force_threshold=force_threshold,
        spec=spec,
    )
    expected_shape = (int(env.num_envs), spec.frame_dim)
    if frame.shape != expected_shape or not torch.isfinite(frame).all():
        raise ValueError(
            f"AMP WM frame must be finite with shape {expected_shape}, got {tuple(frame.shape)}."
        )
    return AmpWorldModelContractAudit(
        num_envs=int(env.num_envs),
        joint_names=joint_names,
        action_dim=action_dim,
        frame_dim=spec.frame_dim,
        policy_dt=float(env.step_dt),
    )
