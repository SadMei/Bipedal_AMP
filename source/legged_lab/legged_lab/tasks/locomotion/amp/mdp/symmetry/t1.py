"""Left-right policy symmetry for the Men T1 25-DoF observation contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_JOINT_COUNT = 25
_POLICY_LAYOUTS = {261: (3, False), 267: (3, True), 435: (5, False), 445: (5, True)}

# Policy/action order is T1_JOINT_NAMES. Pitch and elbow/knee coordinates keep
# their sign under a sagittal-plane reflection; roll and yaw coordinates flip.
_LEFT_JOINTS = (0, 1, 2, 3, 4, 5, 17, 18, 19, 20)
_RIGHT_JOINTS = (6, 7, 8, 9, 10, 11, 21, 22, 23, 24)
_PAIRED_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0)
_CENTER_JOINTS = (12, 13, 14, 15, 16)
_CENTER_SIGNS = (-1.0, 1.0, 1.0, -1.0, -1.0)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and sagittal-plane-reflected policy states/actions."""
    del env  # Both supported T1 policy layouts are independent of scene state.

    if obs is None:
        obs_aug = None
    else:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_observation(obs["policy"])

    if actions is None:
        actions_aug = None
    else:
        if actions.shape[-1] != _JOINT_COUNT:
            raise ValueError(f"T1 symmetry expects {_JOINT_COUNT} actions, got {actions.shape[-1]}.")
        batch_size = actions.shape[0]
        actions_aug = torch.empty((2 * batch_size, _JOINT_COUNT), device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _mirror_joint_data(actions)

    return obs_aug, actions_aug


def _mirror_policy_observation(observation: torch.Tensor) -> torch.Tensor:
    policy_obs_dim = observation.shape[-1]
    if policy_obs_dim not in _POLICY_LAYOUTS:
        raise ValueError(
            f"T1 symmetry expects policy observation dim in {tuple(_POLICY_LAYOUTS)}, got {policy_obs_dim}."
        )
    history_length, has_heading = _POLICY_LAYOUTS[policy_obs_dim]

    mirrored = observation.clone()
    offset = 0

    # Angular velocity is an axial vector: (wx, wy, wz) -> (-wx, wy, -wz).
    angular_sign = observation.new_tensor((-1.0, 1.0, -1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 3] *= angular_sign
        offset += 3

    # root_local_rot_tan_norm stores the x and z rotation-matrix columns.
    rotation_sign = observation.new_tensor((1.0, -1.0, 1.0, 1.0, -1.0, 1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 6] *= rotation_sign
        offset += 6

    # Command is (vx, vy, yaw_rate).
    command_sign = observation.new_tensor((1.0, -1.0, -1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 3] *= command_sign
        offset += 3

    # Joint position, joint velocity, and previous action histories.
    for _ in range(3):
        for _ in range(history_length):
            mirrored[..., offset : offset + _JOINT_COUNT] = _mirror_joint_data(
                observation[..., offset : offset + _JOINT_COUNT]
            )
            offset += _JOINT_COUNT

    # integrated_heading_error is (sin(error), cos(error)); reflection negates
    # the signed yaw error but leaves its cosine unchanged.
    if has_heading:
        heading_sign = observation.new_tensor((-1.0, 1.0))
        for _ in range(history_length):
            mirrored[..., offset : offset + 2] *= heading_sign
            offset += 2

    if offset != policy_obs_dim:
        raise RuntimeError(f"T1 symmetry consumed {offset} observations, expected {policy_obs_dim}.")
    return mirrored


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    if joint_data.shape[-1] != _JOINT_COUNT:
        raise ValueError(f"T1 symmetry expects {_JOINT_COUNT} joints, got {joint_data.shape[-1]}.")

    mirrored = torch.empty_like(joint_data)
    paired_signs = joint_data.new_tensor(_PAIRED_SIGNS)
    center_signs = joint_data.new_tensor(_CENTER_SIGNS)
    mirrored[..., _LEFT_JOINTS] = joint_data[..., _RIGHT_JOINTS] * paired_signs
    mirrored[..., _RIGHT_JOINTS] = joint_data[..., _LEFT_JOINTS] * paired_signs
    mirrored[..., _CENTER_JOINTS] = joint_data[..., _CENTER_JOINTS] * center_signs
    return mirrored
