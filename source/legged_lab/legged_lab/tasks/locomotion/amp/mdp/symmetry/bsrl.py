"""Left-right policy symmetry for the BSRL 12-DoF observation contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_JOINT_COUNT = 12
_POLICY_LAYOUTS = {144: 3, 240: 5}
_LEFT_JOINTS = (0, 2, 4, 6, 8, 10)
_RIGHT_JOINTS = (1, 3, 5, 7, 9, 11)
# hip yaw, hip roll, hip pitch, knee pitch, ankle pitch, ankle roll
_PAIRED_SIGNS = (-1.0, -1.0, 1.0, 1.0, 1.0, -1.0)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and sagittal-plane-reflected policy states/actions."""
    del env

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
            raise ValueError(f"BSRL symmetry expects {_JOINT_COUNT} actions, got {actions.shape[-1]}.")
        batch_size = actions.shape[0]
        actions_aug = torch.empty((2 * batch_size, _JOINT_COUNT), device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _mirror_joint_data(actions)

    return obs_aug, actions_aug


def _mirror_policy_observation(observation: torch.Tensor) -> torch.Tensor:
    policy_obs_dim = observation.shape[-1]
    if policy_obs_dim not in _POLICY_LAYOUTS:
        raise ValueError(
            f"BSRL symmetry expects policy observation dim in {tuple(_POLICY_LAYOUTS)}, got {policy_obs_dim}."
        )
    history_length = _POLICY_LAYOUTS[policy_obs_dim]

    mirrored = observation.clone()
    offset = 0

    angular_sign = observation.new_tensor((-1.0, 1.0, -1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 3] *= angular_sign
        offset += 3

    rotation_sign = observation.new_tensor((1.0, -1.0, 1.0, 1.0, -1.0, 1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 6] *= rotation_sign
        offset += 6

    command_sign = observation.new_tensor((1.0, -1.0, -1.0))
    for _ in range(history_length):
        mirrored[..., offset : offset + 3] *= command_sign
        offset += 3

    for _ in range(3):
        for _ in range(history_length):
            mirrored[..., offset : offset + _JOINT_COUNT] = _mirror_joint_data(
                observation[..., offset : offset + _JOINT_COUNT]
            )
            offset += _JOINT_COUNT

    if offset != policy_obs_dim:
        raise RuntimeError(f"BSRL symmetry consumed {offset} observations, expected {policy_obs_dim}.")
    return mirrored


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    if joint_data.shape[-1] != _JOINT_COUNT:
        raise ValueError(f"BSRL symmetry expects {_JOINT_COUNT} joints, got {joint_data.shape[-1]}.")
    mirrored = torch.empty_like(joint_data)
    paired_signs = joint_data.new_tensor(_PAIRED_SIGNS)
    mirrored[..., _LEFT_JOINTS] = joint_data[..., _RIGHT_JOINTS] * paired_signs
    mirrored[..., _RIGHT_JOINTS] = joint_data[..., _LEFT_JOINTS] * paired_signs
    return mirrored
