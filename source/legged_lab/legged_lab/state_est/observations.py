"""Simulator-only labels used to supervise the T1 state estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root height above the local environment origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def foot_contact(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for the selected feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    ).float()


def foot_height_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Selected foot heights above the local environment origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)


def foot_mode_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_time: float = 0.4,
) -> torch.Tensor:
    """Return clipped air/contact timers for each selected foot.

    The layout is ``[air_time_per_foot, contact_time_per_foot]``.  Clipping at
    the reward threshold retains all information used by the biped air-time
    reward while preventing long stance times from dominating an unnormalized
    privileged observation.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return torch.cat((air_time.clamp(max=max_time), contact_time.clamp(max=max_time)), dim=-1)


def body_lin_vel_w_xy(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame XY linear velocity of selected bodies, flattened by body."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].flatten(start_dim=1)


def contact_violation_count(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Number of selected bodies whose recent net contact force exceeds a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_norm = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    is_contact = force_norm.max(dim=1)[0] > force_threshold
    return is_contact.float().sum(dim=1, keepdim=True)


def state_estimator_label(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the four simulator-only estimator targets.

    Layout: ``[base_height(1), base_lin_vel_b(3)]``.

    Foot contact and world-frame foot height remain available above for tasks
    that need explicit phase supervision, but are intentionally excluded from
    the AMP actor estimator. They are redundant/noisy for the current flat-
    terrain distribution-matching task and previously caused foot jitter.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cat((base_height(env, asset_cfg), asset.data.root_lin_vel_b), dim=-1)
