from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化关节默认位置，并同步关节位置动作的 offset。"""
    asset: Articulation = env.scene[asset_cfg.name]

    # 保存未随机化的标称关节位置，供模型导出时使用。
    asset.data.default_joint_pos_nominal = asset.data.default_joint_pos[0].clone()

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
        selected_joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
        selected_joint_ids = joint_ids

    if pos_distribution_params is None:
        return

    pos = _randomize_prop_by_op(
        asset.data.default_joint_pos.to(asset.device).clone(),
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )
    if isinstance(joint_ids, slice):
        randomized_pos = pos[env_ids]
        asset.data.default_joint_pos[env_ids] = randomized_pos
    else:
        randomized_pos = pos[env_ids[:, None], joint_ids]
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = randomized_pos

    # Action term 的列顺序由 JOINT_NAMES 决定，不能直接使用资产关节列索引。
    action_term = env.action_manager.get_term("joint_pos")
    if isinstance(action_term._joint_ids, slice):
        action_joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        action_joint_ids = torch.as_tensor(action_term._joint_ids, dtype=torch.long, device=asset.device)
    action_mask = torch.isin(action_joint_ids, selected_joint_ids)
    action_columns = torch.nonzero(action_mask, as_tuple=False).squeeze(-1)
    action_asset_joint_ids = action_joint_ids[action_columns]
    action_offsets = asset.data.default_joint_pos[env_ids[:, None], action_asset_joint_ids]
    action_term._offset[env_ids[:, None], action_columns] = action_offsets


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """在指定范围内随机偏移刚体质心。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)
