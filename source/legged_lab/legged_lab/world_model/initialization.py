from __future__ import annotations

import numpy as np
import torch

from .config import WorldModelStateSpec


def _quat_inverse_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a world vector into the frame expressed by a wxyz quaternion."""

    w = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * torch.cross(xyz, cross - w * vector, dim=-1)


def default_standing_frame(
    nominal,
    reference,
    spec: WorldModelStateSpec,
    *,
    reference_index: int = 0,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the deployment XML/controller stand with NPZ context.

    The physical state comes exclusively from the MuJoCo model and controller
    defaults. The motion frame supplies only exogenous reference q/dq. This is
    the state used when a robot is enabled before motion playback starts, and
    therefore must be present in WM/PPO closed-loop validation.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import mujoco

    device = torch.device(device)
    model = nominal._cpu_model
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[7:] = np.asarray(nominal.cfg.default_joint_pos)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    root_quat = torch.as_tensor(data.qpos[3:7], dtype=torch.float32, device=device).reshape(1, 4)
    gravity = _quat_inverse_rotate(
        root_quat,
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
    )
    q_model = torch.as_tensor(data.qpos[7:], dtype=torch.float32, device=device)
    q_frame = q_model[nominal._model_to_frame_torch.to(device)].reshape(1, spec.num_joints)
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (nominal.cfg.left_foot_body, nominal.cfg.right_foot_body)
    ]
    foot_height = torch.as_tensor(
        data.xpos[foot_ids, 2], dtype=torch.float32, device=device
    ).reshape(1, spec.num_feet)
    if nominal.cfg.collision_mode == "feet_spheres_only":
        contact = [
            float(np.min(data.geom_xpos[np.asarray(ids), 2] - model.geom_size[np.asarray(ids), 0]) <= 0.0)
            for ids in nominal._foot_geom_ids
        ]
    else:
        clearance = nominal.foot_clearance_cpu(data)
        contact = 1.0 / (1.0 + np.exp(np.clip(clearance / nominal.cfg.contact_smoothing, -80, 80)))

    ref = reference.get(reference_index)
    frame = torch.cat(
        (
            gravity,
            torch.zeros((1, 3), device=device),
            q_frame,
            torch.zeros((1, spec.num_joints), device=device),
            root_quat,
            torch.zeros((1, 3), device=device),
            torch.tensor([[float(data.qpos[2])]], device=device),
            torch.as_tensor(np.asarray(contact), dtype=torch.float32, device=device).reshape(1, spec.num_feet),
            foot_height,
            ref.joint_pos.reshape(1, spec.num_joints).to(device),
            ref.joint_vel.reshape(1, spec.num_joints).to(device),
            torch.zeros((1, spec.num_joints), device=device),
        ),
        dim=-1,
    )
    if frame.shape != (1, spec.frame_dim):
        raise RuntimeError(
            f"default standing frame has shape {tuple(frame.shape)}, expected (1, {spec.frame_dim})"
        )
    root_xy = torch.as_tensor(data.qpos[:2], dtype=torch.float32, device=device).reshape(1, 2)
    return frame.expand(batch_size, -1).clone(), root_xy.expand(batch_size, -1).clone()
