from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import ResidualWorldModelCfg, WorldModelStateSpec


MODEL_ORIENTATION_INPUT = "yaw_invariant_tilt"


_DYNAMIC_FIELDS = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "root_quat",
    "base_lin_vel",
    "root_height",
    "foot_contact",
    "foot_height",
)
_CONTEXT_FIELDS = ("reference_joint_pos", "reference_joint_vel", "previous_action")


def pack_frame(spec: WorldModelStateSpec, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Pack named batched tensors into the stable robot-independent WM layout."""

    spec.validate()
    expected = {
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": spec.num_joints,
        "joint_vel": spec.num_joints,
        "root_quat": 4,
        "base_lin_vel": 3,
        "root_height": 1,
        "foot_contact": spec.num_feet,
        "foot_height": spec.num_feet,
        "reference_joint_pos": spec.num_joints,
        "reference_joint_vel": spec.num_joints,
        "previous_action": spec.num_joints,
    }
    missing = [name for name in expected if name not in values]
    if missing:
        raise KeyError(f"Missing world-model frame fields: {missing}.")

    tensors = []
    batch_shape = None
    for name in (*_DYNAMIC_FIELDS, *_CONTEXT_FIELDS):
        tensor = values[name]
        if tensor.ndim < 2:
            raise ValueError(f"{name} must be batched, got shape {tuple(tensor.shape)}.")
        if tensor.shape[-1] != expected[name]:
            raise ValueError(f"{name} must end in {expected[name]}, got {tuple(tensor.shape)}.")
        if batch_shape is None:
            batch_shape = tensor.shape[:-1]
        elif tensor.shape[:-1] != batch_shape:
            raise ValueError(f"{name} batch shape {tensor.shape[:-1]} does not match {batch_shape}.")
        tensors.append(tensor)

    frame = torch.cat(tensors, dim=-1)
    if frame.shape[-1] != spec.frame_dim:
        raise RuntimeError(f"Packed frame has {frame.shape[-1]} values, expected {spec.frame_dim}.")
    return frame


def dynamic_state(spec: WorldModelStateSpec, frame: torch.Tensor) -> torch.Tensor:
    _check_last_dim(frame, spec.frame_dim, "frame")
    return frame[..., : spec.dynamic_dim]


def context_state(spec: WorldModelStateSpec, frame: torch.Tensor) -> torch.Tensor:
    _check_last_dim(frame, spec.frame_dim, "frame")
    return frame[..., spec.dynamic_dim :]


def state_delta_target(
    spec: WorldModelStateSpec,
    real_next_frame: torch.Tensor,
    nominal_next_frame: torch.Tensor,
) -> torch.Tensor:
    """Return the next-state residual used by the directly supervised mode.

    The two transitions must start from aligned states and receive the same
    action.  Context is intentionally excluded because reference motion and
    previous action are exogenous inputs rather than dynamics predictions.
    """

    real_dynamic = dynamic_state(spec, real_next_frame).clone()
    nominal_dynamic = dynamic_state(spec, nominal_next_frame)

    # q and -q encode the same rotation.  Align signs before subtraction so a
    # harmless quaternion sign flip cannot become a large learned sim-real gap.
    quat_start = 6 + 2 * spec.num_joints
    quat_slice = slice(quat_start, quat_start + 4)
    real_quat = real_dynamic[..., quat_slice]
    nominal_quat = nominal_dynamic[..., quat_slice]
    sign = torch.where((real_quat * nominal_quat).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    real_dynamic[..., quat_slice] = real_quat * sign
    return real_dynamic - nominal_dynamic


def model_input(
    config: ResidualWorldModelCfg,
    history: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Build the current privileged-state plus action input."""

    spec = config.state

    if history.shape[-2:] != (spec.history_length, spec.frame_dim):
        raise ValueError(
            "history must end in "
            f"({spec.history_length}, {spec.frame_dim}), got {tuple(history.shape)}."
        )
    _check_last_dim(action, spec.action_dim, "action")
    if history.shape[:-2] != action.shape[:-1]:
        raise ValueError("history and action batch shapes must match.")
    current = history[..., -1, :]
    privileged = current[..., 3 : spec.root_height_slice.stop].clone()
    # Keep the project's body-frame/yaw-equivariant residual contract while
    # excluding temporal history and exogenous reference/action context.
    quat_start = (6 + 2 * spec.num_joints) - 3
    quaternion = privileged[..., quat_start : quat_start + 4]
    privileged = torch.cat(
        (
            privileged[..., :quat_start],
            yaw_invariant_tilt_quaternion(quaternion),
            privileged[..., quat_start + 4 :],
        ),
        dim=-1,
    )
    result = torch.cat((privileged, action), dim=-1)
    if result.shape[-1] != config.model_input_dim:
        raise RuntimeError(
            f"Current-state model input has {result.shape[-1]} values, "
            f"expected {config.model_input_dim}."
        )
    return result


def yaw_invariant_tilt_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Remove world yaw from a body-to-world ``wxyz`` quaternion."""

    _check_last_dim(quaternion, 4, "quaternion")
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=1.0e-6)
    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half_yaw = 0.5 * yaw
    yaw_inv = torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            -torch.sin(half_yaw),
        ),
        dim=-1,
    )
    aw, ax, ay, az = yaw_inv.unbind(dim=-1)
    bw, bx, by, bz = quaternion.unbind(dim=-1)
    tilt = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )
    sign = torch.where(tilt[..., :1] < 0.0, -1.0, 1.0)
    return tilt * sign


@dataclass
class FrameHistory:
    """Per-environment frame stack with reset-safe episode boundaries."""

    spec: WorldModelStateSpec
    tensor: torch.Tensor

    @classmethod
    def create(
        cls,
        spec: WorldModelStateSpec,
        initial_frame: torch.Tensor,
    ) -> "FrameHistory":
        spec.validate()
        _check_last_dim(initial_frame, spec.frame_dim, "initial_frame")
        history = initial_frame.unsqueeze(-2).expand(*initial_frame.shape[:-1], spec.history_length, -1).clone()
        return cls(spec=spec, tensor=history)

    def push(self, frame: torch.Tensor, reset: torch.Tensor | None = None) -> torch.Tensor:
        """Append a frame; reset environments are filled with their new frame."""

        _check_last_dim(frame, self.spec.frame_dim, "frame")
        if frame.shape != self.tensor.shape[:-2] + (self.spec.frame_dim,):
            raise ValueError(
                f"frame shape {tuple(frame.shape)} is incompatible with history {tuple(self.tensor.shape)}."
            )

        self.tensor = torch.cat((self.tensor[..., 1:, :], frame.unsqueeze(-2)), dim=-2)
        if reset is not None:
            if reset.shape != frame.shape[:-1]:
                raise ValueError(f"reset shape {tuple(reset.shape)} must equal batch shape {frame.shape[:-1]}.")
            reset_mask = reset.to(device=frame.device, dtype=torch.bool)
            replacement = frame.unsqueeze(-2).expand_as(self.tensor)
            self.tensor = torch.where(reset_mask[..., None, None], replacement, self.tensor)
        return self.tensor


def _check_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} must end in {expected}, got {tuple(tensor.shape)}.")
