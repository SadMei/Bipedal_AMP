"""Reference-motion provider for model-based tracking rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch


def _quat_inv(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (lw*rw-lx*rx-ly*ry-lz*rz, lw*rx+lx*rw+ly*rz-lz*ry,
         lw*ry-lx*rz+ly*rw+lz*rx, lw*rz+lx*ry-ly*rx+lz*rw),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    return vector + 2.0 * (
        quat[..., :1] * torch.cross(xyz, vector, dim=-1)
        + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


@dataclass(frozen=True)
class MotionReferenceBatch:
    """Reference values at one motion frame for a batch of environments."""

    index: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    contact_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "MotionReferenceBatch":
        return MotionReferenceBatch(
            index=self.index.to(device),
            joint_pos=self.joint_pos.to(device),
            joint_vel=self.joint_vel.to(device),
            body_pos_w=self.body_pos_w.to(device),
            body_quat_w=self.body_quat_w.to(device),
            body_lin_vel_w=self.body_lin_vel_w.to(device),
            body_ang_vel_w=self.body_ang_vel_w.to(device),
            contact_mask=self.contact_mask.to(device),
        )


class MotionReferenceProvider:
    """Load a converted 50 Hz motion NPZ and advance its frame index explicitly.

    The provider is deliberately independent of the dynamics WM. It supplies
    reference data to the tracking observation/reward wrapper while the WM
    predicts only physical robot state.
    """

    def __init__(
        self,
        path: str | Path,
        device: torch.device | str = "cpu",
        require_contact_mask: bool = True,
        body_names: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.device = torch.device(device)
        with np.load(self.path, allow_pickle=False) as data:
            required = (
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "joint_names",
                "body_names",
            )
            if require_contact_mask:
                required = (*required, "contact_mask")
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"Motion reference is missing fields {missing}; available={data.files}.")
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"Invalid motion fps={fps!r} in {self.path}.")
            self.fps = fps
            self.joint_names = tuple(str(value) for value in data["joint_names"].tolist())
            source_body_names = tuple(str(value) for value in data["body_names"].tolist())
            if body_names is None:
                self.body_names = source_body_names
                body_indices = np.arange(len(source_body_names), dtype=np.int64)
            else:
                self.body_names = tuple(body_names)
                missing_bodies = [name for name in self.body_names if name not in source_body_names]
                if missing_bodies:
                    raise ValueError(
                        f"Requested bodies are missing from {self.path}: {missing_bodies}."
                    )
                body_indices = np.asarray(
                    [source_body_names.index(name) for name in self.body_names], dtype=np.int64
                )
            tensor_fields = [
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ]
            fields = {}
            for name in tensor_fields:
                value = np.asarray(data[name])
                if name.startswith("body_"):
                    value = value[:, body_indices]
                fields[name] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            fields["contact_mask"] = (
                torch.as_tensor(np.asarray(data["contact_mask"]), dtype=torch.float32, device=self.device)
                if "contact_mask" in data
                else torch.zeros((fields["joint_pos"].shape[0], 2), dtype=torch.float32, device=self.device)
            )
        self.joint_pos = fields["joint_pos"]
        self.joint_vel = fields["joint_vel"]
        self.body_pos_w = fields["body_pos_w"]
        self.body_quat_w = fields["body_quat_w"]
        self.body_lin_vel_w = fields["body_lin_vel_w"]
        self.body_ang_vel_w = fields["body_ang_vel_w"]
        self.contact_mask = fields["contact_mask"].to(torch.bool)
        self._alignment_motion_pos: torch.Tensor | None = None
        self._alignment_robot_pos: torch.Tensor | None = None
        self._alignment_yaw: torch.Tensor | None = None
        self._validate_shapes()

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def get(self, index: torch.Tensor | int) -> MotionReferenceBatch:
        index = torch.as_tensor(index, device=self.device, dtype=torch.long)
        index = index.clamp(0, self.frame_count - 1)
        scalar = index.ndim == 0
        flat_index = index.reshape(-1)
        body_pos = self.body_pos_w[flat_index]
        body_quat = self.body_quat_w[flat_index]
        body_lin_vel = self.body_lin_vel_w[flat_index]
        body_ang_vel = self.body_ang_vel_w[flat_index]
        if self._alignment_yaw is not None:
            if self._alignment_yaw.shape[0] != flat_index.shape[0]:
                raise ValueError(
                    "Reference alignment batch does not match get(index); call set_alignment() "
                    "with one anchor per rollout environment."
                )
            delta = body_pos - self._alignment_motion_pos[:, None, :]
            body_pos = self._alignment_robot_pos[:, None, :] + _quat_apply(
                self._alignment_yaw[:, None, :], delta
            )
            body_quat = _quat_mul(self._alignment_yaw[:, None, :], body_quat)
            body_lin_vel = _quat_apply(self._alignment_yaw[:, None, :], body_lin_vel)
            body_ang_vel = _quat_apply(self._alignment_yaw[:, None, :], body_ang_vel)
        if scalar:
            body_pos = body_pos[0]
            body_quat = body_quat[0]
            body_lin_vel = body_lin_vel[0]
            body_ang_vel = body_ang_vel[0]
        return MotionReferenceBatch(
            index=index,
            joint_pos=self.joint_pos[index],
            joint_vel=self.joint_vel[index],
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin_vel,
            body_ang_vel_w=body_ang_vel,
            contact_mask=self.contact_mask[index],
        )

    def set_alignment(
        self,
        reference_index: torch.Tensor | int,
        robot_anchor_pos: torch.Tensor,
        robot_anchor_quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Yaw/translation-align raw NPZ motion to each rollout's physical seed state."""

        reference_index = torch.as_tensor(
            reference_index, device=self.device, dtype=torch.long
        ).reshape(-1).clamp(0, self.frame_count - 1)
        robot_anchor_pos = torch.as_tensor(
            robot_anchor_pos, device=self.device, dtype=torch.float32
        ).reshape(-1, 3)
        robot_anchor_quat = torch.as_tensor(
            robot_anchor_quat, device=self.device, dtype=torch.float32
        ).reshape(-1, 4)
        if not (reference_index.shape[0] == robot_anchor_pos.shape[0] == robot_anchor_quat.shape[0]):
            raise ValueError("Alignment indices, positions and quaternions must have the same batch size.")
        motion_pos = self.body_pos_w[reference_index, 0]
        motion_quat = self.body_quat_w[reference_index, 0]
        yaw = _yaw_quat(_quat_mul(robot_anchor_quat, _quat_inv(motion_quat)))
        if env_ids is None:
            self._alignment_motion_pos = motion_pos.clone()
            self._alignment_robot_pos = robot_anchor_pos.clone()
            self._alignment_yaw = yaw.clone()
            return
        if self._alignment_yaw is None:
            raise RuntimeError("Full-batch set_alignment() must be called before subset alignment updates.")
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if env_ids.shape[0] != reference_index.shape[0]:
            raise ValueError("env_ids must match the subset alignment batch size.")
        self._alignment_motion_pos[env_ids] = motion_pos
        self._alignment_robot_pos[env_ids] = robot_anchor_pos
        self._alignment_yaw[env_ids] = yaw

    def advance(
        self,
        index: torch.Tensor,
        done: torch.Tensor | None = None,
        reset_index: int = 0,
    ) -> torch.Tensor:
        """Advance one 50 Hz frame and reset completed episodes."""

        next_index = torch.as_tensor(index, device=self.device, dtype=torch.long) + 1
        finished = next_index >= self.frame_count
        if done is not None:
            finished = torch.logical_or(finished, torch.as_tensor(done, device=self.device, dtype=torch.bool))
        return torch.where(finished, torch.as_tensor(reset_index, device=self.device), next_index)

    def _validate_shapes(self) -> None:
        count = self.frame_count
        expected_first = {
            "joint_vel": self.joint_pos.shape,
            "body_pos_w": (count, len(self.body_names), 3),
            "body_quat_w": (count, len(self.body_names), 4),
            "body_lin_vel_w": (count, len(self.body_names), 3),
            "body_ang_vel_w": (count, len(self.body_names), 3),
            "contact_mask": (count, 2),
        }
        if self.joint_pos.ndim != 2 or self.joint_pos.shape[1] != len(self.joint_names):
            raise ValueError(f"joint_pos shape/order mismatch: {tuple(self.joint_pos.shape)}.")
        for name, expected in expected_first.items():
            actual = tuple(getattr(self, name).shape)
            if actual != tuple(expected):
                raise ValueError(f"{name} has shape {actual}, expected {tuple(expected)}.")
        if not self.joint_names:
            raise ValueError("Motion reference must contain at least one joint.")
        if not all(torch.isfinite(value).all() for value in self._tensor_fields()):
            raise ValueError(f"Motion reference contains non-finite values: {self.path}.")

    def _tensor_fields(self) -> tuple[torch.Tensor, ...]:
        return (
            self.joint_pos,
            self.joint_vel,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
            self.contact_mask,
        )
