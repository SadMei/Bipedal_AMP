from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch


@dataclass
class TransitionBatch:
    """SAC transition plus the world-model state histories on both sides."""

    policy_obs: torch.Tensor
    critic_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_policy_obs: torch.Tensor
    next_critic_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    wm_history: torch.Tensor
    next_wm_history: torch.Tensor
    state_est_label: torch.Tensor
    next_state_est_label: torch.Tensor
    state_est_label_valid: torch.Tensor
    next_state_est_label_valid: torch.Tensor
    reference_index: torch.Tensor | None = None
    next_reference_index: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.policy_obs.shape[0]

    @property
    def done(self) -> torch.Tensor:
        return torch.logical_or(self.terminated, self.truncated)

    def to(self, device: torch.device | str) -> "TransitionBatch":
        return TransitionBatch(
            **{
                field.name: (
                    None
                    if getattr(self, field.name) is None
                    else getattr(self, field.name).to(device)
                )
                for field in fields(self)
            }
        )

    def index(self, index: torch.Tensor | slice) -> "TransitionBatch":
        return TransitionBatch(
            **{
                field.name: (
                    None
                    if getattr(self, field.name) is None
                    else getattr(self, field.name)[index]
                )
                for field in fields(self)
            }
        )

    @classmethod
    def cat(cls, batches: list["TransitionBatch"]) -> "TransitionBatch":
        if not batches:
            raise ValueError("Cannot concatenate an empty transition list.")
        values = {}
        for field in fields(cls):
            field_values = [getattr(batch, field.name) for batch in batches]
            if all(value is None for value in field_values):
                values[field.name] = None
            elif any(value is None for value in field_values):
                raise ValueError(f"Cannot concatenate mixed None/tensor field {field.name!r}.")
            else:
                values[field.name] = torch.cat(field_values)
        return cls(**values)


class ReplayBuffer:
    """Device-aware circular replay buffer with an explicit, versioned schema."""

    FORMAT_VERSION = 5

    def __init__(
        self,
        capacity: int,
        policy_obs_dim: int,
        action_dim: int,
        history_length: int,
        frame_dim: int,
        critic_obs_dim: int | None = None,
        state_est_label_dim: int = 0,
        storage_device: torch.device | str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.policy_obs_dim = int(policy_obs_dim)
        self.critic_obs_dim = int(policy_obs_dim if critic_obs_dim is None else critic_obs_dim)
        self.action_dim = int(action_dim)
        self.history_length = int(history_length)
        self.frame_dim = int(frame_dim)
        self.state_est_label_dim = int(state_est_label_dim)
        if self.state_est_label_dim < 0:
            raise ValueError("state_est_label_dim must be non-negative.")
        self.storage_device = torch.device(storage_device)
        self.position = 0
        self.size = 0

        common = {"device": self.storage_device, "dtype": torch.float32}
        self.policy_obs = torch.empty((capacity, policy_obs_dim), **common)
        self.critic_obs = torch.empty((capacity, self.critic_obs_dim), **common)
        self.action = torch.empty((capacity, action_dim), **common)
        self.reward = torch.empty((capacity, 1), **common)
        self.next_policy_obs = torch.empty((capacity, policy_obs_dim), **common)
        self.next_critic_obs = torch.empty((capacity, self.critic_obs_dim), **common)
        self.terminated = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.bool)
        self.truncated = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.bool)
        history_shape = (capacity, history_length, frame_dim)
        self.wm_history = torch.empty(history_shape, **common)
        self.next_wm_history = torch.empty(history_shape, **common)
        label_shape = (capacity, self.state_est_label_dim)
        self.state_est_label = torch.empty(label_shape, **common)
        self.next_state_est_label = torch.empty(label_shape, **common)
        self.state_est_label_valid = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.bool)
        self.next_state_est_label_valid = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.bool)
        self.reference_index = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.long)
        self.next_reference_index = torch.empty((capacity, 1), device=self.storage_device, dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def add(self, batch: TransitionBatch) -> None:
        count = len(batch)
        if batch.reference_index is None:
            batch = replace(batch, reference_index=torch.zeros((count, 1), dtype=torch.long, device=batch.action.device))
        if batch.next_reference_index is None:
            batch = replace(
                batch,
                next_reference_index=torch.zeros((count, 1), dtype=torch.long, device=batch.action.device),
            )
        self._validate_batch(batch)
        if len(batch) > self.capacity:
            batch = batch.index(slice(len(batch) - self.capacity, None))
        batch = batch.to(self.storage_device)
        count = len(batch)
        index = (torch.arange(count, device=self.storage_device) + self.position) % self.capacity
        for field in fields(TransitionBatch):
            getattr(self, field.name)[index] = getattr(batch, field.name)
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        index = torch.randint(self.size, (batch_size,), device=self.storage_device)
        batch = TransitionBatch(**{field.name: getattr(self, field.name)[index] for field in fields(TransitionBatch)})
        return batch if device is None else batch.to(device)

    def state_dict(self) -> dict:
        if self.size < self.capacity:
            index = torch.arange(self.size, device=self.storage_device)
        else:
            index = torch.cat(
                (
                    torch.arange(self.position, self.capacity, device=self.storage_device),
                    torch.arange(0, self.position, device=self.storage_device),
                )
            )
        return {
            "format_version": self.FORMAT_VERSION,
            "capacity": self.capacity,
            "policy_obs_dim": self.policy_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "action_dim": self.action_dim,
            "history_length": self.history_length,
            "frame_dim": self.frame_dim,
            "state_est_label_dim": self.state_est_label_dim,
            "size": self.size,
            "data": {field.name: getattr(self, field.name)[index].cpu() for field in fields(TransitionBatch)},
        }

    def load_state_dict(self, state: dict) -> None:
        version = state.get("format_version")
        if version not in (4, self.FORMAT_VERSION):
            raise ValueError(f"Unsupported replay format version: {version!r}.")
        expected = (
            self.policy_obs_dim,
            self.critic_obs_dim,
            self.action_dim,
            self.history_length,
            self.frame_dim,
            self.state_est_label_dim,
        )
        actual = (
            state["policy_obs_dim"],
            state["critic_obs_dim"],
            state["action_dim"],
            state["history_length"],
            state["frame_dim"],
            state["state_est_label_dim"],
        )
        if actual != expected:
            raise ValueError(f"Replay shape contract {actual} does not match current buffer {expected}.")
        if state["capacity"] != self.capacity:
            raise ValueError(f"Replay capacity {state['capacity']} does not match {self.capacity}.")
        data = dict(state["data"])
        size = int(state["size"])
        if version == 4:
            data["reference_index"] = torch.zeros((size, 1), dtype=torch.long)
            data["next_reference_index"] = torch.zeros((size, 1), dtype=torch.long)
        self.clear()
        self.add(TransitionBatch(**data))
        if self.size != int(state["size"]):
            raise RuntimeError(f"Restored {self.size} transitions, expected {state['size']}.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, storage_device: torch.device | str = "cpu") -> "ReplayBuffer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        buffer = cls(
            capacity=state["capacity"],
            policy_obs_dim=state["policy_obs_dim"],
            critic_obs_dim=state["critic_obs_dim"],
            action_dim=state["action_dim"],
            history_length=state["history_length"],
            frame_dim=state["frame_dim"],
            state_est_label_dim=state["state_est_label_dim"],
            storage_device=storage_device,
        )
        buffer.load_state_dict(state)
        return buffer

    def _validate_batch(self, batch: TransitionBatch) -> None:
        count = len(batch)
        expected = {
            "policy_obs": (count, self.policy_obs_dim),
            "critic_obs": (count, self.critic_obs_dim),
            "action": (count, self.action_dim),
            "reward": (count, 1),
            "next_policy_obs": (count, self.policy_obs_dim),
            "next_critic_obs": (count, self.critic_obs_dim),
            "terminated": (count, 1),
            "truncated": (count, 1),
            "wm_history": (count, self.history_length, self.frame_dim),
            "next_wm_history": (count, self.history_length, self.frame_dim),
            "state_est_label": (count, self.state_est_label_dim),
            "next_state_est_label": (count, self.state_est_label_dim),
            "state_est_label_valid": (count, 1),
            "next_state_est_label_valid": (count, 1),
            "reference_index": (count, 1),
            "next_reference_index": (count, 1),
        }
        for name, shape in expected.items():
            if getattr(batch, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(getattr(batch, name).shape)}.")


def sample_mixed(
    real_buffer: ReplayBuffer,
    synthetic_buffer: ReplayBuffer,
    batch_size: int,
    real_ratio: float,
    device: torch.device | str,
) -> TransitionBatch:
    """Sample one shuffled SAC batch from real and current-model data."""

    if not 0.0 <= real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1].")
    if len(real_buffer) == 0:
        raise ValueError("The real/target replay buffer is empty.")
    real_count = batch_size if len(synthetic_buffer) == 0 else int(round(batch_size * real_ratio))
    real_count = min(max(real_count, 0), batch_size)
    synthetic_count = batch_size - real_count
    batches = []
    if real_count:
        batches.append(real_buffer.sample(real_count, device))
    if synthetic_count:
        batches.append(synthetic_buffer.sample(synthetic_count, device))
    batch = TransitionBatch.cat(batches)
    permutation = torch.randperm(len(batch), device=batch.policy_obs.device)
    return batch.index(permutation)
