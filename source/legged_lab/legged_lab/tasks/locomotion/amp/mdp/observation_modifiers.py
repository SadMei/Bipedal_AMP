"""Stateful observation perturbations used by opt-in robustness experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


class EpisodeBiasDelay(ModifierBase):
    """Apply an episode-constant elementwise bias and a fixed 0..N frame delay.

    Biases and delay choices are sampled independently for every environment.
    The first sample following a reset is used to prime the delay buffer, so a
    reset can never expose a synthetic all-zero observation.
    """

    def __init__(self, cfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        low, high = cfg.bias_range
        if low > high:
            raise ValueError(f"Invalid observation bias range: {cfg.bias_range}")
        if not isinstance(cfg.max_delay_steps, int) or cfg.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be a non-negative integer")
        self.bias = torch.empty(data_dim, device=device)
        self.delay_steps = torch.empty(data_dim[0], dtype=torch.long, device=device)
        self.previous = torch.zeros(data_dim, device=device)
        self.initialized = torch.zeros(data_dim[0], dtype=torch.bool, device=device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        rows = slice(None) if env_ids is None else env_ids
        self.bias[rows].uniform_(*self._cfg.bias_range)
        shape = self.delay_steps[rows].shape
        self.delay_steps[rows] = torch.randint(
            0, self._cfg.max_delay_steps + 1, shape, device=self._device
        )
        self.initialized[rows] = False

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if tuple(data.shape) != self._data_dim:
            raise ValueError(f"Observation shape changed from {self._data_dim} to {tuple(data.shape)}")
        fresh = ~self.initialized
        if fresh.any():
            self.previous[fresh] = data[fresh]
            self.initialized[fresh] = True
        delayed = torch.where(
            (self.delay_steps > 0).reshape((-1,) + (1,) * (data.ndim - 1)),
            self.previous,
            data,
        )
        self.previous.copy_(data)
        return delayed + self.bias


@configclass
class EpisodeBiasDelayCfg(ModifierCfg):
    """Configuration for :class:`EpisodeBiasDelay`."""

    func: type[EpisodeBiasDelay] = EpisodeBiasDelay
    bias_range: tuple[float, float] = (0.0, 0.0)
    max_delay_steps: int = 1
