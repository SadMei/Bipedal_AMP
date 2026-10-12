from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class PipelineState:
    """Policy observation and self-contained world-model frame at one control tick."""

    policy_obs: torch.Tensor
    wm_frame: torch.Tensor
    state_est_label: torch.Tensor | None = None
    state_est_label_valid: torch.Tensor | None = None
    critic_obs: torch.Tensor | None = None
    reference_index: torch.Tensor | None = None


@dataclass
class EnvStepResult:
    """Environment step with explicit terminal and post-reset states.

    ``transition_next`` is the true s[t+1] stored in replay.  ``continuation``
    is the state used for the next action and may be a reset state for done
    environments.  Keeping both prevents auto-reset observations from
    corrupting transition targets.
    """

    transition_next: PipelineState
    continuation: PipelineState
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class DataCollectionEnv(Protocol):
    """Adapter contract for Isaac simulation or the target/real system."""

    num_envs: int
    action_dim: int

    def reset(self) -> PipelineState: ...

    def step(self, action: torch.Tensor) -> EnvStepResult: ...
