from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .config import WorldModelStateSpec
from .interfaces import DataCollectionEnv, PipelineState
from .replay import ReplayBuffer, TransitionBatch
from .state import FrameHistory


ActionFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class CollectionMetrics:
    transitions: int = 0
    episodes: int = 0
    reward_sum: float = 0.0


class TransitionCollector:
    """Stateful collector that preserves causal histories across calls."""

    def __init__(self, env: DataCollectionEnv, state_spec: WorldModelStateSpec) -> None:
        self.env = env
        self.state_spec = state_spec
        self.state: PipelineState | None = None
        self.history: FrameHistory | None = None

    def reset(self) -> None:
        self.state = self.env.reset()
        self.history = FrameHistory.create(self.state_spec, self.state.wm_frame)

    @torch.no_grad()
    def collect(self, buffer: ReplayBuffer, steps: int, action_fn: ActionFn) -> CollectionMetrics:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if self.state is None or self.history is None:
            self.reset()
        assert self.state is not None and self.history is not None
        metrics = CollectionMetrics()

        for _ in range(steps):
            policy_obs = self.state.policy_obs
            current_history = self.history.tensor.clone()
            action = action_fn(policy_obs)
            result = self.env.step(action)
            self._validate_step(result.reward, result.terminated, result.truncated)
            done = torch.logical_or(result.terminated, result.truncated).reshape(-1)

            transition_history = torch.cat(
                (current_history[:, 1:, :], result.transition_next.wm_frame.unsqueeze(1)), dim=1
            )
            buffer.add(
                TransitionBatch(
                    policy_obs=policy_obs,
                    critic_obs=self._critic_or_policy(self.state, buffer),
                    action=action,
                    reward=result.reward.reshape(-1, 1),
                    next_policy_obs=result.transition_next.policy_obs,
                    next_critic_obs=self._critic_or_policy(result.transition_next, buffer),
                    terminated=result.terminated.reshape(-1, 1).bool(),
                    truncated=result.truncated.reshape(-1, 1).bool(),
                    wm_history=current_history,
                    next_wm_history=transition_history,
                    state_est_label=self._label_or_empty(self.state, buffer),
                    next_state_est_label=self._label_or_empty(result.transition_next, buffer),
                    state_est_label_valid=self._label_valid(self.state, buffer),
                    next_state_est_label_valid=self._label_valid(result.transition_next, buffer),
                    reference_index=self._reference_index_or_empty(self.state, buffer),
                    next_reference_index=self._reference_index_or_empty(result.transition_next, buffer),
                )
            )

            continuation_history = transition_history.clone()
            reset_history = result.continuation.wm_frame.unsqueeze(1).expand_as(continuation_history)
            continuation_history = torch.where(done[:, None, None], reset_history, continuation_history)
            self.history.tensor = continuation_history
            self.state = result.continuation

            metrics.transitions += self.env.num_envs
            metrics.episodes += int(done.sum())
            metrics.reward_sum += float(result.reward.sum())
        return metrics

    def _critic_or_policy(self, state: PipelineState, buffer: ReplayBuffer) -> torch.Tensor:
        value = state.policy_obs if state.critic_obs is None else state.critic_obs
        expected = (self.env.num_envs, buffer.critic_obs_dim)
        if value.shape != expected:
            raise ValueError(f"critic observation must have shape {expected}, got {tuple(value.shape)}.")
        return value

    def _label_or_empty(self, state: PipelineState, buffer: ReplayBuffer) -> torch.Tensor:
        if state.state_est_label is None:
            return state.policy_obs.new_zeros((self.env.num_envs, buffer.state_est_label_dim))
        label = state.state_est_label
        expected = (self.env.num_envs, buffer.state_est_label_dim)
        if label.shape != expected:
            raise ValueError(f"state estimator label must have shape {expected}, got {tuple(label.shape)}.")
        return label

    def _label_valid(self, state: PipelineState, buffer: ReplayBuffer) -> torch.Tensor:
        valid = state.state_est_label is not None and buffer.state_est_label_dim > 0
        if state.state_est_label_valid is not None:
            if state.state_est_label_valid.shape != (self.env.num_envs, 1):
                raise ValueError(
                    "state_est_label_valid must have shape "
                    f"({self.env.num_envs}, 1), got {tuple(state.state_est_label_valid.shape)}."
                )
            return state.state_est_label_valid.bool()
        return torch.full(
            (self.env.num_envs, 1),
            valid,
            device=state.policy_obs.device,
            dtype=torch.bool,
        )

    def _reference_index_or_empty(self, state: PipelineState, buffer: ReplayBuffer) -> torch.Tensor:
        if state.reference_index is None:
            return torch.zeros((self.env.num_envs, 1), device=state.policy_obs.device, dtype=torch.long)
        value = state.reference_index.reshape(self.env.num_envs, 1).to(dtype=torch.long)
        return value

    def _validate_step(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        for name, tensor in (("reward", reward), ("terminated", terminated), ("truncated", truncated)):
            if tensor.numel() != self.env.num_envs:
                raise ValueError(f"{name} must contain {self.env.num_envs} values, got {tuple(tensor.shape)}.")
