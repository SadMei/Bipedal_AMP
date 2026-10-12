from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MethodType
from typing import Any

import torch

from .interfaces import EnvStepResult, PipelineState


PolicyObservationFn = Callable[[Any], torch.Tensor]
WorldModelFrameFn = Callable[[Any], torch.Tensor]
TerminalPolicyObservationFn = Callable[[Any], torch.Tensor]
CriticObservationFn = Callable[[Any], torch.Tensor]
StateEstimatorLabelFn = Callable[[Any], torch.Tensor]
ReferenceIndexFn = Callable[[Any], torch.Tensor]


def concatenate_observation_groups(observation: Any, group_names: Sequence[str]) -> torch.Tensor:
    """Flatten selected Isaac observation groups in a declared order."""

    tensors = [observation[name] for name in group_names]
    if not all(isinstance(tensor, torch.Tensor) and tensor.ndim == 2 for tensor in tensors):
        raise ValueError("Selected Isaac observation groups must be batched 2-D tensors.")
    return torch.cat(tensors, dim=-1)


class IsaacLabDataCollectionEnv:
    """Adapt an auto-reset ManagerBasedRLEnv to the causal replay contract.

    Isaac resets done environments before returning observations.  This adapter
    temporarily wraps ``_reset_idx`` and captures the terminal world-model
    frame before reset, preventing episode-initial state from becoming the
    residual target.  The private hook is isolated here and restored by
    :meth:`close`; it should be rechecked when IsaacLab is upgraded.
    """

    def __init__(
        self,
        env: Any,
        policy_observation_fn: PolicyObservationFn,
        world_model_frame_fn: WorldModelFrameFn,
        action_dim: int,
        terminal_policy_observation_fn: TerminalPolicyObservationFn | None = None,
        critic_observation_fn: CriticObservationFn | None = None,
        terminal_critic_observation_fn: CriticObservationFn | None = None,
        state_estimator_label_fn: StateEstimatorLabelFn | None = None,
        terminal_state_estimator_label_fn: StateEstimatorLabelFn | None = None,
        reference_index_fn: ReferenceIndexFn | None = None,
        terminal_reference_index_fn: ReferenceIndexFn | None = None,
    ) -> None:
        self.env = getattr(env, "unwrapped", env)
        self.num_envs = int(self.env.num_envs)
        self.action_dim = int(action_dim)
        self.policy_observation_fn = policy_observation_fn
        self.world_model_frame_fn = world_model_frame_fn
        self.terminal_policy_observation_fn = terminal_policy_observation_fn
        self.critic_observation_fn = critic_observation_fn
        # The ordinary callback consumes the observation returned by env.step,
        # while terminal capture happens before reset and therefore needs an
        # explicit env-based callback.  Reusing the former here is a signature
        # mismatch that only surfaces when an environment terminates.
        self.terminal_critic_observation_fn = terminal_critic_observation_fn
        self.state_estimator_label_fn = state_estimator_label_fn
        self.reference_index_fn = reference_index_fn
        self.terminal_reference_index_fn = (
            reference_index_fn if terminal_reference_index_fn is None else terminal_reference_index_fn
        )
        self.terminal_state_estimator_label_fn = (
            state_estimator_label_fn
            if terminal_state_estimator_label_fn is None
            else terminal_state_estimator_label_fn
        )
        self._original_reset_idx = self.env._reset_idx
        self._capturing_step = False
        self._terminal_ids: torch.Tensor | None = None
        self._terminal_wm_frame: torch.Tensor | None = None
        self._terminal_policy_obs: torch.Tensor | None = None
        self._terminal_critic_obs: torch.Tensor | None = None
        self._terminal_state_est_label: torch.Tensor | None = None
        self._terminal_reference_index: torch.Tensor | None = None

        adapter = self

        def reset_idx_with_terminal_capture(env_self, env_ids):
            if adapter._capturing_step:
                ids = torch.as_tensor(env_ids, device=env_self.device, dtype=torch.long)
                adapter._terminal_ids = ids.clone()
                adapter._terminal_wm_frame = adapter.world_model_frame_fn(env_self)[ids].detach().clone()
                if adapter.terminal_policy_observation_fn is not None:
                    adapter._terminal_policy_obs = (
                        adapter.terminal_policy_observation_fn(env_self)[ids].detach().clone()
                    )
                if adapter.terminal_critic_observation_fn is not None:
                    adapter._terminal_critic_obs = (
                        adapter.terminal_critic_observation_fn(env_self)[ids].detach().clone()
                    )
                if adapter.terminal_state_estimator_label_fn is not None:
                    adapter._terminal_state_est_label = (
                        adapter.terminal_state_estimator_label_fn(env_self)[ids].detach().clone()
                    )
                if adapter.terminal_reference_index_fn is not None:
                    adapter._terminal_reference_index = (
                        adapter.terminal_reference_index_fn(env_self)[ids].detach().clone()
                    )
            return adapter._original_reset_idx(env_ids)

        self.env._reset_idx = MethodType(reset_idx_with_terminal_capture, self.env)

    def reset(self) -> PipelineState:
        result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        return self._state(observation)

    def step(self, action: torch.Tensor) -> EnvStepResult:
        self._terminal_ids = None
        self._terminal_wm_frame = None
        self._terminal_policy_obs = None
        self._terminal_critic_obs = None
        self._terminal_state_est_label = None
        self._terminal_reference_index = None
        self._capturing_step = True
        try:
            observation, reward, terminated, truncated, _ = self.env.step(action)
        finally:
            self._capturing_step = False

        continuation = self._state(observation)
        next_policy_obs = continuation.policy_obs.clone()
        next_critic_obs = None if continuation.critic_obs is None else continuation.critic_obs.clone()
        next_wm_frame = continuation.wm_frame.clone()
        next_state_est_label = (
            None if continuation.state_est_label is None else continuation.state_est_label.clone()
        )
        next_state_est_label_valid = (
            None
            if continuation.state_est_label_valid is None
            else continuation.state_est_label_valid.clone()
        )
        next_reference_index = (
            None if continuation.reference_index is None else continuation.reference_index.clone()
        )
        if self._terminal_ids is not None:
            if self._terminal_wm_frame is None:
                raise RuntimeError("Terminal environment ids were captured without terminal world-model state.")
            next_wm_frame[self._terminal_ids] = self._terminal_wm_frame
            if self._terminal_policy_obs is not None:
                next_policy_obs[self._terminal_ids] = self._terminal_policy_obs
            if self._terminal_critic_obs is not None:
                if next_critic_obs is None:
                    raise RuntimeError("Terminal critic observations were captured but critic observations are disabled.")
                next_critic_obs[self._terminal_ids] = self._terminal_critic_obs
            if self._terminal_state_est_label is not None:
                if next_state_est_label is None:
                    raise RuntimeError("Terminal estimator labels were captured but continuation labels are disabled.")
                next_state_est_label[self._terminal_ids] = self._terminal_state_est_label
            elif next_state_est_label_valid is not None:
                # Never supervise the estimator with an auto-reset state as if
                # it were the true terminal state.
                next_state_est_label_valid[self._terminal_ids] = False
            if self._terminal_reference_index is not None:
                if next_reference_index is None:
                    raise RuntimeError("Terminal reference indices were captured but continuation indices are disabled.")
                next_reference_index[self._terminal_ids] = self._terminal_reference_index
            # Without an explicit terminal policy callback, reset observations
            # remain here. This is safe for SAC because done masks bootstrap;
            # the true terminal WM frame above is still used for model loss.

        transition_next = PipelineState(
            policy_obs=next_policy_obs,
            wm_frame=next_wm_frame,
            state_est_label=next_state_est_label,
            state_est_label_valid=next_state_est_label_valid,
            critic_obs=next_critic_obs,
            reference_index=next_reference_index,
        )
        return EnvStepResult(
            transition_next=transition_next,
            continuation=continuation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        if getattr(self.env, "_reset_idx", None) is not self._original_reset_idx:
            self.env._reset_idx = self._original_reset_idx

    def __enter__(self) -> "IsaacLabDataCollectionEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _state(self, observation: Any) -> PipelineState:
        label = (
            None
            if self.state_estimator_label_fn is None
            else self.state_estimator_label_fn(self.env).detach().clone()
        )
        return PipelineState(
            policy_obs=self.policy_observation_fn(observation).detach().clone(),
            wm_frame=self.world_model_frame_fn(self.env).detach().clone(),
            state_est_label=label,
            state_est_label_valid=(
                None
                if label is None
                else torch.ones((self.num_envs, 1), device=label.device, dtype=torch.bool)
            ),
            critic_obs=(
                None
                if self.critic_observation_fn is None
                else self.critic_observation_fn(observation).detach().clone()
            ),
            reference_index=(
                None
                if self.reference_index_fn is None
                else self.reference_index_fn(self.env).detach().clone().to(dtype=torch.long)
            ),
        )
