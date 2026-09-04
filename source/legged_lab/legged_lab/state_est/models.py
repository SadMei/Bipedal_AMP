"""Actor-critic with a supervised four-dimensional locomotion estimator."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP


ESTIMATOR_OUTPUT_DIM = 4
BASE_HEIGHT_SLICE = slice(0, 1)
BASE_LIN_VEL_SLICE = slice(1, 4)
# The former 8-D estimator also predicted foot contact (4:6) and foot height
# (6:8). Keep those task-level observation functions available, but do not
# expose their noisy/redundant predictions to AMP actors.


def postprocess_estimator_output(estimator_raw: torch.Tensor) -> torch.Tensor:
    """Map raw network output to policy-facing physical quantities."""
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _check_last_dim(estimator_raw, ESTIMATOR_OUTPUT_DIM, "estimator raw output")
    return estimator_raw


class EstimatorAugmentedActor(nn.Module):
    """Deployable actor that estimates state internally from its raw observation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | list[int],
        hidden_dims: list[int],
        activation: str,
        estimator_hidden_dims: list[int],
        estimator_activation: str,
        estimator_output_dim: int,
        obs_normalization: bool,
        detach_estimator_for_actor: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.estimator_output_dim = estimator_output_dim
        self.detach_estimator_for_actor = detach_estimator_for_actor
        self.estimator_input_normalizer = EmpiricalNormalization(input_dim)
        self.estimator = MLP(input_dim, estimator_output_dim, estimator_hidden_dims, estimator_activation)
        augmented_dim = input_dim + estimator_output_dim
        self.obs_normalizer = EmpiricalNormalization(augmented_dim) if obs_normalization else nn.Identity()
        self.mlp = MLP(augmented_dim, output_dim, hidden_dims, activation)

    def estimate_raw(self, observation: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _check_last_dim(observation, self.input_dim, "estimator input")
        return self.estimator(self.estimator_input_normalizer(observation))

    def estimate(self, observation: torch.Tensor) -> torch.Tensor:
        return postprocess_estimator_output(self.estimate_raw(observation))

    def augmented_observation(self, observation: torch.Tensor) -> torch.Tensor:
        estimate = self.estimate(observation)
        if self.detach_estimator_for_actor:
            estimate = estimate.detach()
        return torch.cat((observation, estimate), dim=-1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        augmented = self.augmented_observation(observation)
        return self.mlp(self.obs_normalizer(augmented))

    @torch.no_grad()
    def update_normalization(self, observation: torch.Tensor) -> None:
        self.estimator_input_normalizer.update(observation)
        if not isinstance(self.obs_normalizer, nn.Identity):
            self.obs_normalizer.update(self.augmented_observation(observation))


class EstimatorActorCritic(ActorCritic):
    """AMP-compatible ActorCritic whose actor consumes an internal 4-D estimate."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        estimator_hidden_dims: list[int] = [256, 128],
        estimator_activation: str = "elu",
        estimator_output_dim: int = ESTIMATOR_OUTPUT_DIM,
        estimator_label_group: str = "state_est_label",
        estimator_loss_coef: float = 1.0,
        detach_estimator_for_actor: bool = True,
        **kwargs,
    ) -> None:
        if estimator_output_dim != ESTIMATOR_OUTPUT_DIM:
            raise ValueError(
                f"Expected estimator_output_dim={ESTIMATOR_OUTPUT_DIM}, got {estimator_output_dim}."
            )
        if estimator_label_group not in obs.keys():
            raise KeyError(
                f"Estimator label group {estimator_label_group!r} is missing; available groups: {sorted(obs.keys())}."
            )
        _check_last_dim(obs[estimator_label_group], estimator_output_dim, "estimator label")

        # The inherited actor normalizer is disabled because normalization is
        # part of the deployable composite actor exported to TorchScript/ONNX.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            **kwargs,
        )

        actor_obs_dim = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        actor_output_dim: int | list[int] = [2, num_actions] if state_dependent_std else num_actions
        self.actor = EstimatorAugmentedActor(
            input_dim=actor_obs_dim,
            output_dim=actor_output_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            estimator_hidden_dims=estimator_hidden_dims,
            estimator_activation=estimator_activation,
            estimator_output_dim=estimator_output_dim,
            obs_normalization=actor_obs_normalization,
            detach_estimator_for_actor=detach_estimator_for_actor,
        )
        if state_dependent_std:
            torch.nn.init.zeros_(self.actor.mlp[-2].weight[num_actions:])
            if noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor.mlp[-2].bias[num_actions:], init_noise_std)
            elif noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor.mlp[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1.0e-7))
                )

        self.estimator_label_group = estimator_label_group
        self.estimator_loss_coef = float(estimator_loss_coef)
        self.estimator_output_dim = estimator_output_dim
        self._critic_group_dims = [obs[group].shape[-1] for group in obs_groups["critic"]]
        self._critic_group_names = list(obs_groups["critic"])
        self._checkpoint_input_migrated = False
        print(f"Estimator MLP: {self.actor.estimator}")
        print(f"Estimator-augmented Actor MLP: {self.actor.mlp}")

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load legacy estimator checkpoints after appending actor observations.

        New observation terms are appended to the main policy/critic groups.
        Their input weights start at zero, so loading a legacy checkpoint keeps
        its policy exactly unchanged until fine-tuning learns to use them.
        """
        state_dict = copy.copy(state_dict)
        current_state = super().state_dict()
        old_actor_dim = state_dict["actor.estimator.0.weight"].shape[1]
        new_actor_dim = self.actor.input_dim
        actor_delta = new_actor_dim - old_actor_dim

        if actor_delta < 0:
            raise RuntimeError(
                f"Cannot load estimator checkpoint with {old_actor_dim} actor observations "
                f"into a model with only {new_actor_dim}."
            )
        if actor_delta > 0:
            self._checkpoint_input_migrated = True

            for suffix in ("_mean", "_var", "_std"):
                key = f"actor.estimator_input_normalizer.{suffix}"
                expanded = current_state[key].clone()
                expanded[..., :old_actor_dim] = state_dict[key]
                state_dict[key] = expanded

            key = "actor.estimator.0.weight"
            expanded = current_state[key].clone()
            expanded[:, :old_actor_dim] = state_dict[key]
            expanded[:, old_actor_dim:] = 0.0
            state_dict[key] = expanded

            # The actor MLP input is [raw observations, estimator output].
            old_augmented_dim = old_actor_dim + self.estimator_output_dim
            new_augmented_dim = new_actor_dim + self.estimator_output_dim
            for key in ("actor.obs_normalizer._mean", "actor.obs_normalizer._var", "actor.obs_normalizer._std"):
                expanded = current_state[key].clone()
                expanded[..., :old_actor_dim] = state_dict[key][..., :old_actor_dim]
                expanded[..., new_actor_dim:new_augmented_dim] = state_dict[key][..., old_actor_dim:old_augmented_dim]
                state_dict[key] = expanded

            key = "actor.mlp.0.weight"
            expanded = current_state[key].clone()
            expanded[:, :old_actor_dim] = state_dict[key][:, :old_actor_dim]
            expanded[:, old_actor_dim:new_actor_dim] = 0.0
            expanded[:, new_actor_dim:new_augmented_dim] = state_dict[key][:, old_actor_dim:old_augmented_dim]
            state_dict[key] = expanded

        key = "critic.0.weight"
        old_critic_dim = state_dict[key].shape[1]
        new_critic_dim = current_state[key].shape[1]
        critic_delta = new_critic_dim - old_critic_dim
        if critic_delta < 0:
            raise RuntimeError(
                f"Cannot load checkpoint with {old_critic_dim} critic observations "
                f"into a model with only {new_critic_dim}."
            )
        if critic_delta > 0:
            self._checkpoint_input_migrated = True
            expanded = current_state[key].clone()
            appended_reward_dim = 0
            if self._critic_group_names[-1:] == ["reward_privileged"]:
                appended_reward_dim = self._critic_group_dims[-1]

            if critic_delta == appended_reward_dim:
                # Reward privileges were deliberately appended after every
                # legacy critic group, so all existing columns remain an exact
                # prefix and the new columns start from zero.
                expanded[:, :old_critic_dim] = state_dict[key]
                expanded[:, old_critic_dim:] = 0.0
            else:
                # Legacy migration path for observation terms appended inside
                # the main critic group while stable privileged groups remain
                # at the end.
                privileged_tail_dim = sum(self._critic_group_dims[1:])
                old_main_dim = old_critic_dim - privileged_tail_dim
                new_main_dim = new_critic_dim - privileged_tail_dim
                expanded[:, :old_main_dim] = state_dict[key][:, :old_main_dim]
                expanded[:, old_main_dim:new_main_dim] = 0.0
                if privileged_tail_dim:
                    expanded[:, new_main_dim:] = state_dict[key][:, old_main_dim:]
            state_dict[key] = expanded

        return super().load_state_dict(state_dict, strict=strict)

    def compute_estimator_loss(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actor_obs = self.get_actor_obs(obs)
        estimate = self.actor.estimate(actor_obs)
        label = obs[self.estimator_label_group]
        _check_last_dim(label, self.estimator_output_dim, "estimator label")
        error = estimate - label
        losses = {
            "state_est/base_height": torch.mean(torch.square(error[..., BASE_HEIGHT_SLICE])),
            "state_est/base_lin_vel": torch.mean(torch.square(error[..., BASE_LIN_VEL_SLICE])),
            # Removed from the AMP estimator contract:
            # "state_est/contact": ... error[..., 4:6]
            # "state_est/foot_height": ... error[..., 6:8]
        }
        return sum(losses.values()), {name: value.detach() for name, value in losses.items()}

    def update_normalization(self, obs: TensorDict) -> None:
        self.actor.update_normalization(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


def _check_last_dim(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} must have last dimension {expected_dim}, got shape {tuple(tensor.shape)}.")
