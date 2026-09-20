"""AMP/v3 port of wbt_est run46/47's full-history, eight-state estimator.

Only the estimator contract is shared: AMP retains its own observations,
critic, discriminator, rewards and plant. Legacy four-state policies live in
``models.py`` and deliberately keep their original checkpoint semantics.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization, MLP

HISTORY_LENGTH = 3
ESTIMATOR_OUTPUT_DIM = 8


def latest_actor_indices(term_dims, history_length=HISTORY_LENGTH) -> list[int]:
    """Isaac flattens each term's history oldest-to-newest, then joins terms."""
    if history_length < 1 or not term_dims or any(width < 1 for width in term_dims):
        raise ValueError("Positive history length and term widths are required.")
    indices, offset = [], 0
    for width in term_dims:
        indices.extend(range(offset + (history_length - 1) * width, offset + history_length * width))
        offset += history_length * width
    return indices


def continuous_mse_contact_bce(raw: torch.Tensor, target: torch.Tensor):
    """Exactly run46/47: a mean over six continuous states plus two-contact BCE."""
    if raw.shape != target.shape or raw.shape[-1] != ESTIMATOR_OUTPUT_DIM:
        raise ValueError(f"Expected matching (..., 8) estimates/labels, got {raw.shape}, {target.shape}.")
    error = raw - target
    continuous = torch.cat((error[..., :4], error[..., 6:8]), dim=-1).square().mean()
    contact = F.binary_cross_entropy_with_logits(raw[..., 4:6], target[..., 4:6])
    metrics = {
        "state_est/base_height": error[..., :1].square().mean().detach(),
        "state_est/base_lin_vel": error[..., 1:4].square().mean().detach(),
        "state_est/foot_height": error[..., 6:8].square().mean().detach(),
        "state_est/contact": (raw[..., 4:6].sigmoid() - target[..., 4:6]).square().mean().detach(),
        "state_est/continuous_mse": continuous.detach(),
        "state_est/contact_bce": contact.detach(),
    }
    return continuous + contact, metrics


class FullHistoryEstimatorActor(nn.Module):
    """Full history -> estimate; current frame + detached estimate -> action."""

    def __init__(self, term_dims, output_dim, hidden_dims, activation,
                 estimator_hidden_dims, estimator_activation, obs_normalization,
                 detach_estimator_for_actor):
        super().__init__()
        self.input_dim = sum(term_dims) * HISTORY_LENGTH
        self.actor_obs_dim = sum(term_dims)
        self.detach_estimator_for_actor = detach_estimator_for_actor
        self.register_buffer("actor_input_indices", torch.tensor(latest_actor_indices(term_dims)), persistent=False)
        self.estimator_input_normalizer = EmpiricalNormalization(self.input_dim)
        self.estimator = MLP(self.input_dim, 8, estimator_hidden_dims, estimator_activation)
        augmented_dim = self.actor_obs_dim + 8
        self.obs_normalizer = EmpiricalNormalization(augmented_dim) if obs_normalization else nn.Identity()
        self.mlp = MLP(augmented_dim, output_dim, hidden_dims, activation)

    def estimate_raw(self, observation: torch.Tensor) -> torch.Tensor:
        return self.estimator(self.estimator_input_normalizer(observation))

    def estimate(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.estimate_raw(observation)
        return torch.cat((raw[..., :4], raw[..., 4:6].sigmoid(), raw[..., 6:8]), dim=-1)

    def augmented_observation(self, observation: torch.Tensor) -> torch.Tensor:
        estimate = self.estimate(observation)
        if self.detach_estimator_for_actor:
            estimate = estimate.detach()
        current = observation.index_select(-1, self.actor_input_indices)
        return torch.cat((current, estimate), dim=-1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(self.augmented_observation(observation)))

    @torch.no_grad()
    def update_normalization(self, observation: torch.Tensor) -> None:
        self.estimator_input_normalizer.update(observation)
        if not isinstance(self.obs_normalizer, nn.Identity):
            self.obs_normalizer.update(self.augmented_observation(observation))


class WbtEstimatorActorCritic(ActorCritic):
    """Keep AMP PPO/discriminator intact; supervise only the estimator branch."""

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], num_actions: int,
                 estimator_term_dims: list[int], estimator_history_length: int = HISTORY_LENGTH,
                 estimator_output_dim: int = ESTIMATOR_OUTPUT_DIM,
                 estimator_hidden_dims: list[int] = [256, 128], estimator_activation: str = "elu",
                 estimator_label_group: str = "state_est_label", estimator_loss_coef: float = 1.0,
                 detach_estimator_for_actor: bool = True,
                 actor_obs_normalization: bool = True, critic_obs_normalization: bool = True,
                 actor_hidden_dims: list[int] = [512, 256, 128],
                 critic_hidden_dims: list[int] = [512, 256, 128], activation: str = "elu",
                 init_noise_std: float = 1.0, noise_std_type: str = "scalar",
                 state_dependent_std: bool = False, **kwargs):
        if estimator_output_dim != 8 or estimator_history_length != HISTORY_LENGTH:
            raise ValueError("The wbtest profile requires 8 states and 3 history frames.")
        if obs_groups["policy"] != ["policy"]:
            raise ValueError("Estimator input must be only the AMP policy group, never privileged groups.")
        if obs["policy"].shape[-1] != sum(estimator_term_dims) * HISTORY_LENGTH:
            raise ValueError("Policy width does not match the estimator term-major three-frame layout.")
        if estimator_label_group not in obs.keys() or obs[estimator_label_group].shape[-1] != 8:
            raise ValueError("A current-frame eight-state estimator label group is required.")
        # Normalization must be inside the composite actor so export includes it.
        super().__init__(obs, obs_groups, num_actions,
                         actor_obs_normalization=False, critic_obs_normalization=critic_obs_normalization,
                         actor_hidden_dims=actor_hidden_dims, critic_hidden_dims=critic_hidden_dims,
                         activation=activation, init_noise_std=init_noise_std, noise_std_type=noise_std_type,
                         state_dependent_std=state_dependent_std, **kwargs)
        output_dim = [2, num_actions] if state_dependent_std else num_actions
        self.actor = FullHistoryEstimatorActor(
            estimator_term_dims, output_dim, actor_hidden_dims, activation,
            estimator_hidden_dims, estimator_activation, actor_obs_normalization, detach_estimator_for_actor,
        )
        if state_dependent_std:
            nn.init.zeros_(self.actor.mlp[-2].weight[num_actions:])
            std_bias = init_noise_std if noise_std_type == "scalar" else float(torch.log(torch.tensor(init_noise_std + 1e-7)))
            nn.init.constant_(self.actor.mlp[-2].bias[num_actions:], std_bias)
        self.estimator_label_group = estimator_label_group
        self.estimator_loss_coef = float(estimator_loss_coef)
        self.estimator_output_dim = estimator_output_dim
        print(f"[ESTIMATOR] wbtest: {self.actor.input_dim} -> {list(estimator_hidden_dims)} -> 8; "
              f"direct current frame {self.actor.actor_obs_dim} + 8; loss=continuous_mse_bce")

    def compute_estimator_loss(self, obs: TensorDict):
        return continuous_mse_contact_bce(self.actor.estimate_raw(self.get_actor_obs(obs)),
                                          obs[self.estimator_label_group])

    def update_normalization(self, obs: TensorDict) -> None:
        self.actor.update_normalization(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    # Inherit strict ActorCritic loading. Do NOT reuse the legacy four-state
    # input-padding migration: history/layout changes are not padding changes.
