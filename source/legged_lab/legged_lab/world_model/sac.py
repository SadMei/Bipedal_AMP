from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import log
import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules import EmpiricalNormalization, MLP

from .replay import TransitionBatch
ESTIMATOR_OUTPUT_DIM = 8
BASE_HEIGHT_SLICE = slice(0, 1)
BASE_LIN_VEL_SLICE = slice(1, 4)
FOOT_CONTACT_SLICE = slice(4, 6)
FOOT_HEIGHT_SLICE = slice(6, 8)


def postprocess_estimator_output(estimator_raw: torch.Tensor) -> torch.Tensor:
    """Apply the wbtest 8-D estimator's contact-logit contract."""

    if estimator_raw.shape[-1] != ESTIMATOR_OUTPUT_DIM:
        raise ValueError(
            f"Estimator output must end in {ESTIMATOR_OUTPUT_DIM}, got {tuple(estimator_raw.shape)}."
        )
    return torch.cat(
        (
            estimator_raw[..., :4],
            estimator_raw[..., 4:6].sigmoid(),
            estimator_raw[..., 6:8],
        ),
        dim=-1,
    )


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported SAC activation {name!r}.")


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
) -> nn.Sequential:
    dims = (input_dim, *hidden_dims, output_dim)
    modules: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        modules.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            modules.append(_activation(activation))
    return nn.Sequential(*modules)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        action_center: tuple[float, ...],
        action_scale: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.network = _mlp(obs_dim, 2 * action_dim, hidden_dims, activation)
        self.register_buffer("action_center", torch.tensor(action_center, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        output_layer = self.network[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        # Keep the initial action distribution near one raw action unit even
        # when the affine range is much wider than [-1, 1].
        with torch.no_grad():
            output_layer.bias[action_dim:].copy_((-torch.log(self.action_scale)).clamp(LOG_STD_MIN, LOG_STD_MAX))

    def distribution_parameters(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        unit_action = torch.tanh(pre_tanh)
        action = self.action_center + self.action_scale * unit_action
        if deterministic:
            log_prob = torch.zeros((*observation.shape[:-1], 1), device=observation.device, dtype=observation.dtype)
        else:
            normal_log_prob = -0.5 * (
                ((pre_tanh - mean) / log_std.exp()).square() + 2.0 * log_std + log(2.0 * torch.pi)
            )
            log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1.0 - unit_action.square() + 1.0e-6).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(self.action_scale).sum()
        return action, log_prob

    def random_action(self, count: int) -> torch.Tensor:
        action = torch.empty((count, self.action_dim), device=self.action_scale.device).uniform_(-1.0, 1.0)
        lower = self.action_center - self.action_scale
        upper = self.action_center + self.action_scale
        return torch.maximum(torch.minimum(action, upper), lower)


class TwinQ(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...], activation: str
    ) -> None:
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, 1, hidden_dims, activation)
        self.q2 = _mlp(obs_dim + action_dim, 1, hidden_dims, activation)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat((observation, action), dim=-1)
        return self.q1(value), self.q2(value)


@dataclass(frozen=True)
class SACCfg:
    # Raw deployable estimator observation. WBT uses 312 values; AMP BSRL's
    # term-major three-frame profile uses 144.
    obs_dim: int
    action_dim: int
    critic_obs_dim: int | None = None
    # Optional current-frame projection for the actor. The estimator always
    # receives all obs_dim values. Empty retains WBT's full-observation actor.
    actor_input_indices: tuple[int, ...] = ()
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    estimator_output_dim: int = ESTIMATOR_OUTPUT_DIM
    estimator_hidden_dims: tuple[int, ...] = (256, 128)
    estimator_lr: float = 1.0e-3
    estimator_loss_coef: float = 1.0
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.2
    target_entropy: float | None = None
    reward_scale: float = 1.0
    action_center: tuple[float, ...] = ()
    action_scale: tuple[float, ...] = ()

    def validate(self) -> None:
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("SAC observation and action dimensions must be positive.")
        if any(width <= 0 for width in (*self.actor_hidden_dims, *self.critic_hidden_dims)):
            raise ValueError("SAC hidden dimensions must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive.")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if self.activation not in ("elu", "relu"):
            raise ValueError("activation must be 'elu' or 'relu'.")
        if self.estimator_output_dim not in (0, ESTIMATOR_OUTPUT_DIM):
            raise ValueError(
                f"estimator_output_dim must be 0 or {ESTIMATOR_OUTPUT_DIM}, got {self.estimator_output_dim}."
            )
        if self.actor_input_indices:
            if len(set(self.actor_input_indices)) != len(self.actor_input_indices):
                raise ValueError("actor_input_indices must not contain duplicates.")
            if min(self.actor_input_indices) < 0 or max(self.actor_input_indices) >= self.obs_dim:
                raise ValueError("actor_input_indices contains an out-of-range observation index.")
        if self.estimator_output_dim and self.estimator_loss_coef < 0.0:
            raise ValueError("estimator_loss_coef must be non-negative.")
        if self.action_center and len(self.action_center) != self.action_dim:
            raise ValueError("action_center must be empty or have action_dim entries.")
        if self.action_scale and len(self.action_scale) != self.action_dim:
            raise ValueError("action_scale must be empty or have action_dim entries.")
        if self.action_scale and any(scale <= 0.0 for scale in self.action_scale):
            raise ValueError("Every action_scale entry must be positive.")

    @property
    def actor_base_obs_dim(self) -> int:
        return len(self.actor_input_indices) or self.obs_dim

    @property
    def augmented_obs_dim(self) -> int:
        return self.actor_base_obs_dim + self.estimator_output_dim

    @property
    def resolved_critic_obs_dim(self) -> int:
        return self.augmented_obs_dim if self.critic_obs_dim is None else self.critic_obs_dim

    @property
    def resolved_action_center(self) -> tuple[float, ...]:
        return self.action_center or (0.0,) * self.action_dim

    @property
    def resolved_action_scale(self) -> tuple[float, ...]:
        return self.action_scale or (1.0,) * self.action_dim


class SACAgent:
    """Minimal continuous-action SAC implementation for LIFT-style training."""

    def __init__(self, cfg: SACCfg, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.cfg.validate()
        self.device = torch.device(device)
        if cfg.estimator_output_dim:
            self.estimator_input_normalizer: nn.Module = EmpiricalNormalization(cfg.obs_dim)
            self.estimator: nn.Module = MLP(
                cfg.obs_dim,
                cfg.estimator_output_dim,
                cfg.estimator_hidden_dims,
                "elu",
            )
            self.estimator_input_normalizer.to(self.device)
            self.estimator.to(self.device)
            self.estimator_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                self.estimator.parameters(), lr=cfg.estimator_lr
            )
        else:
            self.estimator_input_normalizer = nn.Identity()
            self.estimator = nn.Identity()
            self.estimator_optimizer = None
        self.policy_input_normalizer = EmpiricalNormalization(cfg.augmented_obs_dim).to(self.device)
        self.critic_input_normalizer = EmpiricalNormalization(cfg.resolved_critic_obs_dim).to(self.device)
        self.actor = SquashedGaussianActor(
            cfg.augmented_obs_dim,
            cfg.action_dim,
            cfg.actor_hidden_dims,
            cfg.activation,
            cfg.resolved_action_center,
            cfg.resolved_action_scale,
        ).to(self.device)
        self.critic = TwinQ(
            cfg.resolved_critic_obs_dim, cfg.action_dim, cfg.critic_hidden_dims, cfg.activation
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(log(cfg.initial_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(-cfg.action_dim if cfg.target_entropy is None else cfg.target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: torch.Tensor, deterministic: bool) -> torch.Tensor:
        features = self.policy_features(observation.to(self.device))
        action, _ = self.actor.sample(features, deterministic=deterministic)
        return action

    @torch.no_grad()
    def random_action(self, count: int) -> torch.Tensor:
        return self.actor.random_action(count)

    def estimate(self, observation: torch.Tensor) -> torch.Tensor:
        if not self.cfg.estimator_output_dim:
            return observation.new_empty((*observation.shape[:-1], 0))
        estimate_raw = self.estimator(self.estimator_input_normalizer(observation))
        return postprocess_estimator_output(estimate_raw)

    def augment_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.cfg.obs_dim:
            raise ValueError(
                f"Raw SAC observation must end in {self.cfg.obs_dim}, got {tuple(observation.shape)}."
            )
        actor_observation = observation
        if self.cfg.actor_input_indices:
            indices = torch.as_tensor(
                self.cfg.actor_input_indices, device=observation.device, dtype=torch.long
            )
            actor_observation = observation.index_select(-1, indices)
        if not self.cfg.estimator_output_dim:
            return actor_observation
        # Same contract as PPO: neither actor nor critic RL gradients train the
        # estimator. It is updated only by valid supervised simulation labels.
        return torch.cat((actor_observation, self.estimate(observation).detach()), dim=-1)

    def policy_features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy_input_normalizer(self.augment_observation(observation))

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        batch = batch.to(self.device)
        estimator_metrics = self._update_estimator(batch)
        augmented_obs = self.augment_observation(batch.policy_obs)
        augmented_next_obs = self.augment_observation(batch.next_policy_obs)
        sim_valid = batch.state_est_label_valid.reshape(-1)
        if sim_valid.any():
            # Keep PPO-compatible policy normalization, but update it only
            # from labelled simulation data. Target/real fine-tuning cannot
            # move the deployed normalization frame.
            self.policy_input_normalizer.update(augmented_obs[sim_valid].detach())
            self.critic_input_normalizer.update(batch.critic_obs[sim_valid].detach())
        policy_obs = self.policy_input_normalizer(augmented_obs)
        next_policy_obs = self.policy_input_normalizer(augmented_next_obs)
        critic_obs = self.critic_input_normalizer(batch.critic_obs)
        next_critic_obs = self.critic_input_normalizer(batch.next_critic_obs)
        terminated = batch.terminated.to(batch.reward.dtype)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_policy_obs)
            target_q1, target_q2 = self.target_critic(next_critic_obs, next_action)
            target_value = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = self.cfg.reward_scale * batch.reward + self.cfg.gamma * (1.0 - terminated) * target_value

        q1, q2 = self.critic(critic_obs, batch.action)
        critic_valid = ~batch.truncated.reshape(-1)
        if not critic_valid.any():
            raise RuntimeError("SAC critic batch contains only truncated transitions.")
        critic_loss = F.mse_loss(q1[critic_valid], target[critic_valid]) + F.mse_loss(
            q2[critic_valid], target[critic_valid]
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        action, log_prob = self.actor.sample(policy_obs)
        q1_pi, q2_pi = self.critic(critic_obs, action)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        # Match LIFT/Brax Eq. 18 exactly. Its tuned alpha learning rates assume
        # the gradient is scaled by the current temperature.
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.cfg.tau)

        return {
            "sac/critic_loss": float(critic_loss.detach()),
            "sac/actor_loss": float(actor_loss.detach()),
            "sac/alpha_loss": float(alpha_loss.detach()),
            "sac/alpha": float(self.alpha.detach()),
            "sac/q_target": float(target.mean()),
            **estimator_metrics,
        }

    def _update_estimator(self, batch: TransitionBatch) -> dict[str, float]:
        if not self.cfg.estimator_output_dim or self.estimator_optimizer is None:
            return {}
        valid = batch.state_est_label_valid.reshape(-1)
        if not valid.any():
            return {"state_est/valid_samples": 0.0}
        observation = batch.policy_obs[valid]
        label = batch.state_est_label[valid]
        normalizer = self.estimator_input_normalizer
        assert isinstance(normalizer, EmpiricalNormalization)
        normalizer.update(observation.detach())
        estimate = self.estimate(observation)
        error = estimate - label
        base_height_loss = torch.mean(torch.square(error[..., BASE_HEIGHT_SLICE]))
        base_lin_vel_loss = torch.mean(torch.square(error[..., BASE_LIN_VEL_SLICE]))
        contact_loss = torch.mean(torch.square(error[..., FOOT_CONTACT_SLICE]))
        foot_height_loss = torch.mean(torch.square(error[..., FOOT_HEIGHT_SLICE]))
        loss = (
            base_height_loss + base_lin_vel_loss + contact_loss + foot_height_loss
        ) * self.cfg.estimator_loss_coef
        self.estimator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.estimator_optimizer.step()
        return {
            "state_est/loss": float(loss.detach()),
            "state_est/base_height": float(base_height_loss.detach()),
            "state_est/base_lin_vel": float(base_lin_vel_loss.detach()),
            "state_est/contact": float(contact_loss.detach()),
            "state_est/foot_height": float(foot_height_loss.detach()),
            "state_est/valid_samples": float(valid.sum()),
        }

    def load_estimator_from_ppo(self, checkpoint: str, strict: bool = True) -> None:
        """Load the exact estimator and input normalizer from an RSL-RL actor checkpoint."""

        if not self.cfg.estimator_output_dim:
            raise RuntimeError("SAC estimator is disabled in SACCfg.")
        checkpoint_state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        actor_state = checkpoint_state.get(
            "actor_state_dict",
            checkpoint_state.get("model_state_dict", checkpoint_state),
        )

        def extract(*prefixes: str) -> dict[str, torch.Tensor]:
            for prefix in prefixes:
                selected = {
                    key.removeprefix(prefix): value
                    for key, value in actor_state.items()
                    if key.startswith(prefix)
                }
                if selected:
                    return selected
            return {}

        # wbtest standalone checkpoints store actor weights directly; AMP's
        # RSL-RL ActorCritic state nests the same modules below ``actor.``.
        normalizer_state = extract(
            "estimator_input_normalizer.", "actor.estimator_input_normalizer."
        )
        estimator_state = extract("estimator.", "actor.estimator.")
        policy_normalizer_state = extract("obs_normalizer.", "actor.obs_normalizer.")
        if not normalizer_state or not estimator_state:
            raise KeyError("Checkpoint does not contain estimator_input_normalizer.* and estimator.* weights.")
        self.estimator_input_normalizer.load_state_dict(normalizer_state, strict=strict)
        self.estimator.load_state_dict(estimator_state, strict=strict)
        if policy_normalizer_state:
            self.policy_input_normalizer.load_state_dict(policy_normalizer_state, strict=strict)

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg,
            "actor": self.actor.state_dict(),
            "estimator_input_normalizer": self.estimator_input_normalizer.state_dict(),
            "estimator": self.estimator.state_dict(),
            "policy_input_normalizer": self.policy_input_normalizer.state_dict(),
            "critic_input_normalizer": self.critic_input_normalizer.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "estimator_optimizer": (
                None if self.estimator_optimizer is None else self.estimator_optimizer.state_dict()
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        # Old pre-fix checkpoints do not contain affine action buffers or an
        # independent critic normalizer. Keep them loadable for diagnostics.
        self.actor.load_state_dict(state["actor"], strict=False)
        self.estimator_input_normalizer.load_state_dict(state["estimator_input_normalizer"])
        self.estimator.load_state_dict(state["estimator"])
        self.policy_input_normalizer.load_state_dict(state["policy_input_normalizer"])
        if "critic_input_normalizer" in state:
            self.critic_input_normalizer.load_state_dict(state["critic_input_normalizer"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        if self.estimator_optimizer is not None and state["estimator_optimizer"] is not None:
            self.estimator_optimizer.load_state_dict(state["estimator_optimizer"])
