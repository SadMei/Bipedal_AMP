"""Constrained PPO fine-tuning inside a residual world model.

This module deliberately does not replace the Isaac/RSL-RL PPO runner.  It
operates on an already constructed actor/critic pair and a ``PhysicsInformedWorldModel``
whose observation, reward and termination functions are explicit.  That keeps
the successful deployment PPO checkpoint immutable and makes the model-based
stage opt-in.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Any

import torch
from torch import nn
from tensordict import TensorDict

from .dynamics import PhysicsInformedWorldModel
from .motion_reference import MotionReferenceProvider
from .replay import ReplayBuffer, TransitionBatch


ObservationBuilder = Callable[[torch.Tensor, torch.Tensor | None], Any]
HistoryObservationBuilder = Callable[[torch.Tensor, torch.Tensor | None], Any]


def wbtest_observation_builders(actor_observation_dim: int = 87) -> tuple[ObservationBuilder, ObservationBuilder]:
    """Return builders for the current 312 -> estimator -> 320 wbtest layout.

    The first 87 values are the actor ``policy`` group and the remaining 225
    values are ``state_est_input``.  The critic receives its already-built
    327-value flat group.  The returned callbacks intentionally do not invent
    estimator labels; labels belong to the Isaac/real data collector.
    """

    if actor_observation_dim <= 0 or actor_observation_dim >= 312:
        raise ValueError("actor_observation_dim must be between 1 and 311.")

    def actor_observation(policy_obs: torch.Tensor, _: torch.Tensor | None = None) -> TensorDict:
        if policy_obs.ndim != 2 or policy_obs.shape[-1] != 312:
            raise ValueError(f"wbtest policy observation must be [N, 312], got {tuple(policy_obs.shape)}.")
        return TensorDict(
            {
                "policy": policy_obs[..., :actor_observation_dim],
                "state_est_input": policy_obs[..., actor_observation_dim:],
            },
            batch_size=[policy_obs.shape[0]],
            device=policy_obs.device,
        )

    def critic_observation(critic_obs: torch.Tensor, _: torch.Tensor | None = None) -> TensorDict:
        if critic_obs.ndim != 2 or critic_obs.shape[-1] != 327:
            raise ValueError(f"wbtest critic observation must be [N, 327], got {tuple(critic_obs.shape)}.")
        return TensorDict({"critic": critic_obs}, batch_size=[critic_obs.shape[0]], device=critic_obs.device)

    return actor_observation, critic_observation


@dataclass(frozen=True)
class PPOWorldModelFineTuneCfg:
    """Conservative defaults for the first WM PPO experiment."""

    horizon: int = 20
    update_epochs: int = 2
    minibatch_size: int = 256
    learning_rate: float = 1.0e-5
    clip_param: float = 0.10
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    value_clip_param: float = 5.0
    # Match RSL-RL PPO: the value at the final collected observation is used
    # to bootstrap every non-terminal rollout.
    bootstrap_at_horizon: bool = True
    max_grad_norm: float = 0.5
    reference_kl_coef: float = 0.0
    max_reference_kl: float = 0.01
    max_update_kl: float = 0.02
    enforce_kl_limits: bool = False
    desired_kl: float = 0.01
    adaptive_kl_schedule: bool = True
    actor_trainable: str = "last_layer"
    # Freeze the pretrained estimator as a fixed policy feature extractor.
    # This also freezes both actor-side empirical normalizers and disables the
    # auxiliary estimator loss, so a narrow WM state distribution cannot move
    # the estimator or the input scale seen by the pretrained actor.
    freeze_estimator: bool = False
    def validate(self) -> None:
        if self.horizon != 20:
            raise ValueError("The active PPO-WM baseline requires horizon=20.")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive.")
        if self.learning_rate <= 0.0 or not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.lam <= 1.0:
            raise ValueError("Invalid PPO optimization discount/configuration.")
        if not 0.0 < self.clip_param <= 1.0:
            raise ValueError("clip_param must be in (0, 1].")
        if self.reference_kl_coef < 0.0 or self.max_reference_kl <= 0.0 or self.max_update_kl <= 0.0:
            raise ValueError("Invalid reference KL limits.")
        if self.desired_kl <= 0.0:
            raise ValueError("desired_kl must be positive.")
        if self.value_clip_param <= 0.0:
            raise ValueError("value_clip_param must be positive.")
        if self.actor_trainable not in ("last_layer", "all"):
            raise ValueError("actor_trainable must be 'last_layer' or 'all'.")


@dataclass
class _Rollout:
    observations: list[Any]
    critic_observations: list[Any]
    actions: list[torch.Tensor]
    old_log_probs: list[torch.Tensor]
    old_distribution_params: list[tuple[torch.Tensor, ...]]
    values: list[torch.Tensor]
    rewards: list[torch.Tensor]
    dones: list[torch.Tensor]
    next_critic_observation: Any
    termination_reasons: dict[str, list[torch.Tensor]] = field(default_factory=dict)


class PPOWorldModelFineTuner:
    """A short-horizon, reference-anchored PPO updater for a learned WM.

    ``actor_observation`` and ``critic_observation`` adapt the flat replay
    tensors to the RSL-RL ``TensorDict`` format used by the loaded PPO policy.
    The world model itself must already provide the exact reference clock and
    the full imitation reward; this class intentionally has no fallback reward.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        world_model: PhysicsInformedWorldModel,
        actor_observation: ObservationBuilder,
        critic_observation: ObservationBuilder,
        cfg: PPOWorldModelFineTuneCfg | None = None,
        device: torch.device | str = "cpu",
        reference_provider: MotionReferenceProvider | None = None,
        history_actor_observation: HistoryObservationBuilder | None = None,
        history_critic_observation: HistoryObservationBuilder | None = None,
    ) -> None:
        self.cfg = cfg or PPOWorldModelFineTuneCfg()
        self.cfg.validate()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.world_model = world_model
        self.reference_provider = reference_provider
        self.history_actor_observation = history_actor_observation
        self.history_critic_observation = history_critic_observation
        self.actor_observation = actor_observation
        self.critic_observation = critic_observation
        self.reference_actor = deepcopy(self.actor).to(self.device).eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self._configure_trainable_parameters()
        parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        parameters.extend(parameter for parameter in self.critic.parameters() if parameter.requires_grad)
        if not parameters:
            raise ValueError("No trainable actor/critic parameters remain after freezing.")
        self.optimizer = torch.optim.Adam(parameters, lr=self.cfg.learning_rate)

    def freeze_reference(self) -> None:
        """Refresh the immutable anchor only when the caller explicitly asks."""

        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.eval()

    def state_dict(self) -> dict[str, Any]:
        """Return an isolated fine-tune checkpoint payload."""

        return {
            "format": "wbtest_ppo_world_model_finetune",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "reference_actor": self.reference_actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }

    def save(self, path: str) -> None:
        """Save only to a caller-provided new path; source PPO is untouched."""

        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an interrupted fine-tune without moving its PPO anchor."""

        if state.get("format") != "wbtest_ppo_world_model_finetune":
            raise ValueError(f"Unsupported PPO-WM checkpoint format: {state.get('format')!r}")
        saved_cfg = state.get("cfg")
        if saved_cfg is not None and saved_cfg != self.cfg:
            raise ValueError(f"PPO-WM checkpoint config does not match:\n{saved_cfg}\n!=\n{self.cfg}")
        self.actor.load_state_dict(state["actor"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.reference_actor.load_state_dict(state["reference_actor"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.reference_actor.eval()

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        if self.cfg.actor_trainable == "all":
            for parameter in self.actor.parameters():
                parameter.requires_grad_(True)
        else:
            # Estimator and both observation normalizers remain frozen.  Only
            # the final policy projection is opened for the first experiment.
            linear_layers = [module for module in self.actor.modules() if isinstance(module, nn.Linear)]
            if not linear_layers:
                raise ValueError("Could not find a linear policy output layer.")
            for parameter in linear_layers[-1].parameters():
                parameter.requires_grad_(True)
        if self.cfg.freeze_estimator:
            estimator = getattr(self.actor, "estimator", None)
            if estimator is None:
                raise ValueError(
                    "freeze_estimator=True requires an actor with an estimator module."
                )
            for parameter in estimator.parameters():
                parameter.requires_grad_(False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

    @torch.no_grad()
    def rollout(
        self,
        seeds: ReplayBuffer,
        starts: int | None = None,
        reference_indices: torch.Tensor | None = None,
        horizon: int | None = None,
    ) -> _Rollout:
        """Generate PPO transitions with stochastic actions and mean WM dynamics.

        Seed states come from recorded target/real transitions.  Done samples
        are reseeded from the same buffer so the batch stays rectangular; done
        masks still stop GAE/bootstrap across that boundary.
        """

        if len(seeds) == 0:
            raise ValueError("A non-empty target replay is required for WM rollout.")
        rollout_horizon = self.cfg.horizon if horizon is None else int(horizon)
        if rollout_horizon != 20:
            raise ValueError("The active PPO-WM baseline requires horizon=20.")
        batch_size = min(len(seeds), starts or len(seeds))
        state = seeds.sample(batch_size, self.device)
        history = state.wm_history.clone()
        policy_obs = state.policy_obs.clone()
        critic_obs = state.critic_obs.clone()
        if self.reference_provider is not None:
            if reference_indices is None:
                if state.reference_index is not None:
                    reference_index = state.reference_index.reshape(-1).to(self.reference_provider.device)
                else:
                    reference_index = torch.zeros(batch_size, device=self.reference_provider.device, dtype=torch.long)
            else:
                reference_index = torch.as_tensor(
                    reference_indices, device=self.reference_provider.device, dtype=torch.long
                ).reshape(-1)
                if reference_index.numel() != batch_size:
                    raise ValueError(
                        f"reference_indices must contain {batch_size} values, got {reference_index.numel()}."
                    )
        else:
            reference_index = None
        if self.reference_provider is not None and hasattr(self.reference_provider, "set_alignment"):
            seed_frame = history[:, -1]
            self.reference_provider.set_alignment(
                reference_index,
                torch.cat((seed_frame.new_zeros((batch_size, 2)), seed_frame[:, 63:64]), dim=-1),
                seed_frame[:, 56:60],
            )
        result = _Rollout([], [], [], [], [], [], [], [], None)
        for _ in range(rollout_horizon):
            reference = (
                None
                if self.reference_provider is None
                else self.reference_provider.get(reference_index).to(policy_obs.device)
            )
            actor_obs = (
                self.history_actor_observation(history, reference)
                if self.history_actor_observation is not None
                else self.actor_observation(policy_obs, reference)
            )
            critic_input = (
                self.history_critic_observation(history, reference)
                if self.history_critic_observation is not None
                else self.critic_observation(critic_obs, None)
            )
            action = self.actor(actor_obs, stochastic_output=True).detach()
            log_prob = self.actor.get_output_log_prob(action).detach()
            old_params = tuple(parameter.detach().clone() for parameter in self.actor.output_distribution_params)
            value = self.critic(critic_input).detach()
            next_reference_index = (
                None
                if self.reference_provider is None
                else self.reference_provider.advance(reference_index)
            )
            next_reference = (
                None
                if self.reference_provider is None
                else self.reference_provider.get(next_reference_index).to(policy_obs.device)
            )
            step = self.world_model.step(
                history,
                action,
                reference=reference,
                next_reference=next_reference,
            )
            done = torch.logical_or(step.terminated, step.truncated).reshape(-1, 1)
            termination_reasons = getattr(self.world_model, "last_termination_reasons", {})
            for name, mask in termination_reasons.items():
                result.termination_reasons.setdefault(name, []).append(
                    mask.detach().reshape(-1, 1).bool().clone()
                )
            result.observations.append(actor_obs)
            result.critic_observations.append(critic_input)
            result.actions.append(action)
            result.old_log_probs.append(log_prob)
            result.old_distribution_params.append(old_params)
            result.values.append(value)
            result.rewards.append(step.reward.detach())
            result.dones.append(done)
            alive = ~done.squeeze(-1)
            if torch.any(~alive):
                replacement = seeds.sample(int((~alive).sum()), self.device)
                next_history = step.next_history.clone()
                next_policy = step.next_policy_obs.clone()
                next_critic = step.next_critic_obs.clone()
                next_history[~alive] = replacement.wm_history
                next_policy[~alive] = replacement.policy_obs
                next_critic[~alive] = replacement.critic_obs
                if reference_index is not None:
                    if replacement.reference_index is None:
                        raise RuntimeError("Reference-aware rollout requires reference indices in the replay buffer.")
                    reference_index = next_reference_index.clone()
                    reference_index[~alive] = replacement.reference_index.reshape(-1).to(reference_index.device)
                    if hasattr(self.reference_provider, "set_alignment"):
                        replacement_frame = replacement.wm_history[:, -1]
                        self.reference_provider.set_alignment(
                            replacement.reference_index.reshape(-1),
                            torch.cat(
                                (replacement_frame.new_zeros((replacement_frame.shape[0], 2)),
                                 replacement_frame[:, 63:64]),
                                dim=-1,
                            ),
                            replacement_frame[:, 56:60],
                            env_ids=torch.where(~alive)[0],
                        )
            else:
                next_history, next_policy, next_critic = step.next_history, step.next_policy_obs, step.next_critic_obs
                if reference_index is not None:
                    reference_index = next_reference_index
            # RSL-RL updates empirical actor/critic normalizers with the
            # post-step observation before the next rollout action.
            normalization_reference = (
                None
                if self.reference_provider is None
                else self.reference_provider.get(reference_index).to(next_history.device)
            )
            normalization_actor_obs = (
                self.history_actor_observation(next_history, normalization_reference)
                if self.history_actor_observation is not None
                else self.actor_observation(next_policy, normalization_reference)
            )
            normalization_critic_obs = (
                self.history_critic_observation(next_history, normalization_reference)
                if self.history_critic_observation is not None
                else self.critic_observation(next_critic, None)
            )
            if (
                not self.cfg.freeze_estimator
                and hasattr(self.actor, "update_normalization")
            ):
                self.actor.update_normalization(normalization_actor_obs)
            if hasattr(self.critic, "update_normalization"):
                self.critic.update_normalization(normalization_critic_obs)
            history, policy_obs, critic_obs = next_history, next_policy, next_critic
        final_reference = (
            None
            if self.reference_provider is None
            else self.reference_provider.get(reference_index).to(history.device)
        )
        result.next_critic_observation = (
            self.history_critic_observation(history, final_reference)
            if self.history_critic_observation is not None
            else self.critic_observation(critic_obs, None)
        )
        return result

    def update(self, rollout: _Rollout, minibatch_size: int | None = None) -> dict[str, float]:
        """Apply constrained PPO updates; reject an update that leaves the anchor."""

        if not rollout.rewards:
            raise ValueError("Cannot update from an empty WM rollout.")
        with torch.no_grad():
            bootstrap = (
                self.critic(rollout.next_critic_observation)
                if self.cfg.bootstrap_at_horizon
                else torch.zeros_like(rollout.rewards[-1])
            )
            advantages: list[torch.Tensor] = [torch.zeros_like(rollout.rewards[0]) for _ in rollout.rewards]
            gae = torch.zeros_like(bootstrap)
            for index in reversed(range(len(rollout.rewards))):
                not_done = 1.0 - rollout.dones[index].float()
                next_value = bootstrap if index == len(rollout.rewards) - 1 else rollout.values[index + 1]
                delta = rollout.rewards[index] + self.cfg.gamma * not_done * next_value - rollout.values[index]
                gae = delta + self.cfg.gamma * self.cfg.lam * not_done * gae
                advantages[index] = gae
            advantage = torch.cat(advantages, dim=0)
            old_values = torch.cat(rollout.values, dim=0)
            returns = advantage + old_values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)

        count = advantage.shape[0]
        effective_minibatch_size = self.cfg.minibatch_size if minibatch_size is None else int(minibatch_size)
        if effective_minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        accepted = 0
        total_kl = 0.0
        total_pre_update_kl = 0.0
        max_pre_update_kl = 0.0
        first_pre_update_kl: float | None = None
        total_ref_kl = 0.0
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_estimator_loss = 0.0
        total_estimator_metrics: dict[str, float] = {}
        for _ in range(self.cfg.update_epochs):
            order = torch.randperm(count, device=advantage.device)
            for start in range(0, count, effective_minibatch_size):
                index = order[start : start + effective_minibatch_size]
                obs = _index_nested(rollout.observations, index)
                critic_obs = _index_nested(rollout.critic_observations, index)
                actions = torch.cat(rollout.actions, dim=0)[index]
                old_log_prob = torch.cat(rollout.old_log_probs, dim=0)[index]
                old_params = _index_params(rollout.old_distribution_params, index, count)
                self.actor(obs, stochastic_output=True)
                current_before_params = tuple(
                    parameter.detach() for parameter in self.actor.output_distribution_params
                )
                if self.cfg.adaptive_kl_schedule:
                    with torch.no_grad():
                        schedule_kl = self.actor.get_kl_divergence(
                            old_params, current_before_params
                        ).mean()
                    schedule_kl_value = float(schedule_kl)
                    total_pre_update_kl += schedule_kl_value
                    max_pre_update_kl = max(max_pre_update_kl, schedule_kl_value)
                    if first_pre_update_kl is None:
                        first_pre_update_kl = schedule_kl_value
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    if schedule_kl_value > self.cfg.desired_kl * 2.0:
                        learning_rate = max(1.0e-5, learning_rate / 1.5)
                    elif 0.0 < schedule_kl_value < self.cfg.desired_kl / 2.0:
                        learning_rate = min(1.0e-2, learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = learning_rate
                log_prob = self.actor.get_output_log_prob(actions)
                values = self.critic(critic_obs)
                ratio = torch.exp(log_prob - old_log_prob)
                # RSL-RL stores scalar log-probabilities as [N] while GAE is
                # [N, 1].  Leaving the singleton dimension here broadcasts
                # ratio * advantage to [N, N], corrupting the PPO surrogate
                # (and exhausting memory at the original 57,344 minibatch).
                batch_adv = advantage[index].squeeze(-1)
                if ratio.shape != batch_adv.shape:
                    raise RuntimeError(
                        f"PPO ratio/advantage shape mismatch: {tuple(ratio.shape)} vs {tuple(batch_adv.shape)}"
                    )
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
                policy_loss = -torch.minimum(ratio * batch_adv, clipped * batch_adv).mean()
                old_value = old_values[index]
                value_clipped = old_value + (values - old_value).clamp(
                    -self.cfg.value_clip_param, self.cfg.value_clip_param
                )
                value_loss = torch.maximum(
                    (values - returns[index]).square(),
                    (value_clipped - returns[index]).square(),
                ).mean()
                estimator_loss = torch.zeros((), device=loss_device(values))
                estimator_metrics = {}
                compute_estimator_loss = getattr(self.actor, "compute_estimator_loss", None)
                if (
                    not self.cfg.freeze_estimator
                    and compute_estimator_loss is not None
                    and "state_est_label" in obs.keys()
                ):
                    estimator_loss, estimator_metrics = compute_estimator_loss(obs)
                    loss_estimator_coef = float(getattr(self.actor, "estimator_loss_coef", 1.0))
                else:
                    loss_estimator_coef = 0.0
                ref_obs = _detach_nested(obs)
                with torch.no_grad():
                    self.reference_actor(ref_obs, stochastic_output=True)
                    ref_params = tuple(parameter.detach() for parameter in self.reference_actor.output_distribution_params)
                ref_kl = self.actor.get_kl_divergence(ref_params, self.actor.output_distribution_params).mean()
                # With reference_kl_coef=0 and enforce_kl_limits=False, KL is
                # diagnostic only. PPO ratio clipping remains active.
                loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    + loss_estimator_coef * estimator_loss
                    + self.cfg.reference_kl_coef * ref_kl
                )
                if self.cfg.entropy_coef:
                    loss -= self.cfg.entropy_coef * self.actor.output_entropy.mean()
                actor_before = (
                    {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
                    if self.cfg.enforce_kl_limits
                    else None
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # StateEstPPO clips actor and critic independently. Combining
                # both parameter sets changes the effective update whenever the
                # critic gradient dominates the actor gradient.
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.actor.parameters() if parameter.requires_grad],
                    self.cfg.max_grad_norm,
                )
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    # Re-evaluate both KLs after the optimizer step.  The
                    # pre-step reference KL is only a regularizer; the post-
                    # step value is the actual safety gate.
                    self.actor(obs, stochastic_output=True)
                    current_params = tuple(parameter.detach() for parameter in self.actor.output_distribution_params)
                    update_kl = self.actor.get_kl_divergence(old_params, current_params).mean()
                    post_reference_kl = self.actor.get_kl_divergence(ref_params, current_params).mean()
                total_kl += float(update_kl)
                total_ref_kl += float(post_reference_kl)
                total_loss += float(loss.detach())
                total_policy_loss += float(policy_loss.detach())
                total_value_loss += float(value_loss.detach())
                total_estimator_loss += float(estimator_loss.detach())
                for name, value in estimator_metrics.items():
                    total_estimator_metrics[name] = total_estimator_metrics.get(name, 0.0) + float(value)
                within_limits = (
                    float(update_kl) <= self.cfg.max_update_kl
                    and float(post_reference_kl) <= self.cfg.max_reference_kl
                )
                if not self.cfg.enforce_kl_limits or within_limits:
                    accepted += 1
                else:
                    self.actor.load_state_dict(actor_before)
                    self.optimizer.zero_grad(set_to_none=True)
        batches = max(
            self.cfg.update_epochs
            * ((count + effective_minibatch_size - 1) // effective_minibatch_size),
            1,
        )
        metrics = {
            "ppo_wm/loss": total_loss / batches,
            "ppo_wm/policy_loss": total_policy_loss / batches,
            "ppo_wm/value_loss": total_value_loss / batches,
            "ppo_wm/estimator_loss": total_estimator_loss / batches,
            "ppo_wm/reward_mean": float(torch.cat(rollout.rewards).mean()),
            "ppo_wm/return_mean": float(returns.mean()),
            "ppo_wm/return_abs_max": float(returns.abs().max()),
            "ppo_wm/old_value_mean": float(old_values.mean()),
            "ppo_wm/termination_fraction": float(torch.cat(rollout.dones).float().mean()),
            "ppo_wm/update_kl": total_kl / batches,
            "ppo_wm/pre_update_kl": total_pre_update_kl / batches,
            "ppo_wm/pre_update_kl_max": max_pre_update_kl,
            "ppo_wm/first_pre_update_kl": float(first_pre_update_kl or 0.0),
            "ppo_wm/reference_kl": total_ref_kl / batches,
            "ppo_wm/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "ppo_wm/accepted_batches": float(accepted),
            "ppo_wm/total_batches": float(batches),
            "ppo_wm/rollout_horizon": float(len(rollout.rewards)),
        }
        for name, masks in rollout.termination_reasons.items():
            if masks:
                metrics[f"ppo_wm/termination_{name}"] = float(
                    torch.cat(masks, dim=0).float().mean()
                )
        metrics.update({name: value / batches for name, value in total_estimator_metrics.items()})
        return metrics


def _index_nested(items: list[Any], index: torch.Tensor) -> Any:
    """Index a time-major list of TensorDicts or tensors after flattening it."""

    # The caller uses a concatenated [time * batch] index. TensorDict supports
    # the same leading-batch indexing as tensors, so concatenate explicitly.
    from tensordict import TensorDict

    if not items:
        raise ValueError("Cannot index empty observation list.")
    if isinstance(items[0], TensorDict):
        keys = list(items[0].keys())
        batch_size = items[0].batch_size[0]
        return TensorDict(
            {key: torch.cat([item[key] for item in items], dim=0)[index] for key in keys},
            batch_size=[index.numel()],
            device=index.device,
        )
    return torch.cat(items, dim=0)[index]


def _detach_nested(value: Any) -> Any:
    from tensordict import TensorDict

    if isinstance(value, TensorDict):
        return value.detach()
    return value.detach()


def _index_params(params: list[tuple[torch.Tensor, ...]], index: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, ...]:
    time = len(params)
    if not time or batch_size % time:
        raise ValueError("Invalid rollout distribution parameter shape.")
    envs = batch_size // time
    return tuple(torch.cat([step[part] for step in params], dim=0)[index] for part in range(len(params[0])))


def loss_device(value: torch.Tensor) -> torch.device:
    """Keep scalar auxiliary-loss construction on the critic's device."""

    return value.device
