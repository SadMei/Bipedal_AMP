"""Configuration helpers for enabling the estimator on an existing AMP task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RslRlEstimatorActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "legged_lab.state_est.models:EstimatorActorCritic"
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_activation: str = "elu"
    estimator_output_dim: int = 4
    estimator_label_group: str = "state_est_label"
    estimator_loss_coef: float = 1.0
    detach_estimator_for_actor: bool = True


def enable_state_estimator(env_cfg, agent_cfg) -> None:
    """Enable the T1 estimator without registering a separate task."""
    observations_cfg = env_cfg.observations
    label_cfg_type = getattr(type(observations_cfg), "StateEstLabelCfg", None)
    critic_cfg_type = getattr(type(observations_cfg), "StateEstCriticCfg", None)
    if label_cfg_type is None:
        raise TypeError(f"{type(observations_cfg).__name__} does not define StateEstLabelCfg.")
    if critic_cfg_type is None:
        raise TypeError(f"{type(observations_cfg).__name__} does not define StateEstCriticCfg.")
    if getattr(observations_cfg, "state_est_label", None) is None:
        observations_cfg.state_est_label = label_cfg_type()
    if getattr(observations_cfg, "state_est_critic", None) is None:
        observations_cfg.state_est_critic = critic_cfg_type()

    # Preserve the legacy critic contract unless the estimator is explicitly
    # enabled. The base critic already contains true base linear velocity; this
    # extra group supplies the remaining privileged target: base height.
    critic_obs_groups = list(agent_cfg.obs_groups["critic"])
    if "state_est_critic" not in critic_obs_groups:
        critic_obs_groups.append("state_est_critic")
    agent_cfg.obs_groups["critic"] = critic_obs_groups

    current = agent_cfg.policy
    agent_cfg.policy = RslRlEstimatorActorCriticCfg(
        init_noise_std=current.init_noise_std,
        noise_std_type=current.noise_std_type,
        state_dependent_std=current.state_dependent_std,
        actor_obs_normalization=True,
        critic_obs_normalization=current.critic_obs_normalization,
        actor_hidden_dims=list(current.actor_hidden_dims),
        critic_hidden_dims=list(current.critic_hidden_dims),
        activation=current.activation,
    )
    agent_cfg.algorithm.state_est_loss_required = True
