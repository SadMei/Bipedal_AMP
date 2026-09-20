"""Configuration helpers for enabling the estimator on an existing AMP task."""

from __future__ import annotations

import copy

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


def _enable_legacy_estimator(env_cfg, agent_cfg) -> None:
    """Preserve the original four-state estimator and checkpoint contract."""
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
        # Match the wbt_est estimator contract: normalize the estimator input
        # internally, then normalize the augmented actor observation and the
        # privileged critic observation independently.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=list(current.actor_hidden_dims),
        critic_hidden_dims=list(current.critic_hidden_dims),
        activation=current.activation,
    )
    agent_cfg.algorithm.state_est_loss_required = True


@configclass
class RslRlWbtEstimatorActorCriticCfg(RslRlEstimatorActorCriticCfg):
    class_name: str = "legged_lab.state_est.wbtest:WbtEstimatorActorCritic"
    estimator_output_dim: int = 8
    estimator_history_length: int = 3
    estimator_term_dims: list[int] = []


def enable_state_estimator(env_cfg, agent_cfg, profile: str = "wbtest") -> None:
    """Enable latest wbtest behavior for BSRL/T1; legacy4d is replay-compatible.

    Preserve all task dynamics, rewards, commands, critic observations and
    mirror choices. Do not import WBT's motion-tracking policy observations.
    """
    if profile not in ("wbtest", "legacy4d"):
        raise ValueError(f"Unknown estimator profile: {profile}")
    _enable_legacy_estimator(env_cfg, agent_cfg)
    if profile == "legacy4d":
        return

    from .observations import wbtest_estimator_label

    policy = env_cfg.observations.policy
    names = [name for name, term in vars(policy).items() if getattr(term, "func", None) is not None]
    expected = ["base_ang_vel", "root_local_rot_tan_norm", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    joints = policy.joint_pos.params["asset_cfg"].joint_names
    if len(joints) not in (12, 25):
        raise ValueError("wbtest AMP estimator supports the ordered BSRL12 / T1-25 joint layouts.")
    dims = [3, 6, 3, len(joints), len(joints), len(joints)]
    if getattr(policy, "heading_error", None) is not None:
        expected.append("heading_error")
        dims.append(2)
    if names != expected or agent_cfg.obs_groups["policy"] != ["policy"]:
        raise ValueError(f"Unexpected AMP estimator policy layout: {names}; expected {expected}.")
    policy.history_length = 3
    policy.flatten_history_dim = True
    for name in names:
        getattr(policy, name).history_length = 3
        getattr(policy, name).flatten_history_dim = True

    # Reuse each robot's explicitly ordered foot selectors, never a regex
    # whose resolved order could reverse contact/height supervision.
    privileged = env_cfg.observations.reward_privileged
    asset_cfg = copy.deepcopy(privileged.foot_lin_vel_w_xy.params["asset_cfg"])
    sensor_cfg = copy.deepcopy(privileged.foot_contact.params["sensor_cfg"])
    if (not asset_cfg.preserve_order or not sensor_cfg.preserve_order
            or list(asset_cfg.body_names) != list(sensor_cfg.body_names)
            or len(asset_cfg.body_names) != 2):
        raise ValueError("Estimator foot labels need matching, ordered left/right body selectors.")
    label_group = env_cfg.observations.state_est_label
    label_group.history_length = 0
    label_group.enable_corruption = False
    label_group.label.func = wbtest_estimator_label
    label_group.label.params = {"asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg, "force_threshold": 1.0}
    label_group.label.history_length = 0
    label_group.label.noise = None
    current = agent_cfg.policy
    agent_cfg.policy = RslRlWbtEstimatorActorCriticCfg(
        init_noise_std=current.init_noise_std, noise_std_type=current.noise_std_type,
        state_dependent_std=current.state_dependent_std,
        actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=list(current.critic_hidden_dims),
        activation="elu", estimator_term_dims=dims,
    )


def estimator_runtime_audit(env, agent_cfg) -> dict:
    """Verify actual Isaac layout after manager initialization, before learning."""
    manager = env.observation_manager
    policy_cfg = agent_cfg.policy
    dims = list(policy_cfg.estimator_term_dims)
    actual = [tuple(shape) for shape in manager.group_obs_term_dim["policy"]]
    expected = [(width * 3,) for width in dims]
    if actual != expected or tuple(manager.group_obs_dim["state_est_label"]) != (8,):
        raise RuntimeError(f"wbtest estimator observation contract mismatch: {actual}, expected {expected}")
    return {"profile": "wbtest_run46_47", "history_frames": 3, "policy_terms": manager.active_terms["policy"],
            "estimator_input_dim": sum(dims) * 3, "actor_direct_dim": sum(dims),
            "actor_mlp_input_dim": sum(dims) + 8, "label_dim": 8,
            "loss": "continuous_mse_bce", "actor_heading": "heading_error" in manager.active_terms["policy"],
            "symmetry_cfg": agent_cfg.algorithm.symmetry_cfg is not None,
            "critic_observations_changed": False}
