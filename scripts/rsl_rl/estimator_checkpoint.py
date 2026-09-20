"""Select legacy/new estimator from saved metadata, not today's defaults."""

from copy import deepcopy
from pathlib import Path

from t1_heading_ablation import load_saved_config

PROFILE_CLASSES = {
    "legged_lab.state_est.models:EstimatorActorCritic": "legacy4d",
    "legged_lab.state_est.wbtest:WbtEstimatorActorCritic": "wbtest",
}


def checkpoint_profile(saved_agent: dict, requested: str = "auto") -> str:
    name = saved_agent.get("policy", {}).get("class_name")
    if name not in PROFILE_CLASSES:
        raise ValueError(f"Not a supported estimator checkpoint: {name!r}.")
    profile = PROFILE_CLASSES[name]
    if requested not in ("auto", profile):
        raise ValueError(f"Checkpoint uses {profile}, not requested {requested}; no implicit estimator migration.")
    return profile


def _restore_observation_group(current, saved):
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.utils.string import string_to_callable
    from frozen_run_config import _restore_value

    # Older runs may have active optional terms whose current template is None.
    for name, value in saved.items():
        if isinstance(value, dict) and "func" in value and getattr(current, name, None) is None:
            setattr(current, name, ObservationTermCfg(func=string_to_callable(value["func"])))
    return _restore_value(current, saved, "estimator.observations")


def restore_estimator_checkpoint(env_cfg, agent_cfg, checkpoint, requested="auto"):
    from legged_lab.state_est.config import enable_state_estimator
    from frozen_run_config import _restore_value

    params = Path(checkpoint).resolve().parent / "params"
    if not (params / "agent.yaml").is_file() or not (params / "env.yaml").is_file():
        if requested == "auto":
            raise ValueError("Estimator auto-detection requires params/agent.yaml and env.yaml; "
                             "supply --estimator_profile only if the observation contract is known.")
        enable_state_estimator(env_cfg, agent_cfg, requested)
        return requested
    saved_agent = load_saved_config(params / "agent.yaml")
    saved_env = load_saved_config(params / "env.yaml")
    profile = checkpoint_profile(saved_agent, requested)
    observations = saved_env["observations"]
    _restore_observation_group(env_cfg.observations.policy, observations["policy"])
    enable_state_estimator(env_cfg, agent_cfg, profile)
    # Restore the exact network-facing groups (including term order/history,
    # label size and normalization), without rewriting rewards or physics here.
    groups = {name for names in saved_agent["obs_groups"].values() for name in names}
    groups.add(saved_agent["policy"].get("estimator_label_group", "state_est_label"))
    for name in groups:
        current = getattr(env_cfg.observations, name)
        _restore_observation_group(current, observations[name])
    agent_cfg.obs_groups = deepcopy(saved_agent["obs_groups"])
    _restore_value(agent_cfg.policy, saved_agent["policy"], "estimator.policy")
    print(f"[ESTIMATOR] Restored checkpoint profile={profile}, saved observations and network settings.")
    return profile
