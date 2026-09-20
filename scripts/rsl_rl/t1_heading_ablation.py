"""Strict, opt-in T1 actor-heading ablation against a frozen training run.

No task defaults or saved run files are changed. Later smoothness edits are
restored to the baseline values, and any other unrequested difference aborts
before creating an Isaac environment or starting PPO.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml


class SavedConfigLoader(yaml.SafeLoader):
    """Allow only the two inert Python value tags used by Isaac config dumps."""


SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def load_saved_config(path: Path) -> dict:
    with path.open() as stream:
        return yaml.load(stream, Loader=SavedConfigLoader)


def configuration_differences(expected, actual, prefix="") -> list[dict]:
    """Compare semantic config values, including callables serialized by Isaac.

    A missing optional field and an explicit None are equivalent. Tuples and
    lists are equivalent because Hydra/YAML can change their representation.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        changes = []
        for key in sorted(expected.keys() | actual.keys()):
            changes.extend(configuration_differences(
                expected.get(key), actual.get(key), f"{prefix}.{key}" if prefix else key
            ))
        return changes
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) == len(actual):
            changes = []
            for index, (left, right) in enumerate(zip(expected, actual)):
                changes.extend(configuration_differences(left, right, f"{prefix}[{index}]"))
            return changes
    if expected == actual:
        return []
    return [{"path": prefix, "baseline": expected, "experiment": actual}]


def resolved_scene_config(config: dict) -> dict:
    """Apply only InteractiveScene's deterministic pre/post-init substitutions.

    Frozen training YAML is written after scene construction, whereas this
    guard runs before it. These four derived values are not hyperparameters.
    """
    result = copy.deepcopy(config)
    scene = result["scene"]
    for value in scene.values():
        if isinstance(value, dict) and isinstance(value.get("prim_path"), str):
            value["prim_path"] = value["prim_path"].replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    if scene.get("terrain") is not None:
        scene["terrain"]["num_envs"] = scene["num_envs"]
        scene["terrain"]["env_spacing"] = scene["env_spacing"]
    return result


def apply_no_actor_heading_ablation(env_cfg, agent_cfg, baseline_run: str) -> dict:
    baseline = Path(baseline_run).resolve(strict=True)
    env_path = baseline / "params/env.yaml"
    agent_path = baseline / "params/agent.yaml"
    saved_env = load_saved_config(env_path)
    saved_agent = load_saved_config(agent_path)
    if saved_agent["experiment_name"] != "men_t1_amp" or saved_agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Expected an estimator-free T1 AMP baseline.")
    heading = saved_env["observations"]["policy"].get("heading_error")
    if not heading or not heading["func"].endswith(":integrated_heading_error"):
        raise ValueError("The comparison baseline must include integrated actor heading.")
    if agent_cfg.run_name == saved_agent["run_name"]:
        raise ValueError("Supply a distinct --run_name for the no-heading experiment.")

    # These source defaults were changed after the selected formal run started.
    # Restoring them is essential for an observation-only comparison.
    for name in ("action_rate_l2", "dof_acc_l2"):
        getattr(env_cfg.rewards, name).weight = saved_env["rewards"][name]["weight"]
    if saved_env["rewards"].get("action_smoothness") is not None:
        raise ValueError("This ablation expects the historical baseline without action_smoothness.")
    env_cfg.rewards.action_smoothness = None
    env_cfg.observations.policy.heading_error = None

    expected_env = copy.deepcopy(saved_env)
    expected_agent = copy.deepcopy(saved_agent)
    expected_env["observations"]["policy"]["heading_error"] = None
    # Output location/name are metadata, not training hyperparameters.
    expected_env["log_dir"] = env_cfg.log_dir
    expected_agent["run_name"] = agent_cfg.run_name
    differences = configuration_differences(expected_env, resolved_scene_config(env_cfg.to_dict()), "env")
    differences += configuration_differences(expected_agent, agent_cfg.to_dict(), "agent")
    if differences:
        details = "\n".join(
            f"{d['path']}: baseline={d['baseline']!r}; current={d['experiment']!r}" for d in differences
        )
        raise ValueError(f"Unexpected changes beyond actor heading; refusing to train:\n{details}")

    return {
        "baseline_run": str(baseline),
        "baseline_env_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
        "baseline_agent_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        "fresh_training": True,
        "policy_observation_dim": 435,
        "critic_observation_dim": 595,
        "only_training_change": "observations.policy.heading_error: integrated_heading_error -> null",
        "critic_heading_preserved": True,
        "unrequested_configuration_differences": differences,
    }
