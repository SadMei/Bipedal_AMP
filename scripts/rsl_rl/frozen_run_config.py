"""Restore a local run's typed Isaac configs and reject incomplete restoration.

Unlike configclass.from_dict, this handles optional dictionaries (e.g. an old
action clip) and replaces parameter maps instead of merging new source keys.
The existing config objects supply their types; saved YAML is safely parsed.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config


def disable_training_command_sampling(command_cfg) -> None:
    """A fixed replay command must not consult training clip-speed support."""
    for name, value in (("motion_speed_animation", None), ("single_axis_commands", False),
                        ("straight_command_probability", 0.0)):
        if hasattr(command_cfg, name):
            setattr(command_cfg, name, value)


def _restore_value(current, saved, path=""):
    if isinstance(saved, dict):
        if isinstance(current, dict) or current is None:
            # SceneEntityCfg values inside manager-term params must remain
            # typed; plain dicts lack resolve() when the manager is built.
            if current is None and "name" in saved and "joint_ids" in saved and "body_ids" in saved:
                from isaaclab.managers import SceneEntityCfg

                current = SceneEntityCfg(saved["name"])
            else:
                previous = current or {}
                return {key: _restore_value(previous.get(key), value, f"{path}.{key}")
                        for key, value in saved.items()}
        if not hasattr(current, "__dict__"):
            raise TypeError(f"Cannot restore mapping into {path}: {type(current).__name__}")
        for key in saved:
            if not hasattr(current, key):
                raise KeyError(f"Saved field has no typed template: {path}.{key}")
        for key in list(vars(current)):
            if not key.startswith("_") and key not in saved:
                setattr(current, key, None)
        for key, value in saved.items():
            setattr(current, key, _restore_value(getattr(current, key), value, f"{path}.{key}"))
        return current
    if callable(current) and isinstance(saved, str):
        from isaaclab.utils.string import string_to_callable

        return string_to_callable(saved)
    return copy.deepcopy(saved)


def restore_run_config(env_cfg, agent_cfg, run_path: str | Path) -> dict:
    """Restore into compatible typed templates and verify every serialized field."""
    run = Path(run_path).resolve(strict=True)
    env_path, agent_path = run / "params/env.yaml", run / "params/agent.yaml"
    saved_env, saved_agent = load_saved_config(env_path), load_saved_config(agent_path)
    _restore_value(env_cfg, saved_env, "env")
    _restore_value(agent_cfg, saved_agent, "agent")
    differences = configuration_differences(saved_env, resolved_scene_config(env_cfg.to_dict()), "env")
    differences += configuration_differences(saved_agent, agent_cfg.to_dict(), "agent")
    if differences:
        raise ValueError(f"Incomplete saved-run restoration: {differences}")
    return {
        "baseline_run": str(run),
        "baseline_env_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
        "baseline_agent_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        "unrequested_configuration_differences": differences,
    }


def apply_bsrl_serial_no_mirror(env_cfg, agent_cfg, baseline_run: str) -> dict:
    """Restore direct-PD serial BSRL and disable only the mirror objective."""
    from legged_lab.assets.bsrl import BSRL_CFG

    baseline = Path(baseline_run).resolve(strict=True)
    saved_env = load_saved_config(baseline / "params/env.yaml")
    saved_agent = load_saved_config(baseline / "params/agent.yaml")
    if saved_agent["experiment_name"] != "bsrl_amp" or saved_agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Expected an estimator-free BSRL baseline.")
    if set(saved_env["scene"]["robot"]["actuators"]) != {"large_motors", "small_motors"}:
        raise ValueError("Expected legacy direct-PD serial ankles, not reduced or explicit parallel ankles.")
    symmetry = saved_agent["algorithm"].get("symmetry_cfg")
    if not symmetry or not symmetry["use_mirror_loss"] or symmetry["use_data_augmentation"]:
        raise ValueError("Expected a mirror-only baseline without data augmentation.")
    if saved_env["observations"]["policy"].get("heading_error") is not None:
        raise ValueError("The serial comparison must not expose actor heading.")
    if saved_env["rewards"].get("zero_yaw_initial_heading_lin_vel_xy_exp") is not None:
        raise ValueError("Do not combine the mirror ablation with a new heading reward.")
    run_name = agent_cfg.run_name
    if not run_name or run_name == saved_agent["run_name"]:
        raise ValueError("A distinct --run_name is required.")
    for key, value, expected in (
        ("num_envs", env_cfg.scene.num_envs, saved_env["scene"]["num_envs"]),
        ("seed", agent_cfg.seed, saved_agent["seed"]),
        ("max_iterations", agent_cfg.max_iterations, saved_agent["max_iterations"]),
    ):
        if value != expected:
            raise ValueError(f"{key} differs from the requested comparison baseline: {value} != {expected}")
    env_cfg.scene.robot = BSRL_CFG.copy()
    audit = restore_run_config(env_cfg, agent_cfg, baseline)
    agent_cfg.run_name = run_name
    # None also avoids the library's logging-only symmetry forward pass, so
    # this run has neither a mirror objective nor a misleading mirror-loss log.
    agent_cfg.algorithm.symmetry_cfg = None
    audit.update({
        "fresh_training": True,
        "topology": "legacy_direct_pd_serial",
        "only_training_change": "algorithm.symmetry_cfg -> null; no mirror loss or data augmentation",
        "actor_heading": False,
        "new_heading_reward": False,
    })
    return audit
