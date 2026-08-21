"""Prefer dependencies pinned inside this Bipedal_AMP checkout."""

from __future__ import annotations

import pathlib
import sys
import tomllib

from packaging import version


def prefer_local_source_tree(entry_file: str) -> tuple[pathlib.Path, str]:
    """Prepend this repository's package and RSL-RL fork to ``sys.path``."""
    repo_root = pathlib.Path(entry_file).resolve().parents[2]
    source_paths = (
        repo_root / "source" / "legged_lab",
        repo_root / "external" / "rsl_rl",
    )
    for source_path in reversed(source_paths):
        path = str(source_path)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    rsl_pyproject = repo_root / "external" / "rsl_rl" / "pyproject.toml"
    if not rsl_pyproject.is_file():
        raise FileNotFoundError(
            f"Missing local AMP RSL-RL fork at {rsl_pyproject.parent}. "
            "Run `git submodule update --init external/rsl_rl`."
        )
    with rsl_pyproject.open("rb") as stream:
        rsl_version = tomllib.load(stream)["project"]["version"]
    return repo_root, rsl_version


def prepare_agent_cfg_for_local_rsl_rl(agent_cfg, rsl_version: str):
    """Remove Isaac Lab fields unsupported by the repository's pre-v4 AMP fork."""
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)
    if version.parse(rsl_version) < version.parse("4.0.0") and hasattr(
        agent_cfg.algorithm, "share_cnn_encoders"
    ):
        del agent_cfg.algorithm.share_cnn_encoders
    return agent_cfg
