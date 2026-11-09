"""Reconstruct a frozen training env from nominal-recording config differences."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re

import yaml

from frozen_run_config import SavedConfigLoader, configuration_differences, resolved_scene_config


def tokens(path):
    return [int(index) if index else key
            for key, index in re.findall(r"(?:^|\.)([^.\[]+)|\[(\d+)\]", path)]


def replace_at(value, path_tokens, replacement):
    if not path_tokens:
        return copy.deepcopy(replacement)
    head, *tail = path_tokens
    if isinstance(head, int):
        mutable = list(value)
        mutable[head] = replace_at(mutable[head], tail, replacement)
        return tuple(mutable) if isinstance(value, tuple) else mutable
    mutable = copy.deepcopy(value)
    mutable[head] = replace_at(mutable.get(head), tail, replacement)
    return mutable


def read(path):
    with Path(path).open() as stream:
        return yaml.load(stream, Loader=SavedConfigLoader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-env", type=Path, required=True)
    parser.add_argument("--evaluation-overrides", type=Path, required=True)
    parser.add_argument("--saved-agent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    effective, changes = read(args.effective_env), read(args.evaluation_overrides)
    baseline = copy.deepcopy(effective)
    for change in changes:
        path = tokens(change["path"])
        baseline = replace_at(baseline, path, change["baseline"])
    # Training env.yaml is dumped after InteractiveScene construction, where
    # scene-relative prim paths and terrain dimensions have been resolved.
    baseline = resolved_scene_config(baseline)
    # Reapplying every experiment value must reproduce the recorded effective
    # config exactly; this proves the reverse patch has complete path handling.
    replay = copy.deepcopy(baseline)
    for change in changes:
        replay = replace_at(replay, tokens(change["path"]), change["experiment"])
    # InteractiveScene derives terrain.num_envs/env_spacing from the scene at
    # construction.  A recording may later override scene.num_envs without
    # rebuilding terrain, so compare both configs after the same resolution.
    differences = configuration_differences(
        resolved_scene_config(effective), resolved_scene_config(replay), "env")
    if differences:
        raise RuntimeError(f"Recorded baseline reconstruction is incomplete: {differences}")
    args.output.mkdir(parents=True)
    (args.output / "params").mkdir()
    with (args.output / "params/env.yaml").open("w") as stream:
        yaml.dump(baseline, stream, sort_keys=False)
    (args.output / "params/agent.yaml").write_bytes(args.saved_agent.read_bytes())
    print(args.output)


if __name__ == "__main__":
    main()
