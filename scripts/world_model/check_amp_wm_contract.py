"""Build a frozen AMP run and audit its live world-model collection interface."""

from __future__ import annotations

import argparse
import builtins
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace

RSL_RL_SCRIPTS = Path(__file__).resolve().parents[1] / "rsl_rl"
if str(RSL_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPTS))

from local_source import prefer_local_source_tree


DEFAULT_BSRL_R2 = (
    "logs/rsl_rl/bsrl_amp/"
    "2026-09-28_15-29-53_accad34_scale01_deploypd_robust_r2_ankle_transmission_"
    "40000it_8192env_20260928_1531"
)


def main() -> None:
    root, version = prefer_local_source_tree(__file__)
    original_import = builtins.__import__

    def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "isaaclab.utils.version" and not hasattr(
            module, "get_isaac_sim_version"
        ):
            module.get_isaac_sim_version = lambda: SimpleNamespace(
                major=5, minor=1, patch=0
            )
        return module

    builtins.__import__ = compatible_import
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=root / DEFAULT_BSRL_R2)
    parser.add_argument("--num-envs", type=int, default=8)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or not (args.run_dir / "params/env.yaml").is_file():
        parser.error("Require a valid frozen run and positive environment count.")
    try:
        launcher = AppLauncher(args, multi_gpu=False)
    finally:
        builtins.__import__ = original_import
    try:
        audit(args, root, version)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def audit(args, root: Path, version: str) -> None:
    from isaaclab.managers import SceneEntityCfg
    from legged_lab.envs import ManagerBasedAmpEnv
    from legged_lab.world_model import (
        BSRL_12_JOINT_NAMES,
        amp_world_model_frame,
        bsrl_world_model_state_spec,
        validate_amp_world_model_contract,
    )
    from bsrl_dynamics_restore import restore_bsrl_dynamics
    from frozen_run_config import restore_run_config
    from record_latent import restore_templates
    from t1_heading_ablation import load_saved_config

    run = args.run_dir.resolve(strict=True)
    saved_env = load_saved_config(run / "params/env.yaml")
    saved_agent = load_saved_config(run / "params/agent.yaml")
    cfg, agent, robot_name, _, _ = restore_templates(saved_env, saved_agent, version)
    if robot_name != "BSRL":
        raise ValueError("This audit currently requires a frozen BSRL run.")
    restore_bsrl_dynamics(cfg, saved_env)
    restoration = restore_run_config(cfg, agent, run)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.log_dir = None
    feet = ["link_left_ankle_roll", "link_right_ankle_roll"]
    spec = bsrl_world_model_state_spec()
    with tempfile.TemporaryDirectory(prefix="bsrl_wm_contract_usd_") as usd_dir:
        cfg.scene.robot.spawn.usd_dir = usd_dir
        env = ManagerBasedAmpEnv(cfg=cfg)
        try:
            asset_cfg = SceneEntityCfg(
                "robot",
                joint_names=list(BSRL_12_JOINT_NAMES),
                body_names=feet,
                preserve_order=True,
            )
            sensor_cfg = SceneEntityCfg(
                "contact_forces", body_names=feet, preserve_order=True
            )
            asset_cfg.resolve(env.scene)
            sensor_cfg.resolve(env.scene)
            contract = validate_amp_world_model_contract(
                env,
                asset_cfg,
                sensor_cfg,
                spec=spec,
                expected_joint_names=BSRL_12_JOINT_NAMES,
            )
            frame = amp_world_model_frame(
                env, asset_cfg, sensor_cfg, spec=spec
            )
            result = {
                "contract": asdict(contract),
                "frame_shape": list(frame.shape),
                "frame_finite": bool(frame.isfinite().all()),
                "restoration": restoration,
                "run": str(run),
            }
            print("[WM_CONTRACT] " + json.dumps(result, sort_keys=True), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
