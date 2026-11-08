"""Train one R2-based LAFAN/ACCAD style-data ablation from scratch."""
from __future__ import annotations

import argparse
import builtins
from datetime import datetime
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace

from local_source import prefer_local_source_tree
from bsrl_r2_lafan_ablation import VARIANTS


def main():
    root, version = prefer_local_source_tree(__file__)
    original = builtins.__import__

    def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original(name, globals, locals, fromlist, level)
        if name == "isaaclab.utils.version" and not hasattr(module, "get_isaac_sim_version"):
            module.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
        return module

    builtins.__import__ = compatible_import
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=40000)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 40000 or args.num_envs < 7:
        parser.error("Iterations must be 1..40000; at least seven environments required")
    if not (args.baseline / "params/env.yaml").is_file():
        parser.error("R2 baseline config missing")
    try:
        launcher = AppLauncher(args, multi_gpu=False)
    finally:
        builtins.__import__ = original
    try:
        print("[BSRL_R2_LAFAN] Isaac app ready; entering training setup", flush=True)
        train(args, root, version)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def train(args, root, version):
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from legged_lab.envs import ManagerBasedAmpEnv
    from rsl_rl.runners import AMPRunner
    from record_latent import restore_templates
    from bsrl_r2_lafan_ablation import (
        apply_ablation, prepare_dataset, runtime_audit, snapshot_sources, TASK_STYLE_LERP,
    )
    from t1_heading_ablation import load_saved_config

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    dataset_dir, dataset_manifest = prepare_dataset(root, args.variant)
    old_env = load_saved_config(args.baseline / "params/env.yaml")
    old_agent = load_saved_config(args.baseline / "params/agent.yaml")
    cfg, agent, robot_name, _, _ = restore_templates(old_env, old_agent, version)
    if robot_name != "BSRL":
        raise ValueError("BSRL R2 baseline required")
    audit = apply_ablation(
        cfg, agent, args.baseline, args.variant, dataset_dir=dataset_dir,
        dataset_manifest=dataset_manifest, run_name=args.run_name,
        num_envs=args.num_envs, max_iterations=args.max_iterations, device=args.device,
    )
    log_dir = root / "logs/rsl_rl/bsrl_amp" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    cfg.log_dir = str(log_dir)
    cfg.scene.robot.spawn.usd_dir = tempfile.mkdtemp(prefix=f"bsrl_{args.variant}_usd_")
    audit.update(
        pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        usd_cache=cfg.scene.robot.spawn.usd_dir, source_sha256=snapshot_sources(log_dir),
    )
    dump_yaml(str(log_dir / "params/matrix_audit.yaml"), audit)
    print(f"[BSRL_R2_LAFAN] Log directory: {log_dir}", flush=True)
    print(
        f"[BSRL_R2_LAFAN] variant={args.variant}, fresh=True, envs={args.num_envs}, "
        f"iterations={args.max_iterations}, task_style_lerp={TASK_STYLE_LERP[args.variant]}",
        flush=True,
    )
    env = ManagerBasedAmpEnv(cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runtime = runtime_audit(env, agent, args.variant)
        dump_yaml(str(log_dir / "params/matrix_runtime_audit.yaml"), runtime)
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        print(f"[BSRL_R2_LAFAN] Runtime verified: {runtime}", flush=True)
        runner = AMPRunner(wrapped, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
        if hasattr(runner.alg.policy, "estimator"):
            raise RuntimeError("Unexpected estimator module")
        runner.add_git_repo_to_log(__file__)
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
        print(f"[BSRL_R2_LAFAN] Completed {agent.max_iterations} updates: {log_dir}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
