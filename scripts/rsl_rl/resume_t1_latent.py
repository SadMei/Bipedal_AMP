"""Resume a frozen T1 latent checkpoint to a TOTAL update cap (at most 40000).

This entry point never changes task defaults or a running process. It restores
the exact saved configuration and every saved learning state into a new run.
"""

from __future__ import annotations

import argparse
import builtins
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from local_source import prefer_local_source_tree


def main():
    root, version = prefer_local_source_tree(__file__)
    original_import = builtins.__import__

    def compatible_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "isaaclab.utils.version":
            version_module = sys.modules.get(name)
            if version_module is not None and not hasattr(version_module, "get_isaac_sim_version"):
                version_module.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
        return module

    builtins.__import__ = compatible_import
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max_iterations", type=int, default=40000, help="Total completed updates, not additional updates.")
    parser.add_argument("--num_envs", type=int, default=None, help="Default: saved count; smaller values are smoke-only.")
    parser.add_argument("--run_name", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    try:
        launcher = AppLauncher(args)
    finally:
        builtins.__import__ = original_import
    try:
        resume(args, root, version)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


def resume(args, root, version):
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from legged_lab.envs import ManagerBasedAmpEnv
    from legged_lab.latent.config import runtime_audit
    from rsl_rl.runners import AMPRunner
    from record_latent import restore_templates
    from frozen_run_config import restore_run_config
    from t1_heading_ablation import load_saved_config
    from t1_latent_ablation import snapshot_sources
    from t1_latent_resume import continuation_counts, check_configuration, restore_training_clock, audit_loaded_runner

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed, remaining = continuation_counts(checkpoint["iter"], args.max_iterations)
    saved_env = load_saved_config(checkpoint_path.parent / "params/env.yaml")
    saved_agent = load_saved_config(checkpoint_path.parent / "params/agent.yaml")
    cfg, agent, robot_name, _, _ = restore_templates(saved_env, saved_agent, version)
    if robot_name != "T1":
        raise ValueError("This continuation is restricted to T1 latent checkpoints.")
    audit = restore_run_config(cfg, agent, checkpoint_path.parent)
    count = saved_env["scene"]["num_envs"] if args.num_envs is None else args.num_envs
    log_dir = root / "logs/rsl_rl" / agent.experiment_name / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    cfg.scene.num_envs = count
    cfg.sim.device = agent.device = args.device
    cfg.log_dir = str(log_dir)
    agent.run_name = args.run_name
    agent.max_iterations = args.max_iterations
    agent.resume = True
    agent.load_run = str(checkpoint_path.parent)
    agent.load_checkpoint = checkpoint_path.name
    options = dict(checkpoint=checkpoint_path, run_name=args.run_name, total_iterations=args.max_iterations,
                   num_envs=count, device=args.device, log_dir=str(log_dir), completed=completed)
    check_configuration(cfg, agent, saved_env, saved_agent, **options, apply_curriculum=False)
    audit.update(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                 completed_updates=completed, additional_updates=remaining, total_update_cap=args.max_iterations,
                 final_checkpoint_iteration=args.max_iterations - 1, num_envs=count,
                 smoke_test=count != saved_env["scene"]["num_envs"],
                 fresh_simulator_and_amp_replay=True, baseline_conditions_preserved=True)
    print(f"[RESUME] {checkpoint_path}: {completed} completed + {remaining} additional = {args.max_iterations} total", flush=True)
    print(f"[RESUME] Log directory: {log_dir}", flush=True)
    dump_yaml(str(log_dir / "params/continuation_audit.yaml"), audit)
    env = ManagerBasedAmpEnv(cfg=cfg)
    try:
        restore_training_clock(env, completed, agent.num_steps_per_env)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        # Wrapper reset runs the curriculum with the restored clock. Verify
        # BOTH the runtime managers and the original configuration contract.
        env.curriculum_manager.compute(env_ids=torch.arange(count, device=env.device))
        params = cfg.curriculum.velocity_range_and_tracking_std.params
        phase = sum(completed >= boundary for boundary in params["phase_boundaries"])
        command = env.command_manager.get_term(params["command_name"])
        ranges = command.cfg.ranges
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            expected = tuple(params[key + "_ranges"][phase])
            if tuple(getattr(ranges, key)) != expected:
                raise RuntimeError(f"Runtime curriculum mismatch: {key}")
            setattr(cfg.commands.base_velocity.ranges, key, expected)
        reward = env.reward_manager.get_term_cfg(params["reward_term_name"])
        if reward.params["std"] != params["tracking_stds"][phase]:
            raise RuntimeError("Runtime tracking std differs from resumed curriculum phase.")
        cfg.rewards.track_lin_vel_xy_exp.params["std"] = reward.params["std"]
        check_configuration(cfg, agent, saved_env, saved_agent, **options, apply_curriculum=True)
        runtime = runtime_audit(env, agent)
        commands = command.command
        runtime.update(curriculum_phase=phase + 1, common_step_counter=env.common_step_counter,
                       sim_step_counter=env._sim_step_counter,
                       command_ranges={key: list(getattr(ranges, key)) for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
                       tracking_std=reward.params["std"],
                       first_command_nonzero_vy_fraction=float((commands[:, 1].abs() > 1e-6).float().mean()),
                       first_command_nonzero_wz_fraction=float((commands[:, 2].abs() > 1e-6).float().mean()))
        runner = AMPRunner(wrapped, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
        runner.load(str(checkpoint_path), load_optimizer=True, map_location=agent.device)
        runtime.update(audit_loaded_runner(runner, checkpoint, completed))
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError("Source checkpoint changed during restoration.")
        runtime["source_sha256"] = snapshot_sources(log_dir)
        for relative in ("scripts/rsl_rl/resume_t1_latent.py", "scripts/rsl_rl/t1_latent_resume.py",
                         "scripts/rsl_rl/launch_t1_latent_resume_40000.sh", "source/legged_lab/test/test_t1_latent_resume.py"):
            target = log_dir / "latent_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        dump_yaml(str(log_dir / "params/continuation_runtime_audit.yaml"), runtime)
        runner.add_git_repo_to_log(__file__)
        print(f"[RESUME] Full configuration and saved-state audit passed: {runtime}", flush=True)
        del checkpoint
        start = time.monotonic()
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
        if runner.current_learning_iteration != args.max_iterations - 1:
            raise RuntimeError("Continuation did not stop at the requested total update count.")
        print(f"[COMPLETE] Total {args.max_iterations} updates; final model_{runner.current_learning_iteration}.pt; "
              f"continuation time={time.monotonic()-start:.1f}s", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
