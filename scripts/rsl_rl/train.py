# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
import os
import sys
from types import SimpleNamespace

# AppLauncher removes and reloads IsaacLab modules while Kit starts. Intercept
# only this version-module import so the mixed 2.3.1/5.1 install can initialize.
_original_import = builtins.__import__


def _import_with_isaaclab_version_compat(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "isaaclab.utils.version":
        version_module = sys.modules.get(name)
        if version_module is not None and not hasattr(version_module, "get_isaac_sim_version"):
            version_module.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
    return module


builtins.__import__ = _import_with_isaaclab_version_compat

from local_source import prefer_local_source_tree, prepare_agent_cfg_for_local_rsl_rl  # isort: skip

repo_root, local_rsl_rl_version = prefer_local_source_tree(__file__)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--video_folder",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos")),
    help="Directory used for training recordings.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--t1_latent_baseline", type=str, default=None,
                    help="Restore a no-heading T1 baseline and enable the no-mirror predictive latent branch.")
parser.add_argument("--t1_clean_amp_baseline", type=str, default=None,
                    help="Frozen T1 latent comparison: uniform commands, no heading terms or command-conditioned AMP.")
parser.add_argument("--bsrl_latent_baseline", type=str, default=None,
                    help="Restore an ACCAD34/no-mirror/reduced-ankle BSRL run and enable predictive latents.")
parser.add_argument(
    "--t1_no_actor_heading_baseline",
    type=str,
    default=None,
    help="Run the no-actor-heading T1 ablation, strictly checked against this saved baseline run.",
)
parser.add_argument(
    "--bsrl_serial_no_mirror_baseline", type=str, default=None,
    help="Restore this saved direct-PD serial BSRL run and disable only its mirror loss.",
)
parser.add_argument(
    "--bsrl_reduced_heading_baseline", type=str, default=None,
    help="Restore this reduced-ankle run; add only heading-frame velocity reward and critic state.",
)
parser.add_argument('--bsrl_accad_baseline', type=str, default=None,
                    help='Frozen reduced-heading baseline for the full ACCAD mirror comparison.')
parser.add_argument('--bsrl_accad_manifest', type=str, default=None)
parser.add_argument('--bsrl_accad_mirror', choices=('off','on'), default=None)
parser.add_argument(
    "--use_estimator",
    action="store_true",
    default=False,
    help="Enable the wbtest-aligned 3-frame / 8-state supervised estimator (legacy checkpoints supported).",
)
parser.add_argument("--estimator_profile", choices=("auto", "wbtest", "legacy4d"), default="auto",
                    help="auto: wbtest for fresh training, saved profile for resumed estimator checkpoints.")
parser.add_argument(
    "--t1_amp_finetune",
    action="store_true",
    default=False,
    help="Keep resumed T1 AMP training at the final command range instead of restarting its curriculum.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.estimator_profile != "auto" and not args_cli.use_estimator:
    parser.error("--estimator_profile requires --use_estimator.")
if args_cli.t1_clean_amp_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-T1-v0" or args_cli.resume or args_cli.use_estimator
    or args_cli.distributed or args_cli.t1_amp_finetune or args_cli.t1_latent_baseline
    or args_cli.bsrl_latent_baseline or args_cli.t1_no_actor_heading_baseline
    or args_cli.bsrl_serial_no_mirror_baseline or args_cli.bsrl_reduced_heading_baseline
    or args_cli.bsrl_accad_baseline or args_cli.bsrl_accad_manifest or args_cli.bsrl_accad_mirror
):
    parser.error("Clean AMP requires fresh single-GPU T1 latent training without other ablation flags.")
if args_cli.bsrl_latent_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-BSRL-v0" or args_cli.resume or args_cli.use_estimator
    or args_cli.distributed or args_cli.t1_amp_finetune or args_cli.t1_no_actor_heading_baseline
    or args_cli.t1_latent_baseline or args_cli.bsrl_serial_no_mirror_baseline or args_cli.bsrl_reduced_heading_baseline
    or args_cli.bsrl_accad_baseline or args_cli.bsrl_accad_manifest or args_cli.bsrl_accad_mirror
    or os.environ.get("BSRL_EXPLICIT_CLOSED_ANKLES", "0") != "0"
):
    parser.error("BSRL latents require fresh single-GPU analytical-parallel training without other ablations.")
if args_cli.t1_latent_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-T1-v0" or args_cli.resume or args_cli.use_estimator
    or args_cli.distributed or args_cli.t1_amp_finetune or args_cli.t1_no_actor_heading_baseline
    or args_cli.bsrl_serial_no_mirror_baseline or args_cli.bsrl_reduced_heading_baseline
    or args_cli.bsrl_accad_baseline or args_cli.bsrl_accad_manifest or args_cli.bsrl_accad_mirror
):
    parser.error("The latent branch requires fresh single-GPU T1 training, without other ablations.")
if any((args_cli.bsrl_accad_baseline,args_cli.bsrl_accad_manifest,args_cli.bsrl_accad_mirror)) and (
    not all((args_cli.bsrl_accad_baseline,args_cli.bsrl_accad_manifest,args_cli.bsrl_accad_mirror))
    or args_cli.task != 'LeggedLab-Isaac-AMP-BSRL-v0' or args_cli.bsrl_reduced_heading_baseline
    or args_cli.bsrl_serial_no_mirror_baseline or args_cli.t1_no_actor_heading_baseline
    or args_cli.resume or args_cli.use_estimator or args_cli.t1_amp_finetune or args_cli.distributed
):
    parser.error('ACCAD comparison needs all three ACCAD arguments and fresh estimator-free single-GPU BSRL.')
if args_cli.bsrl_reduced_heading_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-BSRL-v0" or args_cli.bsrl_serial_no_mirror_baseline
    or args_cli.t1_no_actor_heading_baseline or args_cli.resume or args_cli.use_estimator
    or args_cli.t1_amp_finetune or args_cli.distributed
):
    parser.error("The reduced heading experiment requires fresh, single-GPU, estimator-free BSRL training.")
if args_cli.bsrl_serial_no_mirror_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-BSRL-v0" or args_cli.t1_no_actor_heading_baseline
    or args_cli.resume or args_cli.use_estimator or args_cli.t1_amp_finetune or args_cli.distributed
):
    parser.error("The serial no-mirror ablation requires fresh, single-GPU, estimator-free BSRL training.")
if args_cli.t1_no_actor_heading_baseline and (
    args_cli.task != "LeggedLab-Isaac-AMP-T1-v0"
    or args_cli.resume or args_cli.use_estimator or args_cli.t1_amp_finetune or args_cli.distributed
):
    parser.error("The no-actor-heading ablation requires fresh, single-GPU, estimator-free T1 training.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
finally:
    builtins.__import__ = _original_import

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = prepare_agent_cfg_for_local_rsl_rl(agent_cfg, local_rsl_rl_version)
    if args_cli.use_estimator:
        from legged_lab.state_est.config import enable_state_estimator

        if agent_cfg.resume:
            from estimator_checkpoint import restore_estimator_checkpoint

            estimator_checkpoint = get_checkpoint_path(
                os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)),
                agent_cfg.load_run, agent_cfg.load_checkpoint,
            )
            estimator_profile = restore_estimator_checkpoint(
                env_cfg, agent_cfg, estimator_checkpoint, args_cli.estimator_profile)
        else:
            estimator_profile = "wbtest" if args_cli.estimator_profile == "auto" else args_cli.estimator_profile
            enable_state_estimator(env_cfg, agent_cfg, estimator_profile)
        print(f"[INFO] State estimator enabled: {estimator_profile}.")
    if args_cli.t1_amp_finetune:
        curriculum_term = getattr(env_cfg.curriculum, "velocity_range_and_tracking_std", None)
        if curriculum_term is None:
            raise ValueError("--t1_amp_finetune requires the T1 velocity curriculum.")
        params = curriculum_term.params
        command_ranges = env_cfg.commands.base_velocity.ranges
        command_ranges.lin_vel_x = tuple(params["lin_vel_x_ranges"][-1])
        command_ranges.lin_vel_y = tuple(params["lin_vel_y_ranges"][-1])
        command_ranges.ang_vel_z = tuple(params["ang_vel_z_ranges"][-1])
        env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = params["tracking_stds"][-1]
        env_cfg.curriculum.velocity_range_and_tracking_std = None
        print("[INFO] T1 AMP fine-tune mode: fixed final command range with curriculum disabled.")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    ablation_audit = None
    ablation_audit_name = "heading_ablation_audit.yaml"
    if args_cli.t1_clean_amp_baseline:
        from t1_clean_amp_ablation import apply_clean_amp

        ablation_audit = apply_clean_amp(env_cfg, agent_cfg, args_cli.t1_clean_amp_baseline)
        ablation_audit_name = "clean_amp_ablation_audit.yaml"
        print("[CLEAN_AMP] T1 latent retained; uniform +/-0.8,+/-0.6,+/-1.5; no heading/mirror/conditional AMP.")
    if args_cli.bsrl_latent_baseline:
        from bsrl_latent_ablation import apply_bsrl_latent

        ablation_audit = apply_bsrl_latent(env_cfg, agent_cfg, args_cli.bsrl_latent_baseline)
        ablation_audit_name = "latent_ablation_audit.yaml"
        print("[LATENT] Frozen BSRL ACCAD34: Actor480 -> latent20 + recent96 -> actions12; no mirror; analytical ankles.")
    if args_cli.t1_latent_baseline:
        from t1_latent_ablation import apply_t1_latent

        ablation_audit = apply_t1_latent(env_cfg, agent_cfg, args_cli.t1_latent_baseline)
        ablation_audit_name = "latent_ablation_audit.yaml"
        print("[LATENT] Frozen T1 baseline: Actor870 -> latent20 + recent174 -> actions25; no mirror.")
    if args_cli.bsrl_accad_baseline:
        from bsrl_accad_ablation import apply_accad_ablation
        ablation_audit = apply_accad_ablation(env_cfg,agent_cfg,args_cli.bsrl_accad_baseline,
                                             args_cli.bsrl_accad_manifest,args_cli.bsrl_accad_mirror)
        ablation_audit_name = 'accad_mirror_ablation_audit.yaml'
        print(f'[ACCAD] Frozen analytical parallel ankles; full 34 clips; estimator off; mirror {args_cli.bsrl_accad_mirror}.')
        print('[ACCAD] Existing heading reward/critic retained; Actor240/Critic370; no explicit closed loops.')
    if args_cli.bsrl_reduced_heading_baseline:
        from bsrl_heading_ablation import apply_bsrl_reduced_heading

        ablation_audit = apply_bsrl_reduced_heading(
            env_cfg, agent_cfg, args_cli.bsrl_reduced_heading_baseline
        )
        ablation_audit_name = "reduced_heading_audit.yaml"
        print("[ABLATION] Frozen reduced-ankle BSRL restored; no physical closed loops; mirror 0.1 retained.")
        print("[ABLATION] Added heading-frame velocity reward 1.0/std0.4; Actor240 unchanged, Critic350->370.")
        print("[ABLATION] Uniform commands unchanged; heading reference follows commanded yaw, not measured yaw drift.")
    if args_cli.bsrl_serial_no_mirror_baseline:
        from frozen_run_config import apply_bsrl_serial_no_mirror

        ablation_audit = apply_bsrl_serial_no_mirror(
            env_cfg, agent_cfg, args_cli.bsrl_serial_no_mirror_baseline
        )
        ablation_audit_name = "serial_no_mirror_audit.yaml"
        print("[ABLATION] Saved direct-PD serial BSRL restored exactly; only mirror loss disabled.")
    if args_cli.t1_no_actor_heading_baseline:
        from t1_heading_ablation import apply_no_actor_heading_ablation

        ablation_audit = apply_no_actor_heading_ablation(
            env_cfg, agent_cfg, args_cli.t1_no_actor_heading_baseline
        )
        print("[ABLATION] Saved-baseline comparison passed: only Policy heading is removed (445 -> 435).")
        print("[ABLATION] Critic heading, mirror=0.1, estimator-free PPO/AMP and all other settings preserved.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    if ablation_audit is not None:
        dump_yaml(os.path.join(log_dir, "params", ablation_audit_name), ablation_audit)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.use_estimator and estimator_profile == "wbtest":
        from legged_lab.state_est.config import estimator_runtime_audit

        estimator_audit = estimator_runtime_audit(env.unwrapped, agent_cfg)
        dump_yaml(os.path.join(log_dir, "params", "estimator_runtime_audit.yaml"), estimator_audit)
        print(f"[ESTIMATOR] Runtime observation audit passed: {estimator_audit}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.expanduser(args_cli.video_folder)),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.bsrl_accad_baseline:
        from bsrl_accad_ablation import runtime_check
        runtime_audit = runtime_check(env.unwrapped,agent_cfg,args_cli.bsrl_accad_baseline,
                                      args_cli.bsrl_accad_manifest,args_cli.bsrl_accad_mirror)
        dump_yaml(os.path.join(log_dir,'params','accad_mirror_runtime_audit.yaml'),runtime_audit)
        print('[ACCAD] Post-initialization full configuration / model / observations audit passed.')
    if args_cli.bsrl_reduced_heading_baseline:
        from bsrl_heading_ablation import check_heading_configuration, expected_heading_configuration
        from t1_heading_ablation import load_saved_config
        from pathlib import Path

        baseline = Path(args_cli.bsrl_reduced_heading_baseline)
        expected_env, expected_agent = expected_heading_configuration(
            load_saved_config(baseline / "params/env.yaml"),
            load_saved_config(baseline / "params/agent.yaml"), agent_cfg.run_name)
        check_heading_configuration(expected_env, expected_agent, env_cfg.to_dict(), agent_cfg.to_dict())
        robot = env.unwrapped.scene["robot"]
        dimensions = env.unwrapped.observation_manager.group_obs_dim
        if robot.num_joints != 12 or set(robot.actuators) != {"large_motors", "small_motors", "parallel_ankles"}:
            raise RuntimeError("Heading experiment loaded a plant other than the 12-joint reduced-ankle model")
        if tuple(dimensions["policy"]) != (240,) or tuple(dimensions["critic"]) != (305,):
            raise RuntimeError(f"Heading experiment observation widths differ: {dimensions}")
        runtime_audit = {
            "num_joints": robot.num_joints, "actuators": list(robot.actuators),
            "policy_dim": list(dimensions["policy"]), "critic_base_group_dim": list(dimensions["critic"]),
            "critic_total_dim": sum(dimensions[name][0] for name in agent_cfg.obs_groups["critic"]),
            "physics_dt": env.unwrapped.physics_dt, "policy_dt": env.unwrapped.step_dt,
        }
        dump_yaml(os.path.join(log_dir, "params", "reduced_heading_runtime_audit.yaml"), runtime_audit)
        print("[ABLATION] Post-initialization frozen config audit passed: no unrequested differences.")

    # run training
    if args_cli.bsrl_latent_baseline:
        from legged_lab.latent.bsrl_config import runtime_audit
        from bsrl_latent_ablation import check_configuration, snapshot_sources

        audit = runtime_audit(env.unwrapped, agent_cfg)
        audit["unrequested_configuration_differences"] = check_configuration(
            env_cfg, agent_cfg, args_cli.bsrl_latent_baseline)
        audit["source_sha256"] = snapshot_sources(log_dir)
        dump_yaml(os.path.join(log_dir, "params", "latent_runtime_audit.yaml"), audit)
        print(f"[LATENT] Runtime contract verified: {audit}")
    if args_cli.t1_latent_baseline:
        from legged_lab.latent.config import runtime_audit
        from t1_latent_ablation import check_runtime_configuration, snapshot_sources

        audit = runtime_audit(env.unwrapped, agent_cfg)
        audit["unrequested_configuration_differences"] = check_runtime_configuration(
            env_cfg, agent_cfg, args_cli.t1_latent_baseline)
        audit["source_sha256"] = snapshot_sources(log_dir)
        dump_yaml(os.path.join(log_dir, "params", "latent_runtime_audit.yaml"), audit)
        print(f"[LATENT] Runtime contract verified: {audit}")
    if args_cli.t1_clean_amp_baseline:
        from t1_clean_amp_ablation import runtime_audit, snapshot_sources

        audit = runtime_audit(env.unwrapped, agent_cfg, args_cli.t1_clean_amp_baseline)
        audit["source_sha256"] = snapshot_sources(log_dir)
        dump_yaml(os.path.join(log_dir, "params", "clean_amp_runtime_audit.yaml"), audit)
        print(f"[CLEAN_AMP] Runtime contract verified: {audit}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
