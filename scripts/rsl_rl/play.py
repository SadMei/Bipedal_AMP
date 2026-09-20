# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
import os
import sys
import tempfile
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
parser.add_argument(
    "--restore_saved_t1_config", action="store_true",
    help="Restore and verify the complete saved T1 environment/agent before applying playback overrides.",
)
parser.add_argument(
    "--replay_audit_dir", type=str, default=None,
    help="Save effective replay configs and provenance in a new directory without changing the training run.",
)
parser.add_argument(
    "--torque_audit_dir", type=str, default=None,
    help="Offline evaluation only: log every physics-step actuator torque for a fixed command suite.",
)
parser.add_argument("--torque_audit_seconds", type=float, default=60.0,
                    help="Seconds per command in the optional torque audit.")
parser.add_argument(
    "--export_policy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export ONNX/JIT during playback; disable to preserve an existing selected export.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos")),
    help="Directory used for playback recordings.",
)
parser.add_argument(
    "--video_name",
    type=str,
    default=None,
    help="Optional filename prefix for the playback recording.",
)
parser.add_argument(
    "--follow_camera",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Follow the first robot with the viewer camera (enabled by default for video recording).",
)
parser.add_argument(
    "--command_profile",
    choices=("config", "mixed", "t1_large", "t1_forward_06", "t1_forward_075", "t1_forward_08", "t1_forward_10"),
    default="config",
    help="Use commands from the environment config or a deterministic mixed-command sequence.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug_state",
    action="store_true",
    default=False,
    help="Print compact robot state and action diagnostics during playback.",
)
parser.add_argument(
    "--nominal_eval",
    action="store_true",
    default=False,
    help="Disable training randomization, pushes, observation corruption, and reference-state resets.",
)
parser.add_argument(
    "--use_estimator",
    action="store_true",
    default=False,
    help="Load an estimator checkpoint; automatically select its saved legacy4d or wbtest contract.",
)
parser.add_argument("--estimator_profile", choices=("auto", "wbtest", "legacy4d"), default="auto",
                    help="Prefer auto: restore estimator version and observations from checkpoint metadata.")
parser.add_argument(
    "--heading_observation_mode",
    choices=("config", "zero_lock", "integrated"),
    default="config",
    help="Override the T1 heading-error observation semantics to match the selected checkpoint.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=25,
    help="Playback steps between state diagnostic lines.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.estimator_profile != "auto" and not args_cli.use_estimator:
    parser.error("--estimator_profile requires --use_estimator.")
if args_cli.torque_audit_dir:
    if not args_cli.nominal_eval or args_cli.video or args_cli.command_profile == "config":
        parser.error("Torque audit requires --nominal_eval, a fixed command profile, and no --video.")
    if args_cli.torque_audit_seconds <= 0:
        parser.error("--torque_audit_seconds must be positive.")
    args_cli.export_policy = False
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

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import carb
import yaml
import isaaclab.envs.mdp as il_mdp
import isaaclab.utils.math as math_utils

if not args_cli.headless:
    import omni.appwindow

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401
from legged_lab.rsl_rl import export_policy_as_onnx

# PLACEHOLDER: Extension template (do not remove this comment)

MIXED_COMMANDS = (
    (0.6, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.4, 0.5, 0.0),
    (0.6, 0.0, 0.35),
    (0.3, -0.5, 0.0),
    (0.6, 0.0, -0.35),
)
T1_LARGE_COMMANDS = (
    (1.5, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.5, 0.0, 0.0),
    (1.5, 0.55, 0.0),
    (1.2, 0.0, 1.5),
    (1.2, 0.0, -1.5),
)
T1_FORWARD_06_COMMANDS = ((0.6, 0.0, 0.0),)
T1_FORWARD_075_COMMANDS = ((0.75, 0.0, 0.0),)
T1_FORWARD_08_COMMANDS = ((0.8, 0.0, 0.0),)
T1_FORWARD_10_COMMANDS = ((1.0, 0.0, 0.0),)
MIXED_PHASE_STEPS = 150


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = prepare_agent_cfg_for_local_rsl_rl(agent_cfg, local_rsl_rl_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # This helper is not available in every Isaac Lab release. Import it
        # only for the code path that actually needs a published checkpoint.
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    if args_cli.use_estimator:
        from estimator_checkpoint import restore_estimator_checkpoint

        restore_estimator_checkpoint(env_cfg, agent_cfg, resume_path, args_cli.estimator_profile)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Replay with the robot asset used by the checkpoint.  T1's default asset
    # can change between training runs (for example, 0722 -> 0809); loading an
    # old policy against the current URDF may keep the same tensor dimensions
    # while silently changing joint frames/limits and producing a twisted pose.
    saved_env_cfg_path = os.path.join(log_dir, "params", "env.yaml")
    if os.path.isfile(saved_env_cfg_path):
        with open(saved_env_cfg_path, encoding="utf-8") as saved_env_cfg_file:
            saved_env_cfg = yaml.load(saved_env_cfg_file, Loader=yaml.BaseLoader)
        # Same URDF can mean three different actuator/physics contracts.
        from bsrl_dynamics_restore import restore_bsrl_dynamics
        restore_bsrl_dynamics(env_cfg, saved_env_cfg)
        saved_asset_path = (
            saved_env_cfg.get("scene", {}).get("robot", {}).get("spawn", {}).get("asset_path")
        )
        if saved_asset_path:
            if os.path.isfile(saved_asset_path):
                current_asset_path = env_cfg.scene.robot.spawn.asset_path
                env_cfg.scene.robot.spawn.asset_path = saved_asset_path
                if current_asset_path != saved_asset_path:
                    print(
                        "[INFO] Restored checkpoint robot asset: "
                        f"{current_asset_path} -> {saved_asset_path}"
                    )
            else:
                print(f"[WARN] Checkpoint robot asset does not exist: {saved_asset_path}")

        # PD gains are part of the plant seen during training, not policy
        # weights. Restore them as well so an older checkpoint is not silently
        # evaluated with gains from the current source tree.
        saved_actuators = saved_env_cfg.get("scene", {}).get("robot", {}).get("actuators", {})
        restored_gain_groups = []
        for actuator_name, actuator_cfg in env_cfg.scene.robot.actuators.items():
            saved_actuator_cfg = saved_actuators.get(actuator_name, {})
            restored = False
            for gain_name in ("stiffness", "damping"):
                saved_gains = saved_actuator_cfg.get(gain_name)
                if isinstance(saved_gains, dict):
                    setattr(actuator_cfg, gain_name, {name: float(value) for name, value in saved_gains.items()})
                    restored = True
                elif saved_gains is not None and str(saved_gains).lower() not in ("none", "null"):
                    setattr(actuator_cfg, gain_name, float(saved_gains))
                    restored = True
            if restored:
                restored_gain_groups.append(actuator_name)
        if restored_gain_groups:
            print(f"[INFO] Restored checkpoint PD gains: {', '.join(restored_gain_groups)}")

    replay_audit = None
    if args_cli.restore_saved_t1_config:
        if train_task_name != "LeggedLab-Isaac-AMP-T1-v0":
            raise ValueError("--restore_saved_t1_config is restricted to T1 AMP.")
        from frozen_run_config import restore_run_config

        replay_audit = restore_run_config(env_cfg, agent_cfg, log_dir)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        agent_cfg.device = env_cfg.sim.device
        env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
        print("[REPLAY] Full saved T1 configuration restored and verified before nominal overrides.")

    if args_cli.heading_observation_mode != "config":
        from legged_lab.tasks.locomotion.amp import mdp as amp_mdp
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        heading_term = env_cfg.observations.policy.heading_error
        critic_heading_term = env_cfg.observations.critic.heading_error
        if args_cli.heading_observation_mode == "zero_lock":
            heading_func = amp_mdp.zero_command_heading_error
            heading_params = {"command_name": "base_velocity", "command_threshold": 0.1}
        else:
            heading_func = amp_mdp.integrated_heading_error
            heading_params = {"command_name": "base_velocity"}
        if heading_term is None:
            heading_term = ObsTerm(func=heading_func, params=heading_params.copy())
            env_cfg.observations.policy.heading_error = heading_term
        for term in (heading_term, critic_heading_term):
            term.func = heading_func
            term.params = heading_params.copy()
        print(f"[INFO] Heading observation mode: {args_cli.heading_observation_mode}")

    command_sequences = {
        "mixed": MIXED_COMMANDS,
        "t1_large": T1_LARGE_COMMANDS,
        "t1_forward_06": T1_FORWARD_06_COMMANDS,
        "t1_forward_075": T1_FORWARD_075_COMMANDS,
        "t1_forward_08": T1_FORWARD_08_COMMANDS,
        "t1_forward_10": T1_FORWARD_10_COMMANDS,
    }
    command_sequence = command_sequences.get(args_cli.command_profile, MIXED_COMMANDS)
    if args_cli.nominal_eval:
        # Preserve the selected task's policy/observation structure while making
        # playback deterministic and representative of nominal deployment.
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False
        for event_name in (
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "actuator_gains",
            "body_mass",
            "joint_params",
            "joint_friction",
            "ankle_transmission",
            "base_external_force_torque",
            "push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        if hasattr(env_cfg.events, "reset_from_ref"):
            env_cfg.events.reset_from_ref = None
        if hasattr(env_cfg.events, "reset_to_default"):
            env_cfg.events.reset_to_default = EventTerm(
                func=il_mdp.reset_scene_to_default,
                mode="reset",
                params={"reset_joint_targets": True},
            )
        for actuator_cfg in env_cfg.scene.robot.actuators.values():
            if hasattr(actuator_cfg, "min_delay"):
                actuator_cfg.min_delay = 0
            if hasattr(actuator_cfg, "max_delay"):
                actuator_cfg.max_delay = 0
        if args_cli.video:
            video_duration_s = args_cli.video_length * env_cfg.decimation * env_cfg.sim.dt
            env_cfg.episode_length_s = max(env_cfg.episode_length_s, video_duration_s + 1.0)
        if args_cli.torque_audit_dir:
            env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.torque_audit_seconds + 1.0)

    if args_cli.command_profile != "config":
        # Seed the first command before reset so the initial observation history is consistent.
        command_cfg = env_cfg.commands.base_velocity
        # A fixed playback profile must not be changed again on an automatic
        # episode reset. This only affects the local playback configuration.
        env_cfg.curriculum.velocity_range_and_tracking_std = None
        initial_command = command_sequence[0]
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        command_cfg.ranges.lin_vel_x = (initial_command[0], initial_command[0])
        command_cfg.ranges.lin_vel_y = (initial_command[1], initial_command[1])
        command_cfg.ranges.ang_vel_z = (initial_command[2], initial_command[2])
        command_cfg.ranges.heading = None
        command_cfg.heading_command = False
        command_cfg.rel_heading_envs = 0.0
        command_cfg.rel_standing_envs = 0.0
        # Historical speed-matched T1 commands reference clip-speed caches
        # that fixed, nominal playback does not need. Do not let training
        # mixture/single-axis sampling replace or restrict the requested input.
        from frozen_run_config import disable_training_command_sampling

        disable_training_command_sampling(command_cfg)

    # The converter's default path uses second-resolution time plus Python's
    # seeded RNG. Parallel replays with the same seed can otherwise share an
    # incomplete USD. Use an OS-unique cache without changing the source URDF.
    robot_spawn = env_cfg.scene.robot.spawn
    if hasattr(robot_spawn, "usd_dir") and robot_spawn.usd_dir is None:
        robot_spawn.usd_dir = tempfile.mkdtemp(prefix="amp_play_usd_")
        print(f"[INFO] Isolated playback USD cache: {robot_spawn.usd_dir}")

    if args_cli.replay_audit_dir:
        from pathlib import Path
        import hashlib
        from isaaclab.utils.io import dump_yaml

        audit_dir = Path(args_cli.replay_audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(str(audit_dir / "effective_env.yaml"), env_cfg)
        dump_yaml(str(audit_dir / "effective_agent.yaml"), agent_cfg)
        dump_yaml(str(audit_dir / "manifest.yaml"), {
            "checkpoint": os.path.abspath(resume_path),
            "checkpoint_sha256": hashlib.sha256(Path(resume_path).read_bytes()).hexdigest(),
            "nominal_eval": args_cli.nominal_eval,
            "command_profile": args_cli.command_profile,
            "video_length_steps": args_cli.video_length,
            "video_name": args_cli.video_name,
            "source_restoration": replay_audit,
        })

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Curriculum initialization can overwrite the fixed playback range while the
    # environment is being constructed.  Re-apply it to the live command term.
    if args_cli.command_profile != "config":
        initial_command = command_sequence[0]
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        command_term.cfg.ranges.lin_vel_x = (initial_command[0], initial_command[0])
        command_term.cfg.ranges.lin_vel_y = (initial_command[1], initial_command[1])
        command_term.cfg.ranges.ang_vel_z = (initial_command[2], initial_command[2])
        command_term.command[:] = command_term.command.new_tensor(initial_command)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_name = args_cli.video_name or (
            f"{train_task_name}_{os.path.splitext(os.path.basename(resume_path))[0]}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.expanduser(args_cli.video_folder)),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": video_name,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        print("[INFO] Policy export disabled; existing ONNX/JIT files are unchanged.")

    if args_cli.torque_audit_dir:
        from torque_audit import run_torque_audit

        try:
            run_torque_audit(env, policy, policy_nn, resume_path, args_cli.torque_audit_dir,
                             args_cli.torque_audit_seconds, simulation_app)
        finally:
            env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    sim_env = env.unwrapped
    follow_camera = args_cli.video if args_cli.follow_camera is None else args_cli.follow_camera
    robot = sim_env.scene["robot"] if args_cli.debug_state or follow_camera else None
    camera_focus = robot.data.root_pos_w[0, :2].detach().clone() if follow_camera else None
    command_term = (
        sim_env.command_manager.get_term("base_velocity")
        if args_cli.debug_state or args_cli.command_profile != "config"
        else None
    )
    if args_cli.command_profile != "config":
        # Runner construction resets the environment once more, so this is the
        # final authoritative override used by both the policy and simulation.
        initial_command = command_sequence[0]
        command_term.cfg.ranges.lin_vel_x = (initial_command[0], initial_command[0])
        command_term.cfg.ranges.lin_vel_y = (initial_command[1], initial_command[1])
        command_term.cfg.ranges.ang_vel_z = (initial_command[2], initial_command[2])
        command_term.command[:] = command_term.command.new_tensor(initial_command)
        obs = env.get_observations()
    mixed_phase = 0
    reset_requested = False
    input_interface = None
    keyboard = None
    keyboard_subscription = None
    if not args_cli.headless:
        input_interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()

        def _on_keyboard_event(event, *args):
            del args
            nonlocal reset_requested
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "R":
                reset_requested = True
            return True

        keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
        print("[INFO] Press R in the viewport to reset the robot to its nominal default state.")

    if args_cli.debug_state:
        action_term = sim_env.action_manager.get_term("joint_pos")
        animation_term = sim_env.animation_manager.get_term("animation")
        if "left_ankle_pitch_Link" in robot.body_names:
            debug_foot_names = ["left_ankle_pitch_Link", "right_ankle_pitch_Link"]
        else:
            debug_foot_names = ["link_right_ankle_roll", "link_left_ankle_roll"]
        debug_body_ids, _ = robot.find_bodies(debug_foot_names, preserve_order=True)

        def _debug_state(label: str, dones=None):
            root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            root_quat = robot.data.root_quat_w[0].detach().cpu().tolist()
            joint_ids = action_term._joint_ids
            joint_pos = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
            joint_vel = robot.data.joint_vel[0, joint_ids].detach().cpu().tolist()
            applied_torque = robot.data.applied_torque[0, joint_ids].detach().cpu().tolist()
            base_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().tolist()
            body_pos_w = robot.data.body_pos_w[0, debug_body_ids, :]
            actual_key_body_pos_b = math_utils.quat_apply_inverse(
                robot.data.root_quat_w[0].unsqueeze(0).expand(body_pos_w.shape[0], -1),
                body_pos_w - robot.data.root_pos_w[0].unsqueeze(0),
            )
            reference_key_body_pos_b = animation_term.key_body_pos_b_buffer[0, 0, : len(debug_body_ids)]
            raw_action = action_term.raw_actions[0].detach().cpu().tolist()
            processed_action = action_term.processed_actions[0].detach().cpu().tolist()
            command = command_term.command[0].detach().cpu().tolist()
            done = None if dones is None else bool(dones[0].item())
            print(
                f"[DEBUG] {label} root_pos={[round(v, 4) for v in root_pos]} "
                f"root_quat={[round(v, 4) for v in root_quat]} "
                f"base_h={root_pos[2]:.4f} command={[round(v, 4) for v in command]} "
                f"raw_action={[round(v, 4) for v in raw_action]} "
                f"target={[round(v, 4) for v in processed_action]} "
                f"joint_pos={[round(v, 4) for v in joint_pos]} "
                f"joint_vel={[round(v, 4) for v in joint_vel]} "
                f"torque={[round(v, 4) for v in applied_torque]} "
                f"base_ang_vel_b={[round(v, 4) for v in base_ang_vel_b]} done={done}"
                f" key_err={float(torch.linalg.vector_norm(actual_key_body_pos_b - reference_key_body_pos_b)):.4f}"
            )

        _debug_state("initial")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                timestep = 0
                mixed_phase = 0
                if args_cli.command_profile != "config":
                    initial_command = command_sequence[0]
                    command_term.cfg.ranges.lin_vel_x = (initial_command[0], initial_command[0])
                    command_term.cfg.ranges.lin_vel_y = (initial_command[1], initial_command[1])
                    command_term.cfg.ranges.ang_vel_z = (initial_command[2], initial_command[2])
                obs, _ = env.reset()
                policy_nn.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device))
                if args_cli.command_profile != "config":
                    command_term.command[:] = command_term.command.new_tensor(command_sequence[0])
                if follow_camera:
                    camera_focus.copy_(robot.data.root_pos_w[0, :2])
                print("[INFO] Robot reset to the nominal default state.")
            if follow_camera:
                camera_focus.lerp_(robot.data.root_pos_w[0, :2], 0.15)
                focus_x, focus_y = camera_focus.detach().cpu().tolist()
                sim_env.sim.set_camera_view(
                    eye=[focus_x - 2.0, focus_y + 2.5, 1.5],
                    target=[focus_x + 0.35, focus_y, 0.55],
                )
            # agent stepping
            actions = policy(obs)
            if args_cli.command_profile != "config":
                # Queue the next phase before env.step builds the observation used by the next action.
                next_phase = min((timestep + 1) // MIXED_PHASE_STEPS, len(command_sequence) - 1)
                if next_phase != mixed_phase:
                    mixed_phase = next_phase
                    next_command = command_sequence[mixed_phase]
                    command_term.cfg.ranges.lin_vel_x = (next_command[0], next_command[0])
                    command_term.cfg.ranges.lin_vel_y = (next_command[1], next_command[1])
                    command_term.cfg.ranges.ang_vel_z = (next_command[2], next_command[2])
                    command_term.command[:] = command_term.command.new_tensor(next_command)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1
        if args_cli.debug_state and (
            timestep % max(args_cli.debug_interval, 1) == 0 or bool(dones[0].item())
        ):
            _debug_state(f"step={timestep}", dones)
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if keyboard_subscription is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
