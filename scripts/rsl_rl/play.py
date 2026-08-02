# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import isaaclab.utils.math as math_utils

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import legged_lab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
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
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    sim_env = env.unwrapped
    follow_camera = args_cli.video if args_cli.follow_camera is None else args_cli.follow_camera
    robot = sim_env.scene["robot"] if args_cli.debug_state or follow_camera else None
    camera_focus = robot.data.root_pos_w[0, :2].detach().clone() if follow_camera else None
    if args_cli.debug_state:
        action_term = sim_env.action_manager.get_term("joint_pos")
        command_term = sim_env.command_manager.get_term("base_velocity")
        animation_term = sim_env.animation_manager.get_term("animation")
        debug_body_ids, _ = robot.find_bodies(
            ["link_right_ankle_roll", "link_left_ankle_roll"], preserve_order=True
        )

        def _debug_state(label: str, dones=None):
            root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            root_quat = robot.data.root_quat_w[0].detach().cpu().tolist()
            joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
            joint_vel = robot.data.joint_vel[0].detach().cpu().tolist()
            body_pos_w = robot.data.body_pos_w[0, debug_body_ids, :]
            actual_key_body_pos_b = math_utils.quat_apply_inverse(
                robot.data.root_quat_w[0].unsqueeze(0).expand(body_pos_w.shape[0], -1),
                body_pos_w - robot.data.root_pos_w[0].unsqueeze(0),
            )
            reference_key_body_pos_b = animation_term.key_body_pos_b_buffer[0, 0]
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
                f"joint_vel={[round(v, 4) for v in joint_vel]} done={done}"
                f" key_err={float(torch.linalg.vector_norm(actual_key_body_pos_b - reference_key_body_pos_b)):.4f}"
            )

        _debug_state("initial")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if follow_camera:
                camera_focus.lerp_(robot.data.root_pos_w[0, :2], 0.15)
                focus_x, focus_y = camera_focus.detach().cpu().tolist()
                sim_env.sim.set_camera_view(
                    eye=[focus_x - 2.0, focus_y + 2.5, 1.5],
                    target=[focus_x + 0.35, focus_y, 0.55],
                )
            # agent stepping
            actions = policy(obs)
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
