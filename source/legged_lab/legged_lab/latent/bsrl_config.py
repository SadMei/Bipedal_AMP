"""BSRL opt-in contract. T1's defaults/checkpoint shapes remain unchanged."""

from .config import POLICY_OPTIONS as T1_POLICY_OPTIONS, ALGORITHM_CLASS

POLICY_OPTIONS = {**T1_POLICY_OPTIONS, "class_name": "legged_lab.latent.models:BSRLLatentActorCritic"}
POLICY_TERMS = ["base_ang_vel", "root_local_rot_tan_norm", "velocity_commands",
                "joint_pos", "joint_vel", "actions"]


def enable_bsrl_latent(env_cfg, agent_cfg):
    for name in ("heading_error", "heading_frame_error"):
        if getattr(env_cfg.observations.policy, name, None) is not None:
            raise ValueError("BSRL latent Actor must not observe cumulative heading")
    if agent_cfg.policy.class_name != "ActorCritic" or agent_cfg.algorithm.symmetry_cfg is not None:
        raise ValueError("Use the estimator-free, no-mirror BSRL baseline")
    env_cfg.observations.policy.history_length = 10
    for key, value in POLICY_OPTIONS.items():
        setattr(agent_cfg.policy, key, value.copy() if isinstance(value, list) else value)
    agent_cfg.algorithm.class_name = ALGORITHM_CLASS
    agent_cfg.algorithm.symmetry_cfg = None
    agent_cfg.algorithm.state_est_loss_required = False


def runtime_audit(env, agent_cfg):
    from legged_lab.assets.bsrl_reduced_ankle import ReducedAnkleArticulation, ReducedAnkleActuator

    manager = env.observation_manager
    if manager.active_terms["policy"] != POLICY_TERMS:
        raise RuntimeError(f"Unexpected BSRL history term order: {manager.active_terms['policy']}")
    dims = manager.group_obs_dim
    critic_width = sum(dims[key][0] for key in agent_cfg.obs_groups["critic"])
    if (tuple(dims["policy"]) != (480,) or critic_width != 370
            or tuple(dims["disc"]) != (4, 39) or tuple(dims["disc_demo"]) != (4, 39)):
        raise RuntimeError(f"Unexpected BSRL observations: {dims}, Critic={critic_width}")
    robot = env.scene["robot"]
    if (not isinstance(robot, ReducedAnkleArticulation) or robot.num_joints != 12
            or set(robot.actuators) != {"large_motors", "small_motors", "parallel_ankles"}
            or not isinstance(robot.actuators["parallel_ankles"], ReducedAnkleActuator)):
        raise RuntimeError("BSRL latent run must use analytical parallel ankles, not explicit loops")
    if agent_cfg.algorithm.symmetry_cfg is not None or env.cfg.commands.base_velocity.heading_command:
        raise RuntimeError("Mirror and outer heading controller must remain disabled")
    return {"policy_terms": POLICY_TERMS, "raw_actor_width": 480, "frame_width": 48,
            "actor_mlp_input_width": 116, "critic_width": critic_width, "num_joints": 12,
            "history_frames": 10, "prediction_width": 33, "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt, "actuators": list(robot.actuators),
            "topology": "analytical_parallel_ankles_no_explicit_loops",
            "mirror": False, "actor_heading": False, "external_heading_controller": False}
