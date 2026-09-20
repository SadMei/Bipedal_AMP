"""Opt-in configuration changes after exact restoration of a saved T1 run."""

POLICY_CLASS = "legged_lab.latent.models:LatentActorCritic"
ALGORITHM_CLASS = "legged_lab.latent.algorithm:LatentPPOAMP"
POLICY_OPTIONS = {
    "class_name": POLICY_CLASS,
    "history_length": 10,
    "recent_frames": 2,
    "latent_dim": 20,
    "encoder_dims": [512, 256, 128],
    "decoder_dims": [128, 256],
    "actor_hidden_dims": [2048, 1024, 512, 128],
    "latent_loss_coef": 1.0,
    "latent_kl_beta": 0.01,
}


def enable_latent(env_cfg, agent_cfg):
    if env_cfg.observations.policy.heading_error is not None:
        raise ValueError("Latent Actor must not observe cumulative heading")
    if agent_cfg.policy.class_name != "ActorCritic":
        raise ValueError("Expected an estimator-free T1 baseline")
    env_cfg.observations.policy.history_length = 10
    for key, value in POLICY_OPTIONS.items():
        setattr(agent_cfg.policy, key, value.copy() if isinstance(value, list) else value)
    agent_cfg.algorithm.class_name = ALGORITHM_CLASS
    agent_cfg.algorithm.symmetry_cfg = None
    agent_cfg.algorithm.state_est_loss_required = False


def runtime_audit(env, agent_cfg):
    manager = env.observation_manager
    names = manager.active_terms["policy"]
    expected = ["base_ang_vel", "root_local_rot_tan_norm", "velocity_commands",
                "joint_pos", "joint_vel", "actions"]
    if names != expected:
        raise RuntimeError(f"Unexpected T1 history term order: {names}")
    dims = manager.group_obs_dim
    critic_width = sum(dims[key][0] for key in agent_cfg.obs_groups["critic"])
    if tuple(dims["policy"]) != (870,) or critic_width != 595:
        raise RuntimeError(f"Unexpected Actor/Critic widths: {dims}, Critic={critic_width}")
    if agent_cfg.algorithm.symmetry_cfg is not None or env.cfg.commands.base_velocity.heading_command:
        raise RuntimeError("Mirror and outer heading controller must be disabled")
    return {"policy_terms": names, "raw_actor_width": 870, "frame_width": 87,
            "actor_mlp_input_width": 194, "critic_width": critic_width,
            "history_frames": 10, "policy_dt": env.step_dt,
            "mirror": False, "actor_heading": False, "external_heading_controller": False}
