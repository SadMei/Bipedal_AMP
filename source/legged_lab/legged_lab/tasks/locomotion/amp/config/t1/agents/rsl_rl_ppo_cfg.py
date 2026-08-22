"""RSL-RL AMP runner configuration for Men T1."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg


@configclass
class T1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "men_t1_amp"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        # 8192 environments produce 196608 transitions per rollout. Splitting
        # those into 16 batches keeps AMP's create-graph gradient penalty below
        # the peak memory of a single 32 GiB GPU without reducing environment count.
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            # T1 has fewer and more homogeneous clips than G1. A longer replay
            # window and slower discriminator prevent near-perfect separation
            # before the policy has learned a useful gait.
            disc_obs_buffer_size=1000,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                # WBT is driven entirely by task/reference rewards. Keep AMP
                # style regularization, but give command tracking the majority
                # of the combined reward so an in-place gait is not optimal.
                task_style_lerp=0.7,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=None,
    )
