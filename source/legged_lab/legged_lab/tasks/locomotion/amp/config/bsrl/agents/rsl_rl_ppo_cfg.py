from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import (
    RslRlAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from legged_lab.tasks.locomotion.amp.mdp.symmetry import bsrl
# Estimator comparison switch (keep these lines for later pipeline runs):
# from legged_lab.state_est.config import RslRlEstimatorActorCriticCfg


@configclass
class BSRLRslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "bsrl_amp"
    clip_actions = None  # No raw-action clipping; actuator limits remain active.
    obs_groups = {
        "policy": ["policy"],
        # Base-height and reward privileges belong to the critic independently
        # of whether the deployable actor estimator is enabled.
        "critic": ["critic", "state_est_critic", "reward_privileged"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    policy = RslRlPpoActorCriticCfg(
        # Estimator comparison: replace with RslRlEstimatorActorCriticCfg.
        init_noise_std=1.0,
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
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        state_est_loss_required=False,
        # Estimator comparison: state_est_loss_required=True.
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            data_augmentation_func=bsrl.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
