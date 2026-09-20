"""RSL-RL AMP runner configuration for Men T1."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from legged_lab.rsl_rl import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg
from legged_lab.tasks.locomotion.amp.mdp.symmetry import t1


@configclass
class T1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AMPRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "men_t1_amp"
    clip_actions = None  # No raw-action clipping; PPO's probability-ratio clip is unchanged.
    obs_groups = {
        "policy": ["policy"],
        # Keep new reward privileges last so legacy checkpoint inputs remain a prefix.
        "critic": ["critic", "state_est_critic", "reward_privileged"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    # Keep the deployable baseline estimator-free. ``train.py --use_estimator``
    # replaces this policy through the shared pipeline without another task.
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
        entropy_coef=0.001,
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
        state_est_loss_required=False,
        amp_cfg=RslRlAmpCfg(
            # T1 has fewer and more homogeneous clips than G1. A longer replay
            # window and slower discriminator prevent near-perfect separation
            # before the policy has learned a useful gait.
            disc_obs_buffer_size=1000,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=2.0e-6,
            disc_max_grad_norm=1.0,
            save_best_style_checkpoint=True,
            best_style_warmup_iterations=250,
            best_style_save_interval=100,
            best_style_min_delta=0.01,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                # The 0.5/5.0 ablation let the task reward dominate: the
                # discriminator saturated and the policy drove joints into
                # their limits.  Restore the last configuration that retained
                # a useful AMP signal throughout training.
                task_style_lerp=0.4,
            ),
            loss_type="LSGAN",
        ),
        symmetry_cfg=RslRlSymmetryCfg(
            # Keep the rollout distribution identical to the baseline and add
            # only an equivariance loss on the policy mean action.
            use_data_augmentation=False,
            data_augmentation_func=t1.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
