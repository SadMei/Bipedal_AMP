from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    """Configuration class for the AMP (Adversarial Motion Priors) in the training"""

    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""

    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""

    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""

    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""

    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""

    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""

    save_best_style_checkpoint: bool = False
    """Whether the AMP runner should retain the checkpoint with the best rolling style reward."""

    best_style_warmup_iterations: int = 250
    """Iterations to wait after startup/resume before selecting the best style checkpoint."""

    best_style_save_interval: int = 100
    """Interval, in iterations, at which to evaluate the rolling style reward for saving."""

    best_style_min_delta: float = 0.01
    """Minimum rolling-style improvement required to replace the best checkpoint."""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""

        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN')"""
