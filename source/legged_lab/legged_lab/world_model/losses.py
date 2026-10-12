from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ResidualWorldModel
from .state import model_input


def discounted_rollout_mean(values: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Normalized gamma**n mean over leading prediction-time dimension.

    n=0 denotes the first predicted control step, NOT a physics substep.
    Normalization preserves the scale of a constant per-step objective. Keep
    the original mean operation for gamma=1, including its floating-point
    reduction order. This is a supervised WM loss weight, not an RL discount.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("rollout_loss_gamma must be finite and in (0, 1].")
    if values.ndim < 1 or values.shape[0] == 0 or not values.is_floating_point():
        raise ValueError("Expected nonempty floating-point time-major losses.")
    if gamma == 1.0:
        return values.mean(dim=0)
    weights = values.new_tensor(gamma).pow(
        torch.arange(values.shape[0], device=values.device, dtype=values.dtype)
    )
    weights = weights / weights.sum()
    return (values * weights.reshape(-1, *([1] * (values.ndim-1)))).sum(dim=0)


@dataclass
class EnsembleLoss:
    total: torch.Tensor
    per_member: torch.Tensor
    mse_per_member: torch.Tensor


def one_step_ensemble_loss(
    model: ResidualWorldModel,
    history: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    bootstrap: bool = True,
) -> EnsembleLoss:
    """LIFT-style bootstrapped one-step loss for force or state residuals."""

    value = model_input(model.cfg, history, action)
    if target.shape[:-1] != value.shape[:-1] or target.shape[-1] != model.cfg.output_dim:
        raise ValueError(
            f"target shape {tuple(target.shape)} must match input batch shape {value.shape[:-1]} "
            f"and output dim {model.cfg.output_dim}."
        )
    value = value.reshape(-1, value.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    batch_size = value.shape[0]
    if batch_size == 0:
        raise ValueError("Cannot train a world model on an empty batch.")

    if bootstrap:
        indices = torch.randint(batch_size, (model.cfg.ensemble_size, batch_size), device=value.device)
        ensemble_value = value[indices]
        ensemble_target = target[indices]
    else:
        ensemble_value = value.unsqueeze(0).expand(model.cfg.ensemble_size, -1, -1)
        ensemble_target = target.unsqueeze(0).expand(model.cfg.ensemble_size, -1, -1)

    normalized_target = model.target_normalizer.normalize(ensemble_target)
    prediction = model.forward_normalized(ensemble_value)
    squared_error = (prediction.mean - normalized_target).square()
    mse_per_member = squared_error.mean(dim=(1, 2))
    if model.cfg.probabilistic:
        if prediction.logvar.shape != squared_error.shape:
            raise ValueError(
                "one_step_ensemble_loss requires uncertainty over the predicted "
                f"target itself; generalized-force WMs predict {model.cfg.state.dynamic_dim}-D propagated-state "
                "uncertainty and must use WorldModelTrainer._generalized_force_loss. "
                f"Got logvar {tuple(prediction.logvar.shape)} and error "
                f"{tuple(squared_error.shape)}."
            )
        element_loss = torch.exp(-prediction.logvar) * squared_error + prediction.logvar
        per_member = element_loss.mean(dim=(1, 2))
    else:
        per_member = mse_per_member
    return EnsembleLoss(total=per_member.sum(), per_member=per_member, mse_per_member=mse_per_member)
