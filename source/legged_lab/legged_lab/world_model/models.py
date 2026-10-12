from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ResidualWorldModelCfg
from .state import model_input


VARIANCE_PARAMETERIZATION = (
    "learned_soft_bounds_normalized_physical_state_estimator_hybrid_v4"
)
FORCE_VARIANCE_PARAMETERIZATION = (
    "learned_soft_bounds_residual_force_diagonal_jacobian_propagation_v1"
)
PAPER_VARIANCE_PARAMETERIZATION = (
    "learned_soft_bounds_lift_privileged_next_state_normalized_diagonal_v2"
)


def variance_parameterization(cfg: ResidualWorldModelCfg) -> str:
    if cfg.uncertainty_mode == "residual_force_diagonal":
        return FORCE_VARIANCE_PARAMETERIZATION
    if cfg.uncertainty_mode == "paper_privileged_diagonal":
        return PAPER_VARIANCE_PARAMETERIZATION
    return VARIANCE_PARAMETERIZATION


class RunningMeanStd(nn.Module):
    """Numerically stable feature normalizer stored with model checkpoints."""

    def __init__(self, size: int, epsilon: float = 1.0e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.detach().reshape(-1, value.shape[-1])
        if value.shape[0] == 0:
            return
        batch_mean = value.mean(dim=0)
        batch_var = value.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(value.shape[0]), device=value.device, dtype=value.dtype)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        new_var = (old_m2 + batch_m2 + delta.square() * self.count * batch_count / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1.0e-12))
        self.count.copy_(total)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) * torch.rsqrt(self.var + 1.0e-6)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sqrt(self.var + 1.0e-6) + self.mean


class EnsembleLinear(nn.Module):
    """Independent linear layer for every ensemble member."""

    def __init__(self, ensemble_size: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))
        bound = 1.0 / sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[0] != self.weight.shape[0]:
            raise ValueError(
                "EnsembleLinear expects [ensemble, batch, features], "
                f"got {tuple(value.shape)}."
            )
        return torch.einsum("ebi,eio->ebo", value, self.weight) + self.bias


@dataclass
class EnsemblePrediction:
    mean: torch.Tensor
    logvar: torch.Tensor


class ResidualWorldModel(nn.Module):
    """Ensemble residual-force MLP with a next-state uncertainty head.

    Dimensions are derived exclusively from ``cfg.state``. For example, T1
    uses an 86-D input and 31-D generalized force while BSRL uses 47-D and
    18-D respectively. The variance is trained by NLL and retained for
    diagnostics, but rollout transitions use the elite force mean.
    """

    def __init__(self, cfg: ResidualWorldModelCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResidualWorldModelCfg()
        self.cfg.validate()

        widths = (self.cfg.model_input_dim, *self.cfg.hidden_dims)
        self.layers = nn.ModuleList(
            EnsembleLinear(self.cfg.ensemble_size, in_dim, out_dim)
            for in_dim, out_dim in zip(widths[:-1], widths[1:])
        )
        self.uncertainty_dim = self.cfg.uncertainty_dim
        head_dim = self.cfg.output_dim + (self.uncertainty_dim if self.cfg.probabilistic else 0)
        self.head = EnsembleLinear(self.cfg.ensemble_size, widths[-1], head_dim)
        if self.cfg.probabilistic:
            # PETS/LIFT-style learned soft bounds.  A hard clamp has exactly
            # zero gradient outside its interval and previously trapped a
            # large fraction of state dimensions at min_logvar during NLL
            # training.  Bounds are shared by ensemble members but learned
            # independently for every predicted state coordinate.
            bound_shape = (1, 1, self.uncertainty_dim)
            self.max_logvar = nn.Parameter(torch.full(bound_shape, self.cfg.max_logvar))
            self.min_logvar = nn.Parameter(torch.full(bound_shape, self.cfg.min_logvar))
        self.activation = nn.SiLU()
        self.input_normalizer = RunningMeanStd(self.cfg.model_input_dim)
        self.target_normalizer = RunningMeanStd(self.cfg.output_dim)
        if self.cfg.uncertainty_mode in (
            "next_state_diagonal",
            "paper_privileged_diagonal",
        ):
            # Evaluate diagonal next-state Gaussians in per-feature normalized
            # coordinates. The mean still comes from physical dynamics; this
            # normalizer prevents high-unit velocity errors from being hidden
            # by variance while gravity support remains underfit.
            self.state_normalizer = RunningMeanStd(self.cfg.uncertainty_dim)
        if self.cfg.uncertainty_mode == "next_state_diagonal":
            # These six policy-facing continuous values remain deterministic
            # targets, but their MSE is evaluated in replay-normalized
            # coordinates so metre-scale height errors are not drowned out by
            # rad/s joint-velocity terms. Contact retains its separate BCE.
            self.estimator_continuous_normalizer = RunningMeanStd(
                4 + self.cfg.state.num_feet
            )
        self.register_buffer("elite_indices", torch.arange(self.cfg.num_elites, dtype=torch.long))

    @torch.no_grad()
    def update_normalizers(self, history: torch.Tensor, action: torch.Tensor, target: torch.Tensor) -> None:
        value = model_input(self.cfg, history, action)
        if target.shape[-1] != self.cfg.output_dim:
            raise ValueError(f"target must end in {self.cfg.output_dim}, got {tuple(target.shape)}.")
        self.input_normalizer.update(value)
        self.target_normalizer.update(target)

    def forward_normalized(self, value: torch.Tensor) -> EnsemblePrediction:
        """Predict normalized targets from an already assembled model input."""

        if value.shape[-1] != self.cfg.model_input_dim:
            raise ValueError(
                f"model input must end in {self.cfg.model_input_dim}, got {tuple(value.shape)}."
            )
        value = self.input_normalizer.normalize(value)
        if value.ndim == 2:
            value = value.unsqueeze(0).expand(self.cfg.ensemble_size, -1, -1)
        elif value.ndim != 3 or value.shape[0] != self.cfg.ensemble_size:
            raise ValueError("model input must have shape [batch, dim] or [ensemble, batch, dim].")

        for layer in self.layers:
            value = self.activation(layer(value))
        output = self.head(value)
        if self.cfg.probabilistic:
            mean, raw_logvar = torch.split(output, (self.cfg.output_dim, self.uncertainty_dim), dim=-1)
            logvar = self.max_logvar - torch.nn.functional.softplus(
                self.max_logvar - raw_logvar
            )
            logvar = self.min_logvar + torch.nn.functional.softplus(
                logvar - self.min_logvar
            )
        else:
            mean = output[..., : self.cfg.output_dim]
            logvar = torch.zeros((*mean.shape[:-1], self.uncertainty_dim), device=mean.device, dtype=mean.dtype)
        return EnsemblePrediction(mean=mean, logvar=logvar)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> EnsemblePrediction:
        value = model_input(self.cfg, history, action)
        prediction = self.forward_normalized(value)
        mean = self.target_normalizer.denormalize(prediction.mean)
        if self.cfg.prediction_type == "state_delta":
            variance_scale = self.target_normalizer.var + 1.0e-6
            logvar = prediction.logvar + torch.log(variance_scale)
        else:
            # The explicit config says whether this physical-unit logvar is a
            # next-state diagonal or a 31-D residual-force diagonal.
            logvar = prediction.logvar
            if self.cfg.uncertainty_mode in (
                "next_state_diagonal",
                "paper_privileged_diagonal",
            ):
                # forward_normalized() is used by the training objective;
                # public rollout predictions expose variance in physical units.
                logvar = logvar + torch.log(self.state_normalizer.var + 1.0e-6)
        return EnsemblePrediction(mean=mean, logvar=logvar)

    @torch.no_grad()
    def set_elites(self, validation_loss: torch.Tensor) -> None:
        if validation_loss.shape != (self.cfg.ensemble_size,):
            raise ValueError(
                f"validation_loss must have shape ({self.cfg.ensemble_size},), got {tuple(validation_loss.shape)}."
            )
        self.elite_indices.copy_(torch.argsort(validation_loss)[: self.cfg.num_elites])

def load_residual_world_model(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    allow_h4_ablation: bool = False,
) -> ResidualWorldModel:
    """Load a WM only when its force, orientation and variance contracts match.

    H=20 remains the deployment default.  Offline validation utilities may
    explicitly opt into loading the controlled H=4 training ablation.
    """

    saved: dict[str, Any] = torch.load(Path(path), map_location=device, weights_only=False)
    if saved.get("format_version") != 4:
        raise ValueError("The active WM baseline requires checkpoint format_version=4.")
    metadata = saved.get("metadata", {})
    if metadata.get("backend") != "mjx_full_step":
        raise ValueError("The active WM baseline requires backend='mjx_full_step'.")
    if metadata.get("integrator") != "implicitfast":
        raise ValueError("The active WM baseline requires integrator='implicitfast'.")
    rollout_horizon = metadata.get("rollout_horizon")
    allowed_horizons = (4, 20) if allow_h4_ablation else (20,)
    if rollout_horizon not in allowed_horizons:
        suffix = " or the explicit H=4 ablation" if allow_h4_ablation else ""
        raise ValueError(f"The active WM baseline requires H=20 training{suffix}.")
    model_cfg = saved["model_cfg"]
    if model_cfg.prediction_type != "generalized_force":
        raise ValueError("The active WM baseline requires generalized-force prediction mode.")
    if not model_cfg.probabilistic:
        raise ValueError("The active WM baseline requires probabilistic NLL training.")
    if model_cfg.model_input_mode != "current_privileged_tilt":
        raise ValueError("The active WM baseline requires the current privileged-state input mode.")
    if model_cfg.uncertainty_mode != "next_state_diagonal":
        raise ValueError("The active WM baseline requires next-state diagonal uncertainty.")
    if saved.get("residual_force_frame") != model_cfg.residual_force_frame:
        raise ValueError(
            "Checkpoint residual-force coordinates do not match model_cfg; "
            "retrain instead of mixing force semantics."
        )
    if saved.get("model_orientation_input") != model_cfg.model_orientation_input:
        raise ValueError(
            "Checkpoint orientation input does not match model_cfg; retrain it "
            "instead of mixing input semantics."
        )
    if saved.get("model_input_mode") != model_cfg.model_input_mode:
        raise ValueError("Checkpoint model-input mode does not match model_cfg.")
    if (
        model_cfg.probabilistic
        and saved.get("variance_parameterization")
        != variance_parameterization(model_cfg)
    ):
        raise ValueError(
            "This probabilistic WM does not use the uncertainty contract declared "
            "by model_cfg. Retrain it instead of mixing checkpoint semantics."
        )
    trainer_cfg = saved.get("trainer", {}).get("cfg")
    deterministic_force_loss = bool(
        getattr(trainer_cfg, "deterministic_force_loss", False)
    )
    if deterministic_force_loss and model_cfg.probabilistic:
        raise ValueError(
            "This checkpoint uses the legacy invalid deterministic-MSE + "
            "probabilistic-head format. Its state variance was never trained; "
            "retrain the WM with the active hybrid NLL contract."
        )
    model = ResidualWorldModel(model_cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model
