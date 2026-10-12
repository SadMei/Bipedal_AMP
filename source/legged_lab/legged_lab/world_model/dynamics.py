from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
import inspect

import torch
from torch.nn import functional as F

from .models import ResidualWorldModel


class NominalPhysics(Protocol):
    """Batched simulator step initialized from the last frame in each history."""

    def __call__(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        generalized_residual_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a full next world-model frame."""


ObservationFn = Callable[[torch.Tensor], torch.Tensor]
RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TerminationFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class WorldModelStep:
    next_history: torch.Tensor
    next_policy_obs: torch.Tensor
    next_critic_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    uncertainty: torch.Tensor
    next_reference: object | None = None


class PhysicsInformedWorldModel:
    """Nominal simulator step corrected by a learned ensemble residual."""

    def __init__(
        self,
        residual_model: ResidualWorldModel,
        nominal_physics: NominalPhysics,
        observation_fn: ObservationFn,
        reward_fn: RewardFn,
        termination_fn: TerminationFn,
        uncertainty_threshold: float | None = None,
        critic_observation_fn: ObservationFn | None = None,
    ) -> None:
        self.residual_model = residual_model
        self.nominal_physics = nominal_physics
        self.observation_fn = observation_fn
        self.critic_observation_fn = observation_fn if critic_observation_fn is None else critic_observation_fn
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.uncertainty_threshold = uncertainty_threshold

    @torch.no_grad()
    def step(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        reference: object | None = None,
        next_reference: object | None = None,
    ) -> WorldModelStep:
        residual, state_logvar, uncertainty = self._select_prediction(history, action)
        if self.residual_model.cfg.prediction_type == "generalized_force":
            next_frame = self.nominal_physics(history, action, residual)
        else:
            next_frame = self.nominal_physics(history, action, None).clone()
            next_frame[..., : self.residual_model.cfg.state.dynamic_dim] += residual
        self._project_state(next_frame)

        # Reference joint targets are exogenous motion-clock context, not
        # physical state predicted by the nominal simulator.  The nominal
        # backend necessarily carries the previous frame's context, so update
        # it here after the caller has advanced the reference clock.  Keeping
        # this in the common WM step also covers both generalized-force and
        # state-delta models.
        if next_reference is not None:
            reference_joint_pos = getattr(next_reference, "joint_pos", None)
            reference_joint_vel = getattr(next_reference, "joint_vel", None)
            if reference_joint_pos is not None and reference_joint_vel is not None:
                spec = self.residual_model.cfg.state
                context_start = spec.dynamic_dim
                next_frame[..., context_start : context_start + spec.num_joints] = reference_joint_pos
                next_frame[
                    ..., context_start + spec.num_joints : context_start + 2 * spec.num_joints
                ] = reference_joint_vel

        next_history = torch.cat((history[:, 1:, :], next_frame.unsqueeze(1)), dim=1)
        next_policy_obs = _call_reference_fn(self.observation_fn, (next_history,), (next_reference,))
        next_critic_obs = _call_reference_fn(self.critic_observation_fn, (next_history,), (next_reference,))
        reward = _call_reference_fn(
            self.reward_fn,
            (history, action, next_history),
            (reference, next_reference),
        ).reshape(-1, 1)
        terminated, truncated = _call_reference_fn(self.termination_fn, (next_history,), (next_reference,))
        terminated = terminated.reshape(-1, 1).bool()
        truncated = truncated.reshape(-1, 1).bool()
        if self.uncertainty_threshold is not None:
            truncated = torch.logical_or(truncated, uncertainty.reshape(-1, 1) > self.uncertainty_threshold)
        return WorldModelStep(
            next_history=next_history,
            next_policy_obs=next_policy_obs,
            next_critic_obs=next_critic_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            uncertainty=uncertainty.reshape(-1, 1),
            next_reference=next_reference,
        )

    @torch.no_grad()
    def nominal_next_frame(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.nominal_physics(history, action, None)

    def _select_prediction(
        self, history: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the elite ensemble mean without sampling model uncertainty."""

        prediction = self.residual_model(history, action)
        elite = self.residual_model.elite_indices
        means = prediction.mean[elite]
        logvars = prediction.logvar[elite]
        selected_mean = means.mean(dim=0)
        # Keep the learned variance as a diagnostic/safety signal. It is never
        # sampled into residual force or next state in the active baseline.
        selected_logvar = torch.logsumexp(logvars, dim=0) - torch.log(
            logvars.new_tensor(float(logvars.shape[0]))
        )

        uncertainty_dim = self.residual_model.cfg.uncertainty_dim
        if selected_logvar.shape[-1] != uncertainty_dim:
            raise RuntimeError(
                f"Expected {uncertainty_dim}-D "
                f"{self.residual_model.cfg.uncertainty_mode} logvar, "
                f"got {selected_logvar.shape[-1]}."
            )
        # Scalar diagnostic used only by an optional rollout safety threshold.
        uncertainty = (
            torch.exp(selected_logvar).mean(dim=-1).sqrt()
            if self.residual_model.cfg.probabilistic
            else selected_logvar.new_zeros(selected_logvar.shape[:-1])
        )
        return selected_mean, selected_logvar, uncertainty

    def _project_state(self, frame: torch.Tensor) -> None:
        spec = self.residual_model.cfg.state
        start = 6 + 2 * spec.num_joints
        quaternion = frame[..., start : start + 4]
        frame[..., start : start + 4] = F.normalize(quaternion, dim=-1, eps=1.0e-6)
        quaternion = frame[..., start : start + 4]
        quaternion_sign = torch.where(quaternion[..., :1] < 0.0, -1.0, 1.0)
        frame[..., start : start + 4] = quaternion * quaternion_sign
        # Projected gravity is a deterministic function of root orientation.
        # Reconstruct it after each transition so redundant fields cannot
        # disagree and feed an impossible observation to the policy.
        quat = frame[..., start : start + 4]
        gravity_world = frame.new_zeros((*frame.shape[:-1], 3))
        gravity_world[..., 2] = -1.0
        conjugate = quat.clone()
        conjugate[..., 1:] *= -1.0
        xyz = conjugate[..., 1:]
        frame[..., :3] = gravity_world + 2.0 * (
            conjugate[..., :1] * torch.cross(xyz, gravity_world, dim=-1)
            + torch.cross(xyz, torch.cross(xyz, gravity_world, dim=-1), dim=-1)
        )
        root_height_index = start + 7
        frame[..., root_height_index : root_height_index + 1].clamp_(min=0.0)
        contact_start = root_height_index + 1
        frame[..., contact_start : contact_start + spec.num_feet].clamp_(0.0, 1.0)


def _call_reference_fn(function: Callable, base_args: tuple, reference_args: tuple):
    """Call legacy callbacks unchanged, or pass reference context when declared."""

    try:
        signature = inspect.signature(function)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        accepts_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_varargs = False
        positional = []
    if accepts_varargs or len(positional) >= len(base_args) + len(reference_args):
        return function(*base_args, *reference_args)
    return function(*base_args)
