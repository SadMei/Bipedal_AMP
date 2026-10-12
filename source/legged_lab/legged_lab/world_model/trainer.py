from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .dynamics import PhysicsInformedWorldModel
from .losses import discounted_rollout_mean, one_step_ensemble_loss
from .replay import ReplayBuffer, TransitionBatch
from .state import dynamic_state, model_input, state_delta_target


@dataclass(frozen=True)
class WorldModelTrainerCfg:
    learning_rate: float = 1.0e-3
    batch_size: int = 512
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 100.0
    normalizer_samples: int = 8192
    rollout_horizon: int = 20
    joint_pos_loss_weight: float = 1.0
    joint_vel_loss_weight: float = 1.0
    root_quat_loss_weight: float = 1.0
    root_height_loss_weight: float = 1.0
    foot_contact_loss_weight: float = 1.0
    # BeyondMimic-style phase curriculum for contiguous replay sequences.
    sequence_uniform_fraction: float = 1.0
    sequence_priority_fraction: float = 0.0
    sequence_event_fraction: float = 0.0
    sequence_priority_alpha: float = 1.0
    sequence_event_ranges: tuple[tuple[int, int], ...] = ()
    deterministic_force_loss: bool = False
    logvar_bound_loss_weight: float = 1.0e-2
    force_state_variance_floor: float = 1.0e-6
    # Prediction-time loss discount; 1.0 preserves the existing equal-weight
    # baseline. Validation MSE / elite ranking always remain time-uniform.
    rollout_loss_gamma: float = 1.0


class WorldModelTrainer:
    """Offline one-step or fully autoregressive generalized-residual trainer."""

    def __init__(
        self,
        world_model: PhysicsInformedWorldModel,
        cfg: WorldModelTrainerCfg | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.residual_model = world_model.residual_model
        self.cfg = cfg or WorldModelTrainerCfg()
        if self.cfg.rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive.")
        if not 0.0 < self.cfg.rollout_loss_gamma <= 1.0:
            raise ValueError("rollout_loss_gamma must be finite and in (0, 1].")
        loss_weights = (
            self.cfg.joint_pos_loss_weight,
            self.cfg.joint_vel_loss_weight,
            self.cfg.root_quat_loss_weight,
            self.cfg.root_height_loss_weight,
            self.cfg.foot_contact_loss_weight,
        )
        if any(weight <= 0.0 for weight in loss_weights):
            raise ValueError("All dynamic-state loss weights must be positive.")
        if self.cfg.logvar_bound_loss_weight < 0.0:
            raise ValueError("logvar_bound_loss_weight must be non-negative.")
        if self.cfg.force_state_variance_floor <= 0.0:
            raise ValueError("force_state_variance_floor must be positive.")
        fractions = (
            self.cfg.sequence_uniform_fraction,
            self.cfg.sequence_priority_fraction,
            self.cfg.sequence_event_fraction,
        )
        if any(fraction < 0.0 for fraction in fractions):
            raise ValueError("Sequence sampling fractions must be non-negative.")
        if abs(sum(fractions) - 1.0) > 1.0e-6:
            raise ValueError("Sequence sampling fractions must sum to one.")
        if self.cfg.sequence_priority_alpha <= 0.0:
            raise ValueError("sequence_priority_alpha must be positive.")
        for start, end in self.cfg.sequence_event_ranges:
            if start < 0 or end < start:
                raise ValueError(f"Invalid sequence event range ({start}, {end}).")
        self.device = torch.device(device)
        self.residual_model.to(self.device)
        if self.residual_model.cfg.prediction_type == "generalized_force":
            expected_probabilistic = not self.cfg.deterministic_force_loss
            if self.residual_model.cfg.probabilistic != expected_probabilistic:
                raise ValueError(
                    "Generalized-force WM architecture/loss mismatch: "
                    f"probabilistic={self.residual_model.cfg.probabilistic} requires "
                    f"deterministic_force_loss={not self.residual_model.cfg.probabilistic}."
                )
        self.optimizer = torch.optim.Adam(
            self.residual_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # Running statistics define the coordinate system seen by the first
        # layer.  Updating them before every gradient step silently moves that
        # coordinate system and is especially harmful when a broad simulator
        # model is fine-tuned on a small real-world replay.  Initialize once
        # for a fresh model; resumed real-world fine-tuning may freeze it.
        self._normalizer_initialized = False

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "normalizer_initialized": self._normalizer_initialized,
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        # Old checkpoints were created after at least one normalizer update,
        # even though they did not persist this bookkeeping flag.
        self._normalizer_initialized = bool(state.get("normalizer_initialized", True))

    @torch.no_grad()
    def update_normalizers(self, buffer: ReplayBuffer) -> None:
        count = min(max(len(buffer), 1), self.cfg.normalizer_samples)
        batch = buffer.sample(count, self.device)
        if self.residual_model.cfg.prediction_type == "state_delta":
            target = self._state_delta_target(batch.wm_history, batch.action, batch.next_wm_history)
            self.residual_model.update_normalizers(batch.wm_history, batch.action, target)
        else:
            value = model_input(self.residual_model.cfg, batch.wm_history, batch.action)
            self.residual_model.input_normalizer.update(value)
            if self.residual_model.cfg.uncertainty_mode in (
                "next_state_diagonal",
                "paper_privileged_diagonal",
            ):
                target_dynamic = dynamic_state(
                    self.residual_model.cfg.state,
                    batch.next_wm_history[:, -1, :],
                )
                state_slice = (
                    self.residual_model.cfg.state.paper_privileged_slice
                    if self.residual_model.cfg.uncertainty_mode
                    == "paper_privileged_diagonal"
                    else slice(
                        0,
                        self.residual_model.cfg.state.probabilistic_dynamic_dim,
                    )
                )
                self.residual_model.state_normalizer.update(
                    target_dynamic[..., state_slice]
                )
                if (
                    self.residual_model.cfg.uncertainty_mode
                    == "next_state_diagonal"
                ):
                    spec = self.residual_model.cfg.state
                    estimator_continuous = torch.cat(
                        (
                            target_dynamic[
                                ..., spec.probabilistic_dynamic_dim : spec.foot_contact_slice.start
                            ],
                            target_dynamic[..., spec.foot_height_slice],
                        ),
                        dim=-1,
                    )
                    self.residual_model.estimator_continuous_normalizer.update(
                        estimator_continuous
                    )

    def train(
        self,
        buffer: ReplayBuffer,
        updates: int,
        *,
        anchor_buffer: ReplayBuffer | None = None,
        target_ratio: float = 1.0,
        freeze_normalizer: bool = False,
    ) -> dict[str, float]:
        """Train on target data while retaining a broad anchor distribution.

        ``buffer`` is the target/real replay.  When ``anchor_buffer`` is
        supplied, every minibatch contains ``target_ratio`` target samples and
        the remainder from the simulator/pretraining replay.  This rehearsal
        prevents a narrow real log from overwriting behavior at states such as
        the deployment default stand.  A resumed model should normally set
        ``freeze_normalizer=True`` so the pretraining input coordinates remain
        fixed.
        """
        if updates <= 0:
            raise ValueError("updates must be positive.")
        if len(buffer) == 0:
            raise ValueError("Cannot train the world model from an empty buffer.")
        if not 0.0 <= target_ratio <= 1.0:
            raise ValueError("target_ratio must be in [0, 1].")
        if anchor_buffer is None and target_ratio != 1.0:
            raise ValueError("target_ratio must be 1.0 when anchor_buffer is not supplied.")
        if anchor_buffer is not None and len(anchor_buffer) == 0:
            raise ValueError("anchor_buffer cannot be empty.")
        if not freeze_normalizer and not self._normalizer_initialized:
            # A fresh pretraining run establishes its normalization once.
            # Mixed/resumed fine-tuning is expected to freeze the loaded
            # statistics, but this branch remains well-defined for callers
            # intentionally starting a mixed model from scratch.
            self.update_normalizers(buffer)
            self._normalizer_initialized = True
        loss_sum = 0.0
        mse_sum = 0.0
        component_sum: dict[str, float] = {}
        for _ in range(updates):
            if self.cfg.rollout_horizon > 1:
                if self.residual_model.cfg.prediction_type != "generalized_force":
                    raise ValueError("Multi-step WM training currently requires generalized_force mode.")
                total_loss, member_mse, member_components = self._generalized_force_rollout_loss(
                    buffer,
                    bootstrap=True,
                    anchor_buffer=anchor_buffer,
                    target_ratio=target_ratio,
                )
                batch = None
            else:
                batch = self._sample_training_batch(buffer, anchor_buffer, target_ratio)
            if self.residual_model.cfg.prediction_type == "state_delta":
                assert batch is not None
                with torch.no_grad():
                    target = self._state_delta_target(batch.wm_history, batch.action, batch.next_wm_history)
                loss = one_step_ensemble_loss(
                    self.residual_model,
                    batch.wm_history,
                    batch.action,
                    target,
                    bootstrap=True,
                )
                total_loss = loss.total
                member_mse = loss.mse_per_member
                member_components = {}
            elif self.cfg.rollout_horizon == 1:
                assert batch is not None
                total_loss, member_mse, member_components = self._generalized_force_loss(
                    batch, bootstrap=True
                )
            if self.residual_model.cfg.probabilistic:
                # Standard probabilistic-ensemble regularizer: discourage the
                # learned interval from expanding merely to reduce NLL while
                # retaining differentiable soft bounds for recovery.
                total_loss = total_loss + self.cfg.logvar_bound_loss_weight * (
                    self.residual_model.max_logvar.sum()
                    - self.residual_model.min_logvar.sum()
                )
            self.optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError("Nonfinite WM loss; optimizer update was not applied.")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.residual_model.parameters(), self.cfg.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            loss_sum += float(total_loss.detach())
            mse_sum += float(member_mse.mean().detach())
            for name, value in member_components.items():
                component_sum[name] = component_sum.get(name, 0.0) + float(
                    value.mean().detach()
                )

        validation = self.validate(buffer, anchor_buffer=anchor_buffer, target_ratio=target_ratio)
        return {
            "world_model/loss": loss_sum / updates,
            "world_model/train_mse": mse_sum / updates,
            **{
                f"world_model/train_{name}": value / updates
                for name, value in component_sum.items()
            },
            **validation,
        }

    @torch.no_grad()
    def validate(
        self,
        buffer: ReplayBuffer,
        *,
        anchor_buffer: ReplayBuffer | None = None,
        target_ratio: float = 1.0,
    ) -> dict[str, float]:
        if self.cfg.rollout_horizon > 1:
            _, member_mse, member_components = self._generalized_force_rollout_loss(
                buffer,
                bootstrap=False,
                anchor_buffer=anchor_buffer,
                target_ratio=target_ratio,
            )
            self.residual_model.set_elites(member_mse)
            return {
                "world_model/validation_mse": float(member_mse.mean()),
                "world_model/best_validation_mse": float(member_mse.min()),
                **{
                    f"world_model/validation_{name}": float(value.mean())
                    for name, value in member_components.items()
                },
            }
        batch = self._sample_training_batch(buffer, anchor_buffer, target_ratio)
        if self.residual_model.cfg.prediction_type == "state_delta":
            target = self._state_delta_target(batch.wm_history, batch.action, batch.next_wm_history)
            member_mse = one_step_ensemble_loss(
                self.residual_model,
                batch.wm_history,
                batch.action,
                target,
                bootstrap=False,
            ).mse_per_member
            member_components = {}
        else:
            _, member_mse, member_components = self._generalized_force_loss(
                batch, bootstrap=False
            )
        self.residual_model.set_elites(member_mse)
        return {
            "world_model/validation_mse": float(member_mse.mean()),
            "world_model/best_validation_mse": float(member_mse.min()),
            **{
                f"world_model/validation_{name}": float(value.mean())
                for name, value in member_components.items()
            },
        }

    def _sample_training_batch(
        self,
        target_buffer: ReplayBuffer,
        anchor_buffer: ReplayBuffer | None,
        target_ratio: float,
    ) -> TransitionBatch:
        if anchor_buffer is None:
            return target_buffer.sample(self.cfg.batch_size, self.device)
        target_count = int(round(self.cfg.batch_size * target_ratio))
        target_count = min(max(target_count, 0), self.cfg.batch_size)
        anchor_count = self.cfg.batch_size - target_count
        batches: list[TransitionBatch] = []
        if target_count:
            batches.append(target_buffer.sample(target_count, self.device))
        if anchor_count:
            batches.append(anchor_buffer.sample(anchor_count, self.device))
        batch = TransitionBatch.cat(batches)
        permutation = torch.randperm(len(batch), device=batch.action.device)
        return batch.index(permutation)

    @torch.no_grad()
    def _state_delta_target(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        real_next_history: torch.Tensor,
    ) -> torch.Tensor:
        nominal_next_frame = self.world_model.nominal_next_frame(history, action)
        real_next_frame = real_next_history[:, -1, :]
        return state_delta_target(self.residual_model.cfg.state, real_next_frame, nominal_next_frame)

    def _generalized_state_objective(
        self,
        predicted_dynamic: torch.Tensor,
        target_dynamic: torch.Tensor,
        state_logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Hybrid objective for physical dynamics and estimator observations.

        The leading physical-state dimensions use Gaussian NLL. The six
        continuous estimator values are deterministic MSE targets, and the two
        contact probabilities use soft-label BCE. Only the NLL prefix has a
        variance head or receives process noise during rollout.
        """

        spec = self.residual_model.cfg.state
        squared_error = (predicted_dynamic - target_dynamic).square()
        weights = self._dynamic_loss_weights(squared_error)
        mse_per_member = (squared_error * weights).mean(dim=(1, 2))
        if not self.residual_model.cfg.probabilistic or self.cfg.deterministic_force_loss:
            return mse_per_member, mse_per_member, {}

        if self.residual_model.cfg.uncertainty_mode == "paper_privileged_diagonal":
            privileged_error = (
                predicted_dynamic[..., spec.paper_privileged_slice]
                - target_dynamic[..., spec.paper_privileged_slice]
            ) * torch.rsqrt(self.residual_model.state_normalizer.var + 1.0e-6)
            privileged_squared_error = privileged_error.square()
            if state_logvar.shape != privileged_squared_error.shape:
                raise RuntimeError(
                    "Paper privileged-state logvar shape mismatch: "
                    f"{tuple(state_logvar.shape)} vs "
                    f"{tuple(privileged_squared_error.shape)}."
                )
            privileged_nll = (
                torch.exp(-state_logvar) * privileged_squared_error + state_logvar
            )
            # Equation (5)/(19): one unweighted diagonal-Gaussian NLL over the
            # exact privileged state. Contact and foot-height are reconstructed
            # diagnostics, not learned observation heads in the paper baseline.
            member_loss = privileged_nll.mean(dim=(1, 2))
            return member_loss, mse_per_member, {
                "paper_privileged_nll": member_loss,
            }

        probabilistic_dim = spec.probabilistic_dynamic_dim
        physical_error = (
            predicted_dynamic[..., :probabilistic_dim]
            - target_dynamic[..., :probabilistic_dim]
        ) * torch.rsqrt(self.residual_model.state_normalizer.var + 1.0e-6)
        physical_squared_error = physical_error.square()
        if state_logvar.shape != physical_squared_error.shape:
            raise RuntimeError(
                "Physical-state logvar shape mismatch: "
                f"{tuple(state_logvar.shape)} vs {tuple(physical_squared_error.shape)}."
            )
        physical_nll = (
            torch.exp(-state_logvar) * physical_squared_error + state_logvar
        )

        estimator_continuous_error = torch.cat(
            (
                predicted_dynamic[
                    ..., probabilistic_dim : spec.foot_contact_slice.start
                ]
                - target_dynamic[
                    ..., probabilistic_dim : spec.foot_contact_slice.start
                ],
                predicted_dynamic[..., spec.foot_height_slice]
                - target_dynamic[..., spec.foot_height_slice],
            ),
            dim=-1,
        ) * torch.rsqrt(
            self.residual_model.estimator_continuous_normalizer.var + 1.0e-6
        )
        estimator_continuous_mse = estimator_continuous_error.square()
        base_height_dim = spec.foot_contact_slice.start - probabilistic_dim
        base_height_mse = estimator_continuous_mse[..., :base_height_dim]
        foot_height_mse = estimator_continuous_mse[..., base_height_dim:]

        predicted_contact = predicted_dynamic[..., spec.foot_contact_slice].clamp(
            1.0e-6, 1.0 - 1.0e-6
        )
        target_contact = target_dynamic[..., spec.foot_contact_slice].clamp(0.0, 1.0)
        contact_bce = F.binary_cross_entropy(
            predicted_contact, target_contact, reduction="none"
        )

        element_loss = torch.cat(
            (
                physical_nll,
                base_height_mse,
                contact_bce,
                foot_height_mse,
            ),
            dim=-1,
        )
        if element_loss.shape != squared_error.shape:
            raise RuntimeError(
                f"Hybrid loss produced {tuple(element_loss.shape)}, expected "
                f"{tuple(squared_error.shape)}."
            )
        member_loss = (element_loss * weights).mean(dim=(1, 2))
        components = {
            "physical_nll": physical_nll.mean(dim=(1, 2)),
            "estimator_continuous_mse": estimator_continuous_mse.mean(dim=(1, 2)),
            "contact_bce": contact_bce.mean(dim=(1, 2)),
        }
        return member_loss, mse_per_member, components

    def _propagated_state_logvar(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        residual_force: torch.Tensor,
        predicted_logvar: torch.Tensor,
        jacobian: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Map a diagonal residual-force covariance into next-state variance.

        The replay has no force labels.  A local MJX Jacobian supplies the
        physically meaningful likelihood bridge

          Var[s_next_i] ~= sum_j (ds_i / dtau_j)^2 Var[tau_j].

        The Jacobian is detached by the backend; gradients still reach the
        force-variance head through ``exp(predicted_logvar)`` and reach the
        force mean through the ordinary differentiable next-state error path.
        """

        if self.residual_model.cfg.uncertainty_mode in (
            "next_state_diagonal",
            "paper_privileged_diagonal",
        ):
            return predicted_logvar, {}
        if predicted_logvar.shape != residual_force.shape:
            raise RuntimeError(
                "Residual-force logvar must match residual force: "
                f"{tuple(predicted_logvar.shape)} vs {tuple(residual_force.shape)}."
            )
        if jacobian is None:
            nominal = self.world_model.nominal_physics
            jacobian_fn = getattr(nominal, "residual_jacobian", None)
            if jacobian_fn is None:
                raise TypeError(
                    "residual_force_diagonal training requires a nominal backend "
                    "with residual_jacobian(history, action, force)."
                )
            jacobian = jacobian_fn(history, action, residual_force)
        probabilistic_dim = self.residual_model.cfg.state.probabilistic_dynamic_dim
        jacobian = jacobian[..., :probabilistic_dim, :]
        state_variance = torch.einsum(
            "...ij,...j->...i", jacobian.square(), torch.exp(predicted_logvar)
        ) + self.cfg.force_state_variance_floor
        return torch.log(state_variance), {
            "residual_force_std": torch.exp(0.5 * predicted_logvar).mean(dim=-1),
            "propagated_state_std": torch.sqrt(state_variance).mean(dim=-1),
        }

    def _generalized_force_loss(
        self,
        batch,
        bootstrap: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Backpropagate next-state error through a differentiable nominal step.

        This is the closest port of LIFT's ``contact_integrate_only`` path.  It
        intentionally fails during backward if the supplied backend detaches
        tensors (as Isaac/ordinary MuJoCo do), making the integration contract
        explicit instead of silently training a force head with no gradient.
        """

        batch_size = len(batch)
        if bootstrap:
            index = torch.randint(
                batch_size,
                (self.residual_model.cfg.ensemble_size, batch_size),
                device=batch.policy_obs.device,
            )
        else:
            index = torch.arange(batch_size, device=batch.policy_obs.device).expand(
                self.residual_model.cfg.ensemble_size, -1
            )
        value = model_input(self.residual_model.cfg, batch.wm_history, batch.action)
        ensemble_value = value[index]
        prediction = self.residual_model.forward_normalized(ensemble_value)
        # Force targets are not available; the output normalizer remains the
        # identity and the predicted mean is integrated directly.
        residual_force = self.residual_model.target_normalizer.denormalize(prediction.mean)

        # All members use the same nominal backend. Flatten ensemble and batch
        # into one MJX call instead of launching seven identical JIT kernels
        # serially; reshape afterward keeps the loss exactly member-wise.
        ensemble_size = self.residual_model.cfg.ensemble_size
        flat_index = index.reshape(-1)
        flat_history = batch.wm_history[flat_index]
        flat_action = batch.action[flat_index]
        flat_force = residual_force.reshape(-1, residual_force.shape[-1])
        if self.residual_model.cfg.uncertainty_mode == "residual_force_diagonal":
            step_fn = getattr(
                self.world_model.nominal_physics,
                "step_with_residual_jacobian",
                None,
            )
            if step_fn is None:
                raise TypeError(
                    "residual_force_diagonal requires step_with_residual_jacobian."
                )
            flat_predicted_next, force_jacobian = step_fn(
                flat_history, flat_action, flat_force
            )
        else:
            flat_predicted_next = self.world_model.nominal_physics(
                flat_history, flat_action, flat_force
            )
            force_jacobian = None
        predicted_next = flat_predicted_next.reshape(ensemble_size, batch_size, -1)
        target_next = dynamic_state(
            self.residual_model.cfg.state,
            batch.next_wm_history[index, -1, :],
        ).clone()
        predicted_dynamic = dynamic_state(self.residual_model.cfg.state, predicted_next)
        state_logvar, uncertainty_components = self._propagated_state_logvar(
            flat_history,
            flat_action,
            flat_force,
            prediction.logvar.reshape(-1, prediction.logvar.shape[-1]),
            jacobian=force_jacobian,
        )
        state_logvar = state_logvar.reshape(
            ensemble_size, batch_size, state_logvar.shape[-1]
        )
        quat_start = 6 + 2 * self.residual_model.cfg.state.num_joints
        quat_slice = slice(quat_start, quat_start + 4)
        sign = torch.where(
            (
                predicted_dynamic[..., quat_slice].detach()
                * target_next[..., quat_slice]
            ).sum(dim=-1, keepdim=True)
            < 0.0,
            -1.0,
            1.0,
        )
        target_next[..., quat_slice] *= sign
        member_loss, mse_per_member, components = self._generalized_state_objective(
            predicted_dynamic, target_next, state_logvar
        )
        components.update(
            {
                name: value.reshape(ensemble_size, batch_size).mean(dim=1)
                for name, value in uncertainty_components.items()
            }
        )
        return member_loss.sum(), mse_per_member, components

    def _generalized_force_rollout_loss(
        self,
        buffer: ReplayBuffer,
        bootstrap: bool,
        *,
        anchor_buffer: ReplayBuffer | None = None,
        target_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Train on contiguous teacher-forced action sequences without crossing episodes.

        Physical state recursively comes from MJX plus the learned force. The
        recorded action and exogenous reference/action context are supplied at
        every step. MJX's custom autograd bridge differentiates through qpos,
        qvel and residual force, so later-step losses reach earlier residuals.
        Only the optimization objective is time-discounted. Reported member
        MSE, original component metrics and elite ranking stay time-uniform.
        """

        horizon = self.cfg.rollout_horizon
        if anchor_buffer is None:
            source_counts = ((buffer, self.cfg.batch_size),)
        else:
            target_count = int(round(self.cfg.batch_size * target_ratio))
            target_count = min(max(target_count, 0), self.cfg.batch_size)
            source_counts = (
                (buffer, target_count),
                (anchor_buffer, self.cfg.batch_size - target_count),
            )
        sources: list[tuple[ReplayBuffer, torch.Tensor]] = []
        for source_buffer, count in source_counts:
            if count:
                sources.append(
                    (
                        source_buffer,
                        self._sample_sequence_starts(
                            source_buffer,
                            horizon,
                            bootstrap,
                            batch_size=count,
                        ),
                    )
                )
        if not sources:
            raise RuntimeError("Multi-step replay mixture produced an empty batch.")
        history = torch.cat(
            [source.wm_history[starts].to(self.device) for source, starts in sources],
            dim=1,
        )
        ensemble_size, batch_size = history.shape[:2]
        loss_steps = []
        mse_steps = []
        component_steps: dict[str, list[torch.Tensor]] = {}
        for step in range(horizon):
            action = torch.cat(
                [source.action[starts + step].to(self.device) for source, starts in sources],
                dim=1,
            )
            target_frame = torch.cat(
                [
                    source.next_wm_history[starts + step, -1, :].to(self.device)
                    for source, starts in sources
                ],
                dim=1,
            )
            value = model_input(self.residual_model.cfg, history, action)
            prediction = self.residual_model.forward_normalized(value)
            residual_force = self.residual_model.target_normalizer.denormalize(prediction.mean)
            flat_history = history.reshape(-1, history.shape[-2], history.shape[-1])
            flat_action = action.reshape(-1, action.shape[-1])
            flat_force = residual_force.reshape(-1, residual_force.shape[-1])
            if self.residual_model.cfg.uncertainty_mode == "residual_force_diagonal":
                step_fn = getattr(
                    self.world_model.nominal_physics,
                    "step_with_residual_jacobian",
                    None,
                )
                if step_fn is None:
                    raise TypeError(
                        "residual_force_diagonal requires step_with_residual_jacobian."
                    )
                flat_predicted_frame, force_jacobian = step_fn(
                    flat_history, flat_action, flat_force
                )
            else:
                flat_predicted_frame = self.world_model.nominal_physics(
                    flat_history, flat_action, flat_force
                )
                force_jacobian = None
            predicted_frame = flat_predicted_frame.reshape(
                ensemble_size, batch_size, -1
            )
            target_dynamic = dynamic_state(
                self.residual_model.cfg.state, target_frame
            ).clone()
            predicted_dynamic = dynamic_state(
                self.residual_model.cfg.state, predicted_frame
            )
            state_logvar, uncertainty_components = self._propagated_state_logvar(
                flat_history,
                flat_action,
                flat_force,
                prediction.logvar.reshape(-1, prediction.logvar.shape[-1]),
                jacobian=force_jacobian,
            )
            state_logvar = state_logvar.reshape(
                ensemble_size, batch_size, state_logvar.shape[-1]
            )
            quat_start = 6 + 2 * self.residual_model.cfg.state.num_joints
            quat_slice = slice(quat_start, quat_start + 4)
            sign = torch.where(
                (
                    predicted_dynamic[..., quat_slice].detach()
                    * target_dynamic[..., quat_slice]
                ).sum(dim=-1, keepdim=True)
                < 0.0,
                -1.0,
                1.0,
            )
            target_dynamic[..., quat_slice] *= sign
            member_loss, member_mse, components = self._generalized_state_objective(
                predicted_dynamic, target_dynamic, state_logvar
            )
            components.update(
                {
                    name: value.reshape(ensemble_size, batch_size).mean(dim=1)
                    for name, value in uncertainty_components.items()
                }
            )
            loss_steps.append(member_loss)
            mse_steps.append(member_mse)
            for name, value in components.items():
                component_steps.setdefault(name, []).append(value)

            # The motion reference and previous-action fields are exogenous.
            # Teacher-force them while recursively carrying the physical state.
            next_frame = torch.cat(
                (
                    predicted_dynamic,
                    target_frame[..., self.residual_model.cfg.state.dynamic_dim :],
                ),
                dim=-1,
            )
            history = torch.cat((history[..., 1:, :], next_frame.unsqueeze(-2)), dim=-2)

        stacked_loss = torch.stack(loss_steps)
        member_loss = discounted_rollout_mean(stacked_loss, self.cfg.rollout_loss_gamma)
        member_mse = torch.stack(mse_steps).mean(dim=0)
        member_components = {
            name: torch.stack(values).mean(dim=0)
            for name, values in component_steps.items()
        }
        # Detached diagnostics never alter the gradient or sequence sampler.
        # Keep legacy component names equal-weight for cross-run comparison.
        member_components["uniform_objective"] = stacked_loss.detach().mean(dim=0)
        member_components["discounted_objective"] = member_loss.detach()
        member_components["discounted_mse"] = discounted_rollout_mean(
            torch.stack(mse_steps).detach(), self.cfg.rollout_loss_gamma
        )
        for offset in sorted({0, min(3, horizon-1), min(9, horizon-1), horizon-1}):
            member_components[f"mse_step_{offset+1}"] = mse_steps[offset].detach()
        return member_loss.sum(), member_mse, member_components

    def _sample_sequence_starts(
        self,
        buffer: ReplayBuffer,
        horizon: int,
        bootstrap: bool,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if len(buffer) < horizon:
            raise ValueError(f"Replay has {len(buffer)} transitions, shorter than horizon {horizon}.")
        # A done on the last transition is allowed; only an earlier done would
        # make the requested sequence cross into another episode.
        done = torch.logical_or(
            buffer.terminated[: len(buffer)], buffer.truncated[: len(buffer)]
        ).flatten()
        candidate = torch.arange(len(buffer) - horizon + 1, device=done.device)
        if horizon > 1:
            crosses = torch.stack(
                [done[offset : offset + len(candidate)] for offset in range(horizon - 1)]
            ).any(dim=0)
            candidate = candidate[~crosses]
        if candidate.numel() == 0:
            raise ValueError("No episode-safe sequence start exists for the requested horizon.")
        rows = self.residual_model.cfg.ensemble_size if bootstrap else 1
        sample_batch_size = self.cfg.batch_size if batch_size is None else batch_size
        if sample_batch_size <= 0:
            raise ValueError("Sequence batch_size must be positive.")
        count = rows * sample_batch_size
        uniform_count = int(round(count * self.cfg.sequence_uniform_fraction))
        priority_count = int(round(count * self.cfg.sequence_priority_fraction))
        event_count = count - uniform_count - priority_count
        if event_count < 0:
            priority_count = max(priority_count + event_count, 0)
            event_count = count - uniform_count - priority_count

        sampled = []
        if uniform_count:
            sampled.append(candidate[torch.randint(candidate.numel(), (uniform_count,), device=candidate.device)])
        if priority_count:
            probability = self._sequence_difficulty(buffer, candidate)
            sampled.append(candidate[torch.multinomial(probability, priority_count, replacement=True)])
        if event_count:
            event_candidate = self._event_sequence_candidates(candidate, horizon)
            if event_candidate.numel():
                sampled.append(
                    event_candidate[
                        torch.randint(event_candidate.numel(), (event_count,), device=candidate.device)
                    ]
                )
            else:
                # Keep the requested batch size even when no explicit event
                # overlaps this replay/horizon.
                probability = self._sequence_difficulty(buffer, candidate)
                sampled.append(candidate[torch.multinomial(probability, event_count, replacement=True)])
        choice = torch.cat(sampled)
        choice = choice[torch.randperm(choice.numel(), device=choice.device)].reshape(
            rows, sample_batch_size
        )
        if not bootstrap:
            choice = choice.expand(self.residual_model.cfg.ensemble_size, -1)
        return choice

    def _sequence_difficulty(self, buffer: ReplayBuffer, candidate: torch.Tensor) -> torch.Tensor:
        """Compute robust phase difficulty solely from recorded hardware state."""

        spec = self.residual_model.cfg.state
        current = buffer.wm_history[candidate, -1, :]
        following = buffer.next_wm_history[candidate, -1, :]
        q = following[:, 6 : 6 + spec.num_joints]
        dq = following[:, 6 + spec.num_joints : 6 + 2 * spec.num_joints]
        ref_q = following[:, spec.dynamic_dim : spec.dynamic_dim + spec.num_joints]
        ref_dq = following[
            :, spec.dynamic_dim + spec.num_joints : spec.dynamic_dim + 2 * spec.num_joints
        ]
        quat_start = 6 + 2 * spec.num_joints
        contact_start = quat_start + 8
        signals = (
            (q - ref_q).square().mean(dim=-1).sqrt(),
            (dq - ref_dq).square().mean(dim=-1).sqrt(),
            following[:, 0:2].square().sum(dim=-1).sqrt(),
            (following[:, quat_start + 7] - current[:, quat_start + 7]).abs(),
            (
                following[:, 6 + spec.num_joints : 6 + 2 * spec.num_joints]
                - current[:, 6 + spec.num_joints : 6 + 2 * spec.num_joints]
            ).square().mean(dim=-1).sqrt(),
            (
                following[:, contact_start : contact_start + spec.num_feet]
                - current[:, contact_start : contact_start + spec.num_feet]
            ).abs().amax(dim=-1),
        )
        score = torch.zeros_like(signals[0])
        for signal in signals:
            positive = signal[signal > 1.0e-8]
            scale = positive.median() if positive.numel() else signal.new_tensor(1.0)
            score += (signal / scale.clamp_min(1.0e-6)).clamp_max(10.0)
        # The floor prevents permanent starvation of nominal/easy phases.
        score = (score + 0.05).pow(self.cfg.sequence_priority_alpha)
        return score / score.sum().clamp_min(1.0e-12)

    def _event_sequence_candidates(self, candidate: torch.Tensor, horizon: int) -> torch.Tensor:
        if not self.cfg.sequence_event_ranges:
            return candidate.new_empty((0,))
        selected = torch.zeros_like(candidate, dtype=torch.bool)
        for event_start, event_end in self.cfg.sequence_event_ranges:
            selected |= (candidate <= event_end) & (candidate + horizon - 1 >= event_start)
        return candidate[selected]

    def _dynamic_loss_weights(self, value: torch.Tensor) -> torch.Tensor:
        spec = self.residual_model.cfg.state
        weights = value.new_ones(spec.dynamic_dim)
        weights[6 : 6 + spec.num_joints] *= self.cfg.joint_pos_loss_weight
        weights[6 + spec.num_joints : 6 + 2 * spec.num_joints] *= self.cfg.joint_vel_loss_weight
        quat_start = 6 + 2 * spec.num_joints
        weights[quat_start : quat_start + 4] *= self.cfg.root_quat_loss_weight
        weights[quat_start + 7] *= self.cfg.root_height_loss_weight
        weights[quat_start + 8 : quat_start + 8 + spec.num_feet] *= self.cfg.foot_contact_loss_weight
        # Preserve the original loss scale when all multipliers are one.
        return weights / weights.mean()
