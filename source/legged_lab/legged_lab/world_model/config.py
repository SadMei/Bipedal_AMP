from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PredictionType = Literal["generalized_force", "state_delta"]
ResidualForceFrame = Literal["body_linear"]
ModelInputMode = Literal["current_privileged_tilt"]
UncertaintyMode = Literal[
    "next_state_diagonal",
    "paper_privileged_diagonal",
    "residual_force_diagonal",
]


@dataclass(frozen=True)
class WorldModelStateSpec:
    """Robot-sized shape contract for the whole-body world-model state.

    A frame is split into a dynamic state and exogenous context.  Only the
    dynamic part is predicted in ``state_delta`` mode; reference motion and
    previous actions are inputs, not prediction targets.
    """

    num_joints: int = 25
    num_feet: int = 2
    history_length: int = 3

    @property
    def dynamic_dim(self) -> int:
        # gravity(3), base angular velocity(3), q, dq, root quaternion(4),
        # base linear velocity(3), root height(1), contact, foot height.
        return 14 + 2 * self.num_joints + 2 * self.num_feet

    @property
    def estimator_state_dim(self) -> int:
        """Policy-facing estimator values stored at the tail of dynamic state."""

        # base linear velocity(3), root height(1), foot contact and foot height.
        return 4 + 2 * self.num_feet

    @property
    def probabilistic_dynamic_dim(self) -> int:
        """Physical prefix supervised by Gaussian NLL and sampled in rollout."""

        return self.dynamic_dim - self.estimator_state_dim

    @property
    def paper_privileged_slice(self) -> slice:
        """LIFT privileged state ``[omega, q, dq, quat, v, height]``."""

        return slice(3, self.root_height_slice.stop)

    @property
    def paper_privileged_dim(self) -> int:
        return self.paper_privileged_slice.stop - self.paper_privileged_slice.start

    @property
    def base_linear_velocity_slice(self) -> slice:
        start = self.probabilistic_dynamic_dim
        return slice(start, start + 3)

    @property
    def root_height_slice(self) -> slice:
        start = self.probabilistic_dynamic_dim + 3
        return slice(start, start + 1)

    @property
    def foot_contact_slice(self) -> slice:
        start = self.probabilistic_dynamic_dim + 4
        return slice(start, start + self.num_feet)

    @property
    def foot_height_slice(self) -> slice:
        start = self.probabilistic_dynamic_dim + 4 + self.num_feet
        return slice(start, start + self.num_feet)

    @property
    def context_dim(self) -> int:
        # Reference joint position/velocity and the previously applied action.
        return 3 * self.num_joints

    @property
    def frame_dim(self) -> int:
        return self.dynamic_dim + self.context_dim

    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def generalized_force_dim(self) -> int:
        # Floating base generalized force (6) plus one force per actuated DoF.
        return 6 + self.num_joints

    def validate(self) -> None:
        if self.num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        if self.num_feet <= 0:
            raise ValueError("num_feet must be positive.")
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")


@dataclass(frozen=True)
class ResidualWorldModelCfg:
    """Configuration for the LIFT-style ensemble residual model.

    All dimensions are derived from :attr:`state`; no 25-DoF/T1 size is
    implicit in the network contract.
    """

    state: WorldModelStateSpec = field(default_factory=WorldModelStateSpec)
    prediction_type: PredictionType = "generalized_force"
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    ensemble_size: int = 7
    num_elites: int = 5
    probabilistic: bool = True
    residual_force_frame: ResidualForceFrame = "body_linear"
    model_input_mode: ModelInputMode = "current_privileged_tilt"
    uncertainty_mode: UncertaintyMode = "next_state_diagonal"
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    @property
    def output_dim(self) -> int:
        if self.prediction_type == "generalized_force":
            return self.state.generalized_force_dim
        return self.state.dynamic_dim

    @property
    def model_input_dim(self) -> int:
        # Current privileged state: [base angular velocity, joint q/dq,
        # yaw-invariant root tilt, base linear velocity, root height] + action.
        # This is 61 + 25 = 86 values for T1 and 35 + 12 = 47 for BSRL.
        return (self.state.root_height_slice.stop - 3) + self.state.action_dim

    @property
    def model_orientation_input(self) -> str:
        return "yaw_invariant_tilt"

    @property
    def uncertainty_dim(self) -> int:
        if self.uncertainty_mode == "residual_force_diagonal":
            if self.prediction_type != "generalized_force":
                raise ValueError(
                    "Residual-force uncertainty requires generalized_force prediction mode."
                )
            return self.state.generalized_force_dim
        if self.uncertainty_mode == "paper_privileged_diagonal":
            return self.state.paper_privileged_dim
        if self.prediction_type == "generalized_force":
            return self.state.probabilistic_dynamic_dim
        return self.output_dim

    def validate(self) -> None:
        self.state.validate()
        if self.prediction_type not in ("generalized_force", "state_delta"):
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type!r}.")
        if self.residual_force_frame != "body_linear":
            raise ValueError("Generalized-force residuals must use body-frame linear force.")
        if self.model_input_mode != "current_privileged_tilt":
            raise ValueError(f"Unsupported model input mode: {self.model_input_mode!r}.")
        if self.uncertainty_mode not in (
            "next_state_diagonal",
            "paper_privileged_diagonal",
            "residual_force_diagonal",
        ):
            raise ValueError(f"Unsupported uncertainty mode: {self.uncertainty_mode!r}.")
        if (
            self.uncertainty_mode == "residual_force_diagonal"
            and self.prediction_type != "generalized_force"
        ):
            raise ValueError(
                "residual_force_diagonal requires generalized_force prediction mode."
            )
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain only positive widths.")
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive.")
        if not 0 < self.num_elites <= self.ensemble_size:
            raise ValueError("num_elites must be in [1, ensemble_size].")
        if self.min_logvar >= self.max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar.")
