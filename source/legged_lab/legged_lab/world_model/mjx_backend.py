from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

from .config import WorldModelStateSpec


CollisionMode = Literal["model", "feet_spheres_only", "feet_mesh_only"]
IntegratorMode = Literal["xml", "euler", "implicitfast"]
ControlMode = Literal["actuator", "qfrc_applied"]


# Isaac/NPZ exposes joints in articulation traversal order, while the MuJoCo
# XML groups them by limb.  World-model frames and policy actions always use
# this Isaac order; controller gains and generalized coordinates use the MJCF
# order declared by ``MJXBackendCfg.joint_names``.
ISAAC_25_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "torso_pitch_joint",
    "left_knee_joint", "right_knee_joint", "torso_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "torso_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
)


@dataclass(frozen=True)
class MJXBackendCfg:
    """Controller and model contract for one differentiable WM step."""

    xml_path: str
    frame_joint_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort_limit: tuple[float, ...]
    action_scale: tuple[float, ...]
    control_mode: ControlMode = "actuator"
    physics_dt: float = 0.001
    integrator: IntegratorMode = "euler"
    control_substeps: int = 20
    collision_mode: CollisionMode = "feet_spheres_only"
    left_foot_body: str = "left_ankle_pitch_Link"
    right_foot_body: str = "right_ankle_pitch_Link"
    contact_smoothing: float = 0.002

    def validate(self, spec: WorldModelStateSpec) -> None:
        vectors = {
            "frame_joint_names": self.frame_joint_names,
            "joint_names": self.joint_names,
            "default_joint_pos": self.default_joint_pos,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "effort_limit": self.effort_limit,
            "action_scale": self.action_scale,
        }
        for name, value in vectors.items():
            if len(value) != spec.num_joints:
                raise ValueError(f"{name} must contain {spec.num_joints} entries, got {len(value)}.")
        if len(set(self.frame_joint_names)) != spec.num_joints:
            raise ValueError("frame_joint_names must contain unique joint names.")
        if len(set(self.joint_names)) != spec.num_joints:
            raise ValueError("joint_names must contain unique joint names.")
        if set(self.frame_joint_names) != set(self.joint_names):
            raise ValueError("frame_joint_names and joint_names must contain the same joints.")
        if self.control_substeps <= 0:
            raise ValueError("control_substeps must be positive.")
        if self.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive.")
        if self.integrator not in ("xml", "euler", "implicitfast"):
            raise ValueError(f"Unsupported integrator {self.integrator!r}.")
        if self.control_mode not in ("actuator", "qfrc_applied"):
            raise ValueError(f"Unsupported control_mode {self.control_mode!r}.")
        if self.contact_smoothing <= 0.0:
            raise ValueError("contact_smoothing must be positive.")
        if self.collision_mode not in ("model", "feet_spheres_only", "feet_mesh_only"):
            raise ValueError(f"Unsupported collision_mode {self.collision_mode!r}.")


def wbtest_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """Current 25-DoF wbtest controller in MuJoCo actuator order.

    The values intentionally live here rather than importing IsaacLab robot
    config objects: this keeps MJX usable in a standalone JAX process and
    makes the exact controller embedded in a WM checkpoint explicit.
    """

    joint_names = (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
        "waist_yaw_joint", "waist_pitch_joint", "torso_pitch_joint",
        "torso_roll_joint", "torso_yaw_joint", "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    )
    default_joint_pos = (
        -0.09, 0.03, 0.03, 0.22, 0.0, -0.13,
        -0.09, -0.03, -0.03, 0.22, 0.0, -0.13,
        0.0, 0.05, 0.0, 0.0, 0.0,
        0.0, 0.18, 0.0, 1.4, 0.0, -0.18, 0.0, 1.4,
    )
    stiffness = (
        100, 100, 100, 600, 100, 100,
        100, 100, 100, 600, 100, 100,
        150, 150, 300, 300, 100,
        40, 40, 40, 40, 40, 40, 40, 40,
    )
    damping = (
        3, 3, 3, 10, 3, 3,
        3, 3, 3, 10, 3, 3,
        5, 5, 5, 5, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
    )
    effort_limit = (
        110, 110, 110, 1000, 36, 100,
        110, 110, 110, 1000, 36, 100,
        110, 110, 100, 100, 60,
        36, 36, 21, 21, 36, 36, 21, 21,
    )
    return MJXBackendCfg(
        xml_path=str(xml_path),
        frame_joint_names=ISAAC_25_JOINT_NAMES,
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        action_scale=(0.25,) * 25,
        collision_mode="model",
    )


def t1_09_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """Legacy T1 ninth-run/model30000 controller.

    Kept for reproducing the historical ``t1_09`` checkpoints.  New WM
    training should use :func:`t1_run39_mjx_cfg` so that the controller and
    plant match the current T1 run-39 baseline.
    """

    base = wbtest_mjx_cfg(xml_path)
    default = list(base.default_joint_pos)
    default[13] = 0.0  # waist_pitch_joint
    stiffness = list(base.stiffness)
    stiffness[14:16] = [150.0, 150.0]  # torso pitch / roll
    effort = list(base.effort_limit)
    effort[5] = effort[11] = 180.0  # ankle pitch
    effort[14] = effort[15] = 180.0  # torso pitch / roll
    return replace(
        base,
        default_joint_pos=tuple(default),
        stiffness=tuple(stiffness),
        effort_limit=tuple(effort),
        integrator="implicitfast",
        collision_mode="feet_spheres_only",
    )


def t1_run39_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """T1 run-39/run-33-baseline controller contract.

    The run-39 Isaac configuration uses the 0809 new-shoe plant, 4x
    decimation at a 5 ms physics step, delayed implicit actuators, and the
    gains/limits below.  Unlike the historical ninth-run WM, knees retain
    their 280 Nm limit. WM contact uses the native shoe meshes against the
    floor only: no foot-point spheres, self-collisions, or non-foot contacts.
    The source XML and all inertial/controller parameters are preserved.
    """

    base = wbtest_mjx_cfg(xml_path)
    return replace(
        base,
        default_joint_pos=(
            -0.09, 0.03, 0.03, 0.22, 0.0, -0.13,
            -0.09, -0.03, -0.03, 0.22, 0.0, -0.13,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.18, 0.0, 1.4, 0.0, -0.18, 0.0, 1.4,
        ),
        stiffness=(
            100.0, 100.0, 100.0, 600.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 600.0, 100.0, 100.0,
            150.0, 150.0, 150.0, 150.0, 100.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        damping=(
            3.0, 3.0, 3.0, 10.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 10.0, 3.0, 3.0,
            5.0, 5.0, 5.0, 5.0, 3.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ),
        effort_limit=(
            110.0, 110.0, 110.0, 280.0, 36.0, 180.0,
            110.0, 110.0, 110.0, 280.0, 36.0, 180.0,
            110.0, 110.0, 180.0, 180.0, 60.0,
            36.0, 36.0, 21.0, 21.0, 36.0, 36.0, 21.0, 21.0,
        ),
        integrator="implicitfast",
        collision_mode="feet_mesh_only",
    )


def t1_wbtest_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """Current WM baseline alias; resolves to the T1 run-39 contract."""

    return t1_run39_mjx_cfg(xml_path)


BSRL_12_FRAME_JOINT_NAMES = (
    "joint_left_hip_yaw", "joint_right_hip_yaw",
    "joint_left_hip_roll", "joint_right_hip_roll",
    "joint_left_hip_pitch", "joint_right_hip_pitch",
    "joint_left_knee_pitch", "joint_right_knee_pitch",
    "joint_left_ankle_pitch", "joint_right_ankle_pitch",
    "joint_left_ankle_roll", "joint_right_ankle_roll",
)


def bsrl_serial_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """BSRL 12-DoF serial-joint nominal controller for framework testing.

    The supplied BSRL MJCF has no MuJoCo actuators, so PD torque is applied to
    joint ``qfrc_applied`` explicitly. This configuration intentionally does
    *not* claim R2 plant parity: its ankle pitch/roll torques are direct joint
    torques and omit the analytical parallel-ankle transmission, motor-space
    saturation/power limit and transmission randomization used by R2.
    """

    joint_names = (
        "joint_right_hip_yaw", "joint_right_hip_roll",
        "joint_right_hip_pitch", "joint_right_knee_pitch",
        "joint_right_ankle_pitch", "joint_right_ankle_roll",
        "joint_left_hip_yaw", "joint_left_hip_roll",
        "joint_left_hip_pitch", "joint_left_knee_pitch",
        "joint_left_ankle_pitch", "joint_left_ankle_roll",
    )
    return MJXBackendCfg(
        xml_path=str(xml_path),
        frame_joint_names=BSRL_12_FRAME_JOINT_NAMES,
        joint_names=joint_names,
        default_joint_pos=(
            0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
            0.0, 0.0, -0.2, 0.4, -0.2, 0.0,
        ),
        stiffness=(
            200.0, 200.0, 300.0, 300.0, 120.0, 40.0,
            200.0, 200.0, 300.0, 300.0, 120.0, 40.0,
        ),
        damping=(2.0, 2.0, 2.0, 2.0, 0.0, 0.0) * 2,
        effort_limit=(112.0, 216.0, 216.0, 216.0, 112.0, 112.0) * 2,
        action_scale=(0.1,) * 12,
        control_mode="qfrc_applied",
        physics_dt=0.005,
        integrator="implicitfast",
        control_substeps=4,
        collision_mode="feet_mesh_only",
        left_foot_body="link_left_ankle_roll",
        right_foot_body="link_right_ankle_roll",
    )


def bsrl_r2_mjx_cfg(xml_path: str | Path) -> MJXBackendCfg:
    """Reject accidental use of serial MJX as the analytical R2 plant."""

    raise NotImplementedError(
        "BSRL R2 analytical parallel-ankle plant requires a dedicated MJX implementation of ReducedAnkleActuator/"
        "AnkleTransmission (motor-space geometry, effort/speed/power limits and "
        "transmission DR). Use bsrl_serial_mjx_cfg only for WM interface tests; "
        "it is not an R2 sim2real nominal model."
    )


class _DifferentiableMJXStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qpos, qvel, action, residual_force, backend):
        qpos_jax = backend._torch_to_jax(qpos)
        qvel_jax = backend._torch_to_jax(qvel)
        action_jax = backend._torch_to_jax(action)
        residual_jax = backend._torch_to_jax(residual_force)
        output = backend._batched_step(qpos_jax, qvel_jax, action_jax, residual_jax)
        ctx.save_for_backward(qpos.detach(), qvel.detach(), action.detach(), residual_force.detach())
        ctx.backend = backend
        return backend._jax_to_torch(output)

    @staticmethod
    def backward(ctx, grad_output):
        qpos, qvel, action, residual_force = ctx.saved_tensors
        arguments = (
            ctx.backend._torch_to_jax(qpos.contiguous()),
            ctx.backend._torch_to_jax(qvel.contiguous()),
            ctx.backend._torch_to_jax(action.contiguous()),
            ctx.backend._torch_to_jax(residual_force.contiguous()),
        )
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            jac_qpos, jac_qvel, jac_residual = ctx.backend._batched_state_residual_jacobians(
                *arguments
            )
            jac_qpos = ctx.backend._jax_to_torch(jac_qpos)
            jac_qvel = ctx.backend._jax_to_torch(jac_qvel)
            grad_qpos = torch.einsum("bo,boj->bj", grad_output, jac_qpos)
            grad_qvel = torch.einsum("bo,boj->bj", grad_output, jac_qvel)
        else:
            jac_residual = ctx.backend._batched_residual_jacobian(*arguments)
            grad_qpos = grad_qvel = None
        jac_residual = ctx.backend._jax_to_torch(jac_residual)
        grad_residual = torch.einsum("bo,boj->bj", grad_output, jac_residual)
        # Actions are recorded supervision.  qpos/qvel gradients are required
        # for full autoregressive credit assignment after the first step.
        return grad_qpos, grad_qvel, None, grad_residual, None


class _DifferentiableMJXStepWithJacobians(torch.autograd.Function):
    """MJX step that exposes and reuses local Jacobians for force NLL.

    The residual Jacobian is a detached auxiliary output used to propagate a
    force covariance.  The same Jacobians are cached for backward, avoiding a
    second MJX differentiation pass during probabilistic force training.
    """

    @staticmethod
    def forward(ctx, qpos, qvel, action, residual_force, backend):
        outputs = backend._batched_step_and_state_residual_jacobians(
            backend._torch_to_jax(qpos),
            backend._torch_to_jax(qvel),
            backend._torch_to_jax(action),
            backend._torch_to_jax(residual_force),
        )
        output, jac_qpos, jac_qvel, jac_residual = (
            backend._jax_to_torch(value) for value in outputs
        )
        ctx.save_for_backward(jac_qpos, jac_qvel, jac_residual)
        ctx.mark_non_differentiable(jac_residual)
        return output, jac_residual

    @staticmethod
    def backward(ctx, grad_output, _grad_jacobian):
        jac_qpos, jac_qvel, jac_residual = ctx.saved_tensors
        grad_qpos = torch.einsum("bo,boj->bj", grad_output, jac_qpos)
        grad_qvel = torch.einsum("bo,boj->bj", grad_output, jac_qvel)
        grad_residual = torch.einsum("bo,boj->bj", grad_output, jac_residual)
        return grad_qpos, grad_qvel, None, grad_residual, None


class MJXNominalPhysics:
    """Differentiable MuJoCo-MJX nominal physics for a ``6 + N`` force WM.

    The residual network predicts ``[body linear force(3), body torque(3),
    joint force(N)]``. At the start of each control step, the linear
    force is rotated into world coordinates using the current root quaternion;
    that world-frame force is then held in ``qfrc_applied`` for all physics
    substeps. A custom
    PyTorch autograd bridge contracts forward-mode JAX Jacobians, so next-state loss reaches the
    PyTorch residual network instead of stopping at a NumPy/JAX boundary.
    """

    def __init__(self, spec: WorldModelStateSpec, cfg: MJXBackendCfg) -> None:
        spec.validate()
        cfg.validate(spec)
        self.spec = spec
        self.cfg = cfg
        try:
            import jax
            import jax.numpy as jnp
            import mujoco
            from mujoco import mjx
        except ImportError as exc:
            raise ImportError(
                "MJXNominalPhysics requires jax, mujoco and mujoco-mjx. "
                "Install a JAX build matching the training device."
            ) from exc

        self._jax = jax
        self._jnp = jnp
        self._mujoco = mujoco
        self._mjx = mjx
        cpu_model = mujoco.MjModel.from_xml_path(str(Path(cfg.xml_path).resolve()))
        self._validate_model(cpu_model)
        # Apply the selected WM controller contract before converting the
        # model to MJX.
        cpu_model.opt.timestep = cfg.physics_dt
        if cfg.integrator != "xml":
            integrator = {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }[cfg.integrator]
            cpu_model.opt.integrator = integrator
        self._apply_effort_limits(cpu_model)
        if cfg.collision_mode == "feet_spheres_only":
            self._retain_only_foot_sphere_contacts(cpu_model)
        elif cfg.collision_mode == "feet_mesh_only":
            self._retain_only_foot_mesh_contacts(cpu_model)
        self._configure_cpu_model(cpu_model)
        try:
            self.model = mjx.put_model(cpu_model)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"MJX model conversion failed with collision_mode={cfg.collision_mode!r}: {exc}. "
                "For shoe meshes use 'feet_mesh_only'; 'feet_spheres_only' requires foot spheres. "
                "The unfiltered 'model' mode requires an MJX-compatible MJCF."
            ) from exc
        self.data = mjx.make_data(self.model)
        self._cpu_model = cpu_model
        self._left_foot_body_id = self._body_id(cfg.left_foot_body)
        self._right_foot_body_id = self._body_id(cfg.right_foot_body)
        self._foot_geom_ids = self._find_foot_contact_geoms()
        self._prepare_foot_contact_geometry()

        # Gather a frame/policy vector into MJCF order and gather an MJCF
        # output back into frame order.  Keep both as explicit arrays so the
        # same mapping is used for q, dq, action and residual joint force.
        frame_to_model = tuple(cfg.frame_joint_names.index(name) for name in cfg.joint_names)
        model_to_frame = tuple(cfg.joint_names.index(name) for name in cfg.frame_joint_names)
        self._frame_to_model = jnp.asarray(frame_to_model, dtype=jnp.int32)
        self._model_to_frame = jnp.asarray(model_to_frame, dtype=jnp.int32)
        self._frame_to_model_torch = torch.tensor(frame_to_model, dtype=torch.long)
        self._model_to_frame_torch = torch.tensor(model_to_frame, dtype=torch.long)

        self._offset = jnp.asarray(cfg.default_joint_pos, dtype=jnp.float32)
        self._kp = jnp.asarray(cfg.stiffness, dtype=jnp.float32)
        self._kd = jnp.asarray(cfg.damping, dtype=jnp.float32)
        self._effort = jnp.asarray(cfg.effort_limit, dtype=jnp.float32)
        self._scale = jnp.asarray(cfg.action_scale, dtype=jnp.float32)
        self._batched_step = jax.jit(jax.vmap(self._single_step))
        # MJX's contact solver contains a data-dependent while_loop, for which
        # JAX reverse-mode VJP is unavailable. The residual has only nv=31
        # inputs, so forward-mode is both valid and bounded. Vmap keeps samples
        # independent and avoids constructing a [batch, batch] Jacobian.
        self._batched_residual_jacobian = jax.jit(
            jax.vmap(jax.jacfwd(self._single_step, argnums=3))
        )
        self._batched_state_residual_jacobians = jax.jit(
            jax.vmap(jax.jacfwd(self._single_step, argnums=(0, 1, 3)))
        )
        self._batched_step_and_state_residual_jacobians = jax.jit(
            jax.vmap(self._single_step_and_state_residual_jacobians)
        )

    @property
    def generalized_force_dim(self) -> int:
        return self._cpu_model.nv

    def validate_checkpoint_metadata(self, metadata: dict) -> None:
        """Prevent old foot-sphere residuals from silently using mesh dynamics."""
        saved_mode = metadata.get("collision_mode")
        if saved_mode != self.cfg.collision_mode and (
            saved_mode is not None or self.cfg.collision_mode == "feet_mesh_only"
        ):
            raise ValueError(
                f"Checkpoint collision_mode={saved_mode!r} does not match {self.cfg.collision_mode!r}. "
                "Train the mesh-contact WM from scratch; legacy sphere checkpoints require their original backend."
            )
        saved_xml = metadata.get("xml")
        if saved_xml is not None and Path(saved_xml).resolve() != Path(self.cfg.xml_path).resolve():
            raise ValueError("Checkpoint XML does not match the nominal physics XML.")

    def __call__(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        generalized_residual_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.shape[-2:] != (self.spec.history_length, self.spec.frame_dim):
            raise ValueError("history does not match the configured world-model state spec.")
        if action.shape != (history.shape[0], self.spec.action_dim):
            raise ValueError("action and history batch dimensions do not match.")
        if generalized_residual_force is None:
            generalized_residual_force = action.new_zeros((action.shape[0], self._cpu_model.nv))
        if generalized_residual_force.shape != (action.shape[0], self._cpu_model.nv):
            raise ValueError(
                f"generalized_residual_force must be [batch, {self._cpu_model.nv}], "
                f"got {tuple(generalized_residual_force.shape)}."
            )
        self._validate_device(history.device)
        last = history[:, -1, :]
        qpos, qvel = self._frame_to_qpos_qvel(last)
        frame_to_model = self._frame_to_model_torch.to(action.device)
        action_model = action.index_select(-1, frame_to_model)
        residual_model = self.residual_force_to_mjcf(
            generalized_residual_force, qpos[:, 3:7]
        )
        dynamic = _DifferentiableMJXStep.apply(
            qpos.contiguous(),
            qvel.contiguous(),
            action_model.contiguous(),
            residual_model.contiguous(),
            self,
        )
        # Reference q/dq are exogenous. They stay fixed for this one-step
        # physics operator; a motion-clock provider must advance them during
        # multi-step synthetic rollouts. The previous-action context does
        # advance here.
        context_start = self.spec.dynamic_dim
        reference_end = context_start + 2 * self.spec.num_joints
        reference = last[:, context_start:reference_end]
        return torch.cat((dynamic, reference, action), dim=-1)

    def residual_force_to_mjcf(
        self,
        generalized_residual_force: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        """Map the WM residual contract to MuJoCo generalized coordinates.

        The network's first three values are a root-frame linear force. MuJoCo
        free-joint translation uses world axes, while its rotational DoFs use
        local/body axes. Therefore only the linear force is rotated. Joint
        forces are also reordered from policy/frame order to MJCF order.
        """

        if generalized_residual_force.ndim != 2 or generalized_residual_force.shape[-1] != self._cpu_model.nv:
            raise ValueError(
                f"generalized_residual_force must be [batch, {self._cpu_model.nv}], "
                f"got {tuple(generalized_residual_force.shape)}."
            )
        if root_quat_wxyz.shape != (generalized_residual_force.shape[0], 4):
            raise ValueError(
                "root_quat_wxyz must have shape "
                f"({generalized_residual_force.shape[0]}, 4), got {tuple(root_quat_wxyz.shape)}."
            )
        frame_to_model = self._frame_to_model_torch.to(generalized_residual_force.device)
        linear_force_w = self._torch_quat_rotate(
            root_quat_wxyz, generalized_residual_force[:, :3]
        )
        return torch.cat(
            (
                linear_force_w,
                generalized_residual_force[:, 3:6],
                generalized_residual_force[:, 6:].index_select(-1, frame_to_model),
            ),
            dim=-1,
        )

    def step_with_residual_jacobian(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        generalized_residual_force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a differentiable next frame and detached force Jacobian."""

        if history.shape[-2:] != (self.spec.history_length, self.spec.frame_dim):
            raise ValueError("history does not match the configured world-model state spec.")
        if action.shape != (history.shape[0], self.spec.action_dim):
            raise ValueError("action and history batch dimensions do not match.")
        if generalized_residual_force.shape != (
            history.shape[0],
            self._cpu_model.nv,
        ):
            raise ValueError(
                "generalized_residual_force must have shape "
                f"({history.shape[0]}, {self._cpu_model.nv})."
            )
        self._validate_device(history.device)
        last = history[:, -1, :]
        qpos, qvel = self._frame_to_qpos_qvel(last)
        frame_to_model = self._frame_to_model_torch.to(action.device)
        action_model = action.index_select(-1, frame_to_model)
        residual_model = self.residual_force_to_mjcf(
            generalized_residual_force, qpos[:, 3:7]
        )
        dynamic, jacobian_model = _DifferentiableMJXStepWithJacobians.apply(
            qpos.contiguous(),
            qvel.contiguous(),
            action_model.contiguous(),
            residual_model.contiguous(),
            self,
        )
        jacobian_network = self._map_residual_jacobian_to_network(
            jacobian_model, qpos[:, 3:7], generalized_residual_force
        )
        context_start = self.spec.dynamic_dim
        reference_end = context_start + 2 * self.spec.num_joints
        reference = last[:, context_start:reference_end]
        frame = torch.cat((dynamic, reference, action), dim=-1)
        return frame, jacobian_network

    @torch.no_grad()
    def residual_jacobian(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        generalized_residual_force: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``d dynamic_next / d residual_network_output``.

        MJX differentiates with respect to its own generalized-force order.
        The public WM contract instead uses body-frame root linear force and
        policy/frame joint order, so the final matrix multiplication applies
        the same rotation and permutation as :meth:`residual_force_to_mjcf`.
        The Jacobian is intentionally detached: state-mean gradients still
        pass through the normal autograd bridge, while this local linearization
        only transports the learned force covariance into state coordinates.
        """

        if history.shape[-2:] != (self.spec.history_length, self.spec.frame_dim):
            raise ValueError("history does not match the configured world-model state spec.")
        if action.shape != (history.shape[0], self.spec.action_dim):
            raise ValueError("action and history batch dimensions do not match.")
        if generalized_residual_force.shape != (
            history.shape[0],
            self._cpu_model.nv,
        ):
            raise ValueError(
                "generalized_residual_force must have shape "
                f"({history.shape[0]}, {self._cpu_model.nv})."
            )
        self._validate_device(history.device)
        last = history[:, -1, :]
        qpos, qvel = self._frame_to_qpos_qvel(last)
        frame_to_model = self._frame_to_model_torch.to(action.device)
        action_model = action.index_select(-1, frame_to_model)
        residual_model = self.residual_force_to_mjcf(
            generalized_residual_force, qpos[:, 3:7]
        )
        jacobian_model = self._jax_to_torch(
            self._batched_residual_jacobian(
                self._torch_to_jax(qpos.contiguous()),
                self._torch_to_jax(qvel.contiguous()),
                self._torch_to_jax(action_model.contiguous()),
                self._torch_to_jax(residual_model.contiguous()),
            )
        )

        return self._map_residual_jacobian_to_network(
            jacobian_model, qpos[:, 3:7], generalized_residual_force
        )

    def _map_residual_jacobian_to_network(
        self,
        jacobian_model: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        generalized_residual_force: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = generalized_residual_force.shape[0]
        mapping = generalized_residual_force.new_zeros(
            (batch_size, self._cpu_model.nv, self._cpu_model.nv)
        )
        basis_body = generalized_residual_force.new_zeros((batch_size, 3, 3))
        basis_body[:, torch.arange(3), torch.arange(3)] = 1.0
        quat = root_quat_wxyz.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 4)
        rotated_basis = self._torch_quat_rotate(
            quat, basis_body.reshape(-1, 3)
        ).reshape(batch_size, 3, 3)
        # rotated_basis[:, input_body_axis, output_world_axis]
        mapping[:, :3, :3] = rotated_basis.transpose(1, 2)
        mapping[:, 3:6, 3:6] = torch.eye(
            3, device=mapping.device, dtype=mapping.dtype
        )
        for model_index, frame_index in enumerate(self._frame_to_model_torch.tolist()):
            mapping[:, 6 + model_index, 6 + frame_index] = 1.0
        return torch.einsum("bos,bsf->bof", jacobian_model, mapping)

    def _single_step_and_state_residual_jacobians(
        self, qpos, qvel, action, residual_force
    ):
        output = self._single_step(qpos, qvel, action, residual_force)
        jac_qpos, jac_qvel, jac_residual = self._jax.jacfwd(
            self._single_step, argnums=(0, 1, 3)
        )(qpos, qvel, action, residual_force)
        return output, jac_qpos, jac_qvel, jac_residual

    def _single_step(self, qpos, qvel, action, residual_force):
        jnp = self._jnp
        target = self._offset + self._scale * action
        data = self.data.replace(qpos=qpos, qvel=qvel, qfrc_applied=residual_force)

        def substep(_, current):
            torque = self._kp * (target - current.qpos[7:]) - self._kd * current.qvel[6:]
            torque = jnp.clip(torque, -self._effort, self._effort)
            if self.cfg.control_mode == "actuator":
                current = current.replace(ctrl=torque, qfrc_applied=residual_force)
            else:
                applied = residual_force.at[6:].add(torque)
                current = current.replace(qfrc_applied=applied)
            return self._mjx.step(self.model, current)

        data = self._jax.lax.fori_loop(0, self.cfg.control_substeps, substep, data)
        if self.cfg.collision_mode != "feet_spheres_only":
            # mjx.step advances qpos after computing Cartesian fields. Mesh
            # clearance/foot height must describe the final, not penultimate,
            # state. Keep the historical sphere-only timing for reproduction.
            data = self._mjx.kinematics(self.model, data)
        quat = data.qpos[3:7]
        base_lin_vel = self._quat_rotate_inverse(quat, data.qvel[:3])
        # MuJoCo/Brax free-joint convention: root translation velocity is in
        # world coordinates, while the three rotational DoFs are local/body
        # angular velocity (same convention as Isaac root_ang_vel_b).
        base_ang_vel = data.qvel[3:6]
        gravity = self._quat_rotate_inverse(quat, jnp.asarray((0.0, 0.0, -1.0), dtype=qpos.dtype))
        foot_height = jnp.stack(
            (data.xpos[self._left_foot_body_id, 2], data.xpos[self._right_foot_body_id, 2])
        )
        contact = self._jax.nn.sigmoid(-self._foot_clearance(data) / self.cfg.contact_smoothing)
        joint_pos_frame = data.qpos[7:][self._model_to_frame]
        joint_vel_frame = data.qvel[6:][self._model_to_frame]
        return jnp.concatenate(
            (
                gravity,
                base_ang_vel,
                joint_pos_frame,
                joint_vel_frame,
                quat,
                base_lin_vel,
                data.qpos[2:3],
                contact,
                foot_height,
            )
        )

    def _frame_to_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.spec.num_joints
        q = frame[:, 6 : 6 + n]
        dq = frame[:, 6 + n : 6 + 2 * n]
        frame_to_model = self._frame_to_model_torch.to(frame.device)
        q = q.index_select(-1, frame_to_model)
        dq = dq.index_select(-1, frame_to_model)
        quat_start = 6 + 2 * n
        quat = frame[:, quat_start : quat_start + 4]
        lin_b = frame[:, quat_start + 4 : quat_start + 7]
        height = frame[:, quat_start + 7 : quat_start + 8]
        ang_b = frame[:, 3:6]
        root_xy = torch.zeros((frame.shape[0], 2), device=frame.device, dtype=frame.dtype)
        root_pos = torch.cat((root_xy, height), dim=-1)
        lin_w = self._torch_quat_rotate(quat, lin_b)
        return torch.cat((root_pos, quat, q), dim=-1), torch.cat((lin_w, ang_b, dq), dim=-1)

    def _validate_model(self, model) -> None:
        expected_nu = self.spec.num_joints if self.cfg.control_mode == "actuator" else 0
        expected = (7 + self.spec.num_joints, 6 + self.spec.num_joints, expected_nu)
        actual = (model.nq, model.nv, model.nu)
        if actual != expected:
            raise ValueError(f"MJCF (nq, nv, nu)={actual}, expected {expected}.")
        names = tuple(
            self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(1, model.njnt)
        )
        if names != self.cfg.joint_names:
            raise ValueError(f"MJCF joint order does not match controller order:\n{names}\n!=\n{self.cfg.joint_names}")

    def _retain_only_foot_sphere_contacts(self, model) -> None:
        foot_names = (self.cfg.left_foot_body, self.cfg.right_foot_body)
        for geom_id in range(model.ngeom):
            geom_name = self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_name = self._mujoco.mj_id2name(
                model, self._mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            )
            is_floor = geom_name == "floor"
            is_foot_sphere = (
                body_name in foot_names
                and model.geom_type[geom_id] == self._mujoco.mjtGeom.mjGEOM_SPHERE
            )
            model.geom_contype[geom_id] = 1 if is_floor or is_foot_sphere else 0
            model.geom_conaffinity[geom_id] = 1 if is_foot_sphere else 0

    def _retain_only_foot_mesh_contacts(self, model) -> None:
        """Enable only native collision shoe meshes against the static floor.

        Visual meshes (both collision masks zero) must stay disabled. Distinct
        floor/foot bits also disable left-right shoe self-collision. Explicit
        MJCF pairs bypass these masks, so reject them instead of silently
        retaining collisions outside this mode's contract.
        """
        if model.npair:
            raise ValueError("feet_mesh_only requires an MJCF without explicit contact pairs.")
        floor_id = self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if (
            floor_id < 0
            or model.geom_type[floor_id] != self._mujoco.mjtGeom.mjGEOM_PLANE
            or model.body_weldid[model.geom_bodyid[floor_id]] != 0
        ):
            raise ValueError("feet_mesh_only requires a static plane geom named 'floor'.")
        foot_geoms = []
        for foot_name in (self.cfg.left_foot_body, self.cfg.right_foot_body):
            body_id = self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_BODY, foot_name)
            ids = [
                g for g in range(model.ngeom)
                if model.geom_bodyid[g] == body_id
                and model.geom_type[g] == self._mujoco.mjtGeom.mjGEOM_MESH
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            ]
            if not ids:
                raise ValueError(f"feet_mesh_only found no active collision mesh on {foot_name!r}.")
            foot_geoms.extend(ids)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        model.geom_contype[floor_id] = 1
        model.geom_conaffinity[floor_id] = 2
        model.geom_contype[foot_geoms] = 2
        model.geom_conaffinity[foot_geoms] = 1

    def _prepare_foot_contact_geometry(self) -> None:
        """Cache compiled mesh vertices in geom-local coordinates, not STL coordinates."""
        import numpy as np

        model = self._cpu_model
        self._floor_geom_id = self._mujoco.mj_name2id(
            model, self._mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if (
            self._floor_geom_id < 0
            or model.geom_type[self._floor_geom_id] != self._mujoco.mjtGeom.mjGEOM_PLANE
        ):
            raise ValueError("Foot contact clearance requires a plane geom named 'floor'.")
        geometry = []
        cpu_geometry = []
        for geom_ids in self._foot_geom_ids:
            foot = []
            cpu_foot = []
            for geom_id in geom_ids.tolist():
                geom_type = model.geom_type[geom_id]
                if geom_type == self._mujoco.mjtGeom.mjGEOM_SPHERE:
                    vertices = np.zeros((1, 3), dtype=np.float32)
                    radius = float(model.geom_size[geom_id, 0])
                elif geom_type == self._mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = model.geom_dataid[geom_id]
                    start = model.mesh_vertadr[mesh_id]
                    count = model.mesh_vertnum[mesh_id]
                    vertices = model.mesh_vert[start : start + count].copy()
                    radius = 0.0
                else:
                    raise ValueError(f"Foot clearance supports sphere/mesh geoms, got type {geom_type}.")
                foot.append((geom_id, self._jnp.asarray(vertices), radius))
                cpu_foot.append((geom_id, vertices, radius))
            geometry.append(tuple(foot))
            cpu_geometry.append(tuple(cpu_foot))
        self._foot_contact_geometry = tuple(geometry)
        self._foot_contact_geometry_cpu = tuple(cpu_geometry)

    def foot_clearance_cpu(self, data):
        """CPU equivalent for deployment initialization and diagnostic checks."""
        import numpy as np

        floor_pos = data.geom_xpos[self._floor_geom_id]
        normal = data.geom_xmat[self._floor_geom_id].reshape(3, 3)[:, 2]
        return np.asarray([
            min(
                np.dot(data.geom_xpos[g] - floor_pos, normal)
                + np.min(vertices @ (data.geom_xmat[g].reshape(3, 3).T @ normal))
                - radius
                for g, vertices, radius in foot
            )
            for foot in self._foot_contact_geometry_cpu
        ])

    def _foot_clearance(self, data):
        """Signed closest surface distance to the floor; NOT a contact-force estimate.

        The mesh support minimum is exact for the convex collision hull and
        remains piecewise smooth at support-vertex switches. Sigmoid only
        softens the contact output, not the actual MJX contact solver.
        """
        jnp = self._jnp
        floor_pos = data.geom_xpos[self._floor_geom_id]
        normal = data.geom_xmat[self._floor_geom_id, :, 2]
        distances = []
        for foot in self._foot_contact_geometry:
            geom_distances = []
            for geom_id, vertices, radius in foot:
                normal_local = data.geom_xmat[geom_id].T @ normal
                center_distance = jnp.dot(data.geom_xpos[geom_id] - floor_pos, normal)
                geom_distances.append(center_distance + jnp.min(vertices @ normal_local) - radius)
            distances.append(jnp.min(jnp.stack(geom_distances)))
        return jnp.stack(distances)

    def _apply_effort_limits(self, model) -> None:
        for joint_name, limit in zip(self.cfg.joint_names, self.cfg.effort_limit):
            joint_id = self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise KeyError(f"Missing joint for effort limit: {joint_name!r}.")
            model.jnt_actfrclimited[joint_id] = True
            model.jnt_actfrcrange[joint_id] = (-limit, limit)
            if self.cfg.control_mode == "qfrc_applied":
                continue
            actuator_name = f"motor_{joint_name.removesuffix('_joint')}"
            actuator_id = self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise KeyError(
                    f"Missing actuator for effort limit: {actuator_name!r}."
                )
            model.actuator_ctrllimited[actuator_id] = True
            model.actuator_ctrlrange[actuator_id] = (-limit, limit)

    def _configure_cpu_model(self, model) -> None:
        """Subclass hook applied immediately before conversion to MJX."""


    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(self._cpu_model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"Body {name!r} is missing from {self.cfg.xml_path}.")
        return int(body_id)

    def _find_foot_contact_geoms(self) -> tuple:
        """Return active contact geoms used to derive smooth foot contact.

        The legacy filtered mode intentionally selects only foot spheres.  The
        active T1-39 0809 new-shoe model uses native collision shoe meshes
        through ``collision_mode="feet_mesh_only"`` (visual meshes stay off).
        """
        result = []
        for body_id in (self._left_foot_body_id, self._right_foot_body_id):
            geom_ids = [
                geom_id
                for geom_id in range(self._cpu_model.ngeom)
                if int(self._cpu_model.geom_bodyid[geom_id]) == body_id
                and self._cpu_model.geom_contype[geom_id] != 0
            ]
            if self.cfg.collision_mode == "feet_spheres_only":
                geom_ids = [
                    geom_id
                    for geom_id in geom_ids
                    if self._cpu_model.geom_type[geom_id] == self._mujoco.mjtGeom.mjGEOM_SPHERE
                ]
            if not geom_ids:
                mode = "active sphere" if self.cfg.collision_mode == "feet_spheres_only" else "active"
                raise ValueError(f"No {mode} collision geoms found on foot body id {body_id}.")
            result.append(self._jnp.asarray(geom_ids, dtype=self._jnp.int32))
        return tuple(result)

    # Kept as a private compatibility alias for downstream diagnostics.
    def _find_foot_sphere_geoms(self) -> tuple:
        return self._find_foot_contact_geoms()

    def _quat_rotate_inverse(self, quat, vector):
        return self._quat_rotate(self._quat_conjugate(quat), vector)

    def _quat_rotate(self, quat, vector):
        w = quat[0]
        xyz = quat[1:]
        return vector + 2.0 * (
            w * self._jnp.cross(xyz, vector) + self._jnp.cross(xyz, self._jnp.cross(xyz, vector))
        )

    @staticmethod
    def _quat_conjugate(quat):
        return quat.at[1:].multiply(-1.0)

    @staticmethod
    def _torch_quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz = quat[:, 1:]
        return vector + 2.0 * (
            quat[:, :1] * torch.cross(xyz, vector, dim=-1)
            + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
        )

    def _torch_to_jax(self, value: torch.Tensor):
        return self._jax.dlpack.from_dlpack(value.detach())

    @staticmethod
    def _jax_to_torch(value) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(value)

    def _validate_device(self, device: torch.device) -> None:
        jax_platform = self._jax.devices()[0].platform
        if device.type == "cuda" and jax_platform != "gpu":
            raise RuntimeError(
                "PyTorch tensors are on CUDA but JAX is CPU-only. Install the matching CUDA jaxlib; "
                "silently copying through CPU would make WM training unusably slow."
            )


class MJXLagrangianPhysics(MJXNominalPhysics):
    """Paper-style differentiable ``M,C,G`` dynamics implemented with MJX.

    This backend never invokes MuJoCo's contact or constraint solver. Every
    substep computes the mass matrix and RNE bias force, adds PD and learned
    residual forces, solves acceleration, and advances with semi-implicit
    Euler. Contact and dissipation must therefore be learned by the residual.

    The project-specific force contract is retained: root linear force is in
    body axes and is rotated using the current substep quaternion.
    """

    def _configure_cpu_model(self, model) -> None:
        model.opt.integrator = self._mujoco.mjtIntegrator.mjINT_EULER
        model.opt.disableflags |= int(
            self._mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        )

    def residual_force_to_mjcf(
        self,
        generalized_residual_force: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        """Reorder joints while retaining body-frame root force."""

        del root_quat_wxyz
        if (
            generalized_residual_force.ndim != 2
            or generalized_residual_force.shape[-1] != self._cpu_model.nv
        ):
            raise ValueError(
                f"generalized_residual_force must be [batch, {self._cpu_model.nv}], "
                f"got {tuple(generalized_residual_force.shape)}."
            )
        frame_to_model = self._frame_to_model_torch.to(
            generalized_residual_force.device
        )
        return torch.cat(
            (
                generalized_residual_force[:, :6],
                generalized_residual_force[:, 6:].index_select(
                    -1, frame_to_model
                ),
            ),
            dim=-1,
        )

    def _map_residual_jacobian_to_network(
        self,
        jacobian_model: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        generalized_residual_force: torch.Tensor,
    ) -> torch.Tensor:
        del root_quat_wxyz
        batch_size = generalized_residual_force.shape[0]
        mapping = generalized_residual_force.new_zeros(
            (batch_size, self._cpu_model.nv, self._cpu_model.nv)
        )
        mapping[:, :6, :6] = torch.eye(
            6, device=mapping.device, dtype=mapping.dtype
        )
        for model_index, frame_index in enumerate(
            self._frame_to_model_torch.tolist()
        ):
            mapping[:, 6 + model_index, 6 + frame_index] = 1.0
        return torch.einsum("bos,bsf->bof", jacobian_model, mapping)

    def _single_step(self, qpos, qvel, action, residual_force):
        jnp = self._jnp
        target = self._offset + self._scale * action
        data = self.data.replace(qpos=qpos, qvel=qvel)

        def substep(_, current):
            current = self._mjx.kinematics(self.model, current)
            current = self._mjx.com_pos(self.model, current)
            current = self._mjx.crb(self.model, current)
            current = self._mjx.com_vel(self.model, current)
            current = self._mjx.rne(self.model, current)

            motor_torque = self._kp * (
                target - current.qpos[7:]
            ) - self._kd * current.qvel[6:]
            motor_torque = jnp.clip(motor_torque, -self._effort, self._effort)
            root_force_world = self._quat_rotate(
                current.qpos[3:7], residual_force[:3]
            )
            applied = jnp.concatenate(
                (
                    root_force_world,
                    residual_force[3:6],
                    motor_torque + residual_force[6:],
                )
            )
            mass = self._mjx.full_m(self.model, current)
            qacc = jnp.linalg.solve(mass, applied - current.qfrc_bias)
            return self._mjx.euler(self.model, current.replace(qacc=qacc))

        data = self._jax.lax.fori_loop(0, self.cfg.control_substeps, substep, data)
        # Euler advances qpos/qvel but derived Cartesian fields are stale.
        data = self._mjx.kinematics(self.model, data)
        quat = data.qpos[3:7]
        base_lin_vel = self._quat_rotate_inverse(quat, data.qvel[:3])
        base_ang_vel = data.qvel[3:6]
        gravity = self._quat_rotate_inverse(
            quat, jnp.asarray((0.0, 0.0, -1.0), dtype=qpos.dtype)
        )
        foot_height = jnp.stack(
            (
                data.xpos[self._left_foot_body_id, 2],
                data.xpos[self._right_foot_body_id, 2],
            )
        )
        contact = self._jax.nn.sigmoid(-self._foot_clearance(data) / self.cfg.contact_smoothing)
        joint_pos_frame = data.qpos[7:][self._model_to_frame]
        joint_vel_frame = data.qvel[6:][self._model_to_frame]
        return jnp.concatenate(
            (
                gravity,
                base_ang_vel,
                joint_pos_frame,
                joint_vel_frame,
                quat,
                base_lin_vel,
                data.qpos[2:3],
                contact,
                foot_height,
            )
        )
