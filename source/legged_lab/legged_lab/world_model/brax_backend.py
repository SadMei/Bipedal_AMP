"""Brax generalized-coordinate backend for the physics-informed T1 WM.

This module is intentionally independent of the existing MJX backends.  It
uses Brax's generalized-coordinate rigid-body primitives for the unconstrained
``M(q), C(q, dq), G(q)`` dynamics and Brax's semi-implicit Euler integrator.
Contact, friction, and other unmodelled effects are supplied by the learned
31-D generalized-force residual.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from .config import WorldModelStateSpec
from .mjx_backend import MJXBackendCfg, MJXLagrangianPhysics


class BraxLagrangianPhysics(MJXLagrangianPhysics):
    """Paper-style Lagrangian dynamics implemented with Brax 0.10.x.

    The public residual contract is unchanged from the project MJX backend:
    ``[body linear force(3), body torque(3), joint force(25)]``.  The body-frame
    linear force is rotated into world coordinates at every 1 ms substep.

    Brax's MJCF helper fuses fixed bodies before loading.  The stock helper does
    not correctly merge multiple ``<inertial>`` elements in this T1 model, so
    MuJoCo's compiler performs ``fusestatic`` first.  MuJoCo preserves the
    composite mass/inertia exactly and produces the 26-link tree expected by
    Brax without changing the source XML on disk.
    """

    def __init__(self, spec: WorldModelStateSpec, cfg: MJXBackendCfg) -> None:
        spec.validate()
        cfg.validate(spec)
        if cfg.integrator != "euler":
            raise ValueError(
                "brax_lagrangian_euler always uses Brax semi-implicit Euler; "
                f"received integrator={cfg.integrator!r}."
            )
        self.spec = spec
        self.cfg = cfg
        try:
            import brax
            from brax import kinematics
            from brax.generalized import base as generalized_base
            from brax.generalized import dynamics, integrator, mass
            from brax.io import mjcf
            import jax
            import jax.numpy as jnp
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "BraxLagrangianPhysics requires brax==0.10.5, flax, jax, "
                "mujoco, and mujoco-mjx."
            ) from exc

        self._brax_version = brax.__version__
        self._jax = jax
        self._jnp = jnp
        self._mujoco = mujoco
        self._brax_kinematics = kinematics
        self._brax_state = generalized_base.State
        self._brax_dynamics = dynamics
        self._brax_integrator = integrator
        self._brax_mass = mass

        cpu_model = self._compile_fused_model(Path(cfg.xml_path).resolve())
        self._validate_model(cpu_model)
        cpu_model.opt.timestep = cfg.physics_dt
        self._apply_effort_limits(cpu_model)
        if cfg.collision_mode == "feet_spheres_only":
            self._retain_only_foot_sphere_contacts(cpu_model)
        self._configure_cpu_model(cpu_model)
        self._cpu_model = cpu_model
        self._left_foot_body_id = self._body_id(cfg.left_foot_body)
        self._right_foot_body_id = self._body_id(cfg.right_foot_body)
        self._foot_geom_ids = self._find_foot_sphere_geoms()

        # mjcf.load_model internally creates an MJX model, so unsupported
        # collision pairs must already have been disabled above.  The original
        # MjModel is then detached from System to skip Brax contact-feature
        # validation: this backend never calls the constraint/contact solver.
        self.system = mjcf.load_model(cpu_model).replace(mj_model=None)
        if (
            self.system.q_size() != cpu_model.nq
            or self.system.qd_size() != cpu_model.nv
            or self.system.act_size() != cpu_model.nu
        ):
            raise RuntimeError(
                "Brax system dimensions differ from the compiled T1 MJCF: "
                f"brax={(self.system.q_size(), self.system.qd_size(), self.system.act_size())}, "
                f"mujoco={(cpu_model.nq, cpu_model.nv, cpu_model.nu)}."
            )

        frame_to_model = tuple(
            cfg.frame_joint_names.index(name) for name in cfg.joint_names
        )
        model_to_frame = tuple(
            cfg.joint_names.index(name) for name in cfg.frame_joint_names
        )
        self._frame_to_model = jnp.asarray(frame_to_model, dtype=jnp.int32)
        self._model_to_frame = jnp.asarray(model_to_frame, dtype=jnp.int32)
        self._frame_to_model_torch = torch.tensor(frame_to_model, dtype=torch.long)
        self._model_to_frame_torch = torch.tensor(model_to_frame, dtype=torch.long)

        self._offset = jnp.asarray(cfg.default_joint_pos, dtype=jnp.float32)
        self._kp = jnp.asarray(cfg.stiffness, dtype=jnp.float32)
        self._kd = jnp.asarray(cfg.damping, dtype=jnp.float32)
        self._effort = jnp.asarray(cfg.effort_limit, dtype=jnp.float32)
        self._scale = jnp.asarray(cfg.action_scale, dtype=jnp.float32)

        link_names = tuple(self.system.link_names)
        try:
            self._brax_left_foot_link_id = link_names.index(cfg.left_foot_body)
            self._brax_right_foot_link_id = link_names.index(cfg.right_foot_body)
        except ValueError as exc:
            raise KeyError(
                "The fused Brax tree is missing one of the configured foot links: "
                f"{cfg.left_foot_body!r}, {cfg.right_foot_body!r}."
            ) from exc
        self._foot_sphere_local = self._foot_sphere_geometry()

        self._batched_step = jax.jit(jax.vmap(self._single_step))
        self._batched_residual_jacobian = jax.jit(
            jax.vmap(jax.jacfwd(self._single_step, argnums=3))
        )
        self._batched_state_residual_jacobians = jax.jit(
            jax.vmap(jax.jacfwd(self._single_step, argnums=(0, 1, 3)))
        )
        self._batched_step_and_state_residual_jacobians = jax.jit(
            jax.vmap(self._single_step_and_state_residual_jacobians)
        )

    def _compile_fused_model(self, xml_path: Path):
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        compiler.set("fusestatic", "true")
        meshdir = compiler.get("meshdir")
        if meshdir:
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))
        xml = ET.tostring(root, encoding="unicode")
        return self._mujoco.MjModel.from_xml_string(xml)

    def _foot_sphere_geometry(self) -> tuple[tuple, tuple]:
        result = []
        for body_name in (self.cfg.left_foot_body, self.cfg.right_foot_body):
            body_id = self._mujoco.mj_name2id(
                self._cpu_model, self._mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            geom_ids = [
                geom_id
                for geom_id in range(self._cpu_model.ngeom)
                if int(self._cpu_model.geom_bodyid[geom_id]) == body_id
                and self._cpu_model.geom_type[geom_id]
                == self._mujoco.mjtGeom.mjGEOM_SPHERE
                and self._cpu_model.geom_contype[geom_id] != 0
            ]
            if not geom_ids:
                raise ValueError(f"No active foot spheres found for {body_name!r}.")
            local_pos = self._jnp.asarray(
                self._cpu_model.geom_pos[geom_ids], dtype=self._jnp.float32
            )
            radius = self._jnp.asarray(
                self._cpu_model.geom_size[geom_ids, 0], dtype=self._jnp.float32
            )
            result.append((local_pos, radius))
        return tuple(result)

    def _refresh_state(self, qpos, qvel):
        x, xd = self._brax_kinematics.forward(self.system, qpos, qvel)
        state = self._brax_state.init(qpos, qvel, x, xd)
        state = self._brax_dynamics.transform_com(self.system, state)
        return self._brax_mass.matrix_inv(self.system, state, 0)

    def _single_step(self, qpos, qvel, action, residual_force):
        jnp = self._jnp
        target = self._offset + self._scale * action
        state = self._refresh_state(qpos, qvel)

        def substep(_, current):
            motor_torque = self._kp * (
                target - current.q[7:]
            ) - self._kd * current.qd[6:]
            motor_torque = jnp.clip(motor_torque, -self._effort, self._effort)
            root_force_world = self._quat_rotate(
                current.q[3:7], residual_force[:3]
            )
            applied = jnp.concatenate(
                (
                    root_force_world,
                    residual_force[3:6],
                    motor_torque + residual_force[6:],
                )
            )
            smooth = self._brax_dynamics.forward(self.system, current, applied)
            current = self._brax_integrator.integrate(
                self.system,
                current.replace(
                    qf_smooth=smooth,
                    qf_constraint=jnp.zeros_like(smooth),
                ),
            )
            return self._refresh_state(current.q, current.qd)

        state = self._jax.lax.fori_loop(
            0, self.cfg.control_substeps, substep, state
        )
        quat = state.q[3:7]
        base_lin_vel = self._quat_rotate_inverse(quat, state.qd[:3])
        base_ang_vel = state.qd[3:6]
        gravity = self._quat_rotate_inverse(
            quat, jnp.asarray((0.0, 0.0, -1.0), dtype=qpos.dtype)
        )

        foot_heights = []
        contacts = []
        foot_link_ids = (
            self._brax_left_foot_link_id,
            self._brax_right_foot_link_id,
        )
        for link_id, (local_pos, radius) in zip(
            foot_link_ids, self._foot_sphere_local
        ):
            link_quat = state.x.rot[link_id]
            link_pos = state.x.pos[link_id]
            centers = link_pos + self._jax.vmap(
                lambda value: self._quat_rotate(link_quat, value)
            )(local_pos)
            bottom = centers[:, 2] - radius
            contacts.append(
                self._jax.nn.sigmoid(-jnp.min(bottom) / self.cfg.contact_smoothing)
            )
            foot_heights.append(link_pos[2])

        joint_pos_frame = state.q[7:][self._model_to_frame]
        joint_vel_frame = state.qd[6:][self._model_to_frame]
        return jnp.concatenate(
            (
                gravity,
                base_ang_vel,
                joint_pos_frame,
                joint_vel_frame,
                quat,
                base_lin_vel,
                state.q[2:3],
                jnp.stack(contacts),
                jnp.stack(foot_heights),
            )
        )
