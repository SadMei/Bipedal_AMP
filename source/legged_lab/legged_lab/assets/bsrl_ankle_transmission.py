"""Ideal reduced parallel transmission, with optional *plant-only* errors.

Coordinates: [env, left/right, pitch/roll], motors long/short. This is NOT
explicit linkage dynamics: it retains the serial URDF's lumped inertia and
cannot reproduce rod inertia, backlash, or compliance modes.
"""
from __future__ import annotations

import torch

from legged_lab.assets.bsrl_closed_ankle_geometry import (
    CADAnkleGeometry, limit_motor_torque, motor_torque_from_joint,
)


def matvec2(matrix, vector):
    # Elementwise 2x2 operations avoid TF32 and tiny batched GEMM overhead.
    return (matrix * vector.unsqueeze(-2)).sum(-1)


class AnkleTransmission:
    def __init__(self, num_envs, device="cpu", dtype=torch.float32, dt=.005,
                 effort=112., speed=26.17993877991494, power=1250.):
        if dt <= 0 or min(effort, speed, power) <= 0:
            raise ValueError("Transmission dt and motor limits must be positive.")
        self.geometry = CADAnkleGeometry(device=device, dtype=dtype)
        self.dt, self.effort, self.speed, self.power = dt, effort, speed, power
        shape = (num_envs, 2, 2)
        self.ratio = torch.ones(shape, device=device, dtype=dtype)
        self.strength = torch.ones_like(self.ratio)
        self.response_time = torch.zeros_like(self.ratio)
        self.response_alpha = torch.ones_like(self.ratio)
        self.motor_torque = torch.zeros_like(self.ratio)
        self.motor_velocity = torch.zeros_like(self.ratio)
        self.requested_motor_torque = torch.zeros_like(self.ratio)
        self.valid = torch.ones(shape[:-1], device=device, dtype=torch.bool)
        self.joint_torque = torch.zeros_like(self.ratio)

    def randomize(self, env_ids=None, ratio_range=(1., 1.), strength_range=(1., 1.),
                  response_time_range=(0., 0.)):
        """Absolute draws, not cumulative; independent motors/legs/environments.

        m_true = ratio * m_nom(q), so J_true = diag(ratio) J_nom. The controller
        continues to invert J_nom. Use J_true for BOTH plant velocity and force
        mapping, preserving virtual work without cancelling the model error.
        """
        for bounds, positive, upper in ((ratio_range, True, float("inf")),
                                        (strength_range, True, 1.),
                                        (response_time_range, False, float("inf"))):
            low, high = bounds
            if not (0 <= low <= high <= upper) or (positive and low == 0):
                raise ValueError(f"Invalid transmission randomization range: {bounds}")
        rows = slice(None) if env_ids is None else env_ids
        for value, bounds in ((self.ratio, ratio_range), (self.strength, strength_range),
                              (self.response_time, response_time_range)):
            value[rows] = torch.empty_like(value[rows]).uniform_(*bounds)
        self.response_alpha[rows] = -torch.expm1(-self.dt / self.response_time[rows].clamp_min(1e-9))

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        # Plant parameters persist across episode resets; only dynamic state clears.
        for value in (self.motor_torque, self.motor_velocity, self.requested_motor_torque,
                      self.joint_torque):
            value[rows] = 0.
        self.valid[rows] = True

    def compute(self, q, dq, requested_joint_torque):
        motor, reachable = self.geometry.inverse_kinematics(q)
        nominal_jac = self.geometry.jacobian(q, motor)
        requested_motor, invertible = motor_torque_from_joint(nominal_jac, requested_joint_torque)
        self.valid = (reachable & invertible & torch.isfinite(q).all(-1)
                      & torch.isfinite(dq).all(-1) & torch.isfinite(requested_joint_torque).all(-1))
        # Never inject NaN into PhysX while the per-rollout audit is collecting
        # a failure. Invalid geometry is reported and stops this run, not hidden.
        jac = torch.where(self.valid[..., None, None], nominal_jac, 0.) * self.ratio.unsqueeze(-1)
        self.requested_motor_torque = torch.where(self.valid.unsqueeze(-1), requested_motor, 0.)
        self.motor_velocity = matvec2(jac, torch.nan_to_num(dq))
        goal = limit_motor_torque(
            self.requested_motor_torque * self.strength, self.motor_velocity,
            self.effort * self.strength, self.speed, self.power * self.strength)
        filtered = self.motor_torque + self.response_alpha * (goal - self.motor_torque)
        # Re-limit AFTER response lag: previous torque cannot keep accelerating
        # an overspeed motor or exceed the current speed-dependent power cap.
        self.motor_torque = limit_motor_torque(
            filtered, self.motor_velocity, self.effort * self.strength,
            self.speed, self.power * self.strength)
        self.motor_torque = torch.where(self.valid.unsqueeze(-1), self.motor_torque, 0.)
        self.joint_torque = matvec2(jac.transpose(-1, -2), self.motor_torque)
        return self.joint_torque
