"""Batched CAD-zero BSRL ankle geometry (metres/radians, pitch then roll).

Points copied from parallel_ankle's user-confirmed 20260914_cad_zero_connected
CAD partition. No encoder recalibration, and no dependency on that workspace.
The motor positive axis is URDF -Y. Side order is left, right; long, short.
"""
from __future__ import annotations

import torch

REVISION = "20260914_cad_zero_connected"
MOUNTS = [
    [[-.06702745999809623, .022000000000328184, -.011897015000328165],
     [-.06702746999831166, -.021999999999671817, -.01189701500032822]],
    [[-.06702746999999999, -.02199999999820395, -.011797030000107733],
     [-.06702746500000001, .022000000001796048, -.0117970199998923]],
]
PIVOTS = [
    [[-.010018149999546831, .02200000000037696, .24490253999995004],
     [-.010031359998984213, -.021999999999622977, .16490248499995175]],
    [[-.010018150000000002, -.021999999999403597, .24500082000008544],
     [-.010031360000000001, .022000000000988118, .16500075499991454]],
]
# Lengths are retained at CAD precision, including the small left/right offset.
CRANK_LENGTHS = [[.0680001589854004, .06800014105548673],
                 [.0679998645456326, .06799983998877179]]
ROD_LENGTHS = [[.24519999469230372, .16529998992615366],
               [.24520001447272452, .16530000020958144]]


def rotation(q):
    """Ry(pitch) Rx(roll), supporting arbitrary leading batch dimensions."""
    p, r = q.unbind(-1)
    cp, sp, cr, sr = p.cos(), p.sin(), r.cos(), r.sin()
    z = torch.zeros_like(p)
    return torch.stack((cp, sp*sr, sp*cr, z, cr, -sr, -sp, cp*sr, cp*cr), -1).reshape(*q.shape[:-1], 3, 3)


def point_z(direction):
    z = torch.nn.functional.normalize(direction, dim=-1)
    ref = torch.zeros_like(z)
    ref[..., 1] = 1
    x = torch.nn.functional.normalize(torch.linalg.cross(ref, z), dim=-1)
    return torch.stack((x, torch.linalg.cross(z, x), z), -1)


class CADAnkleGeometry:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.mounts = torch.tensor(MOUNTS, device=device, dtype=dtype)
        self.pivots = torch.tensor(PIVOTS, device=device, dtype=dtype)
        self.crank_lengths = torch.tensor(CRANK_LENGTHS, device=device, dtype=dtype)
        self.rod_lengths = torch.tensor(ROD_LENGTHS, device=device, dtype=dtype)
        # Select exactly the CAD assembly branch; neutral motor angle is zero.
        self.alpha0 = self._raw_angles(torch.zeros((2, 2), device=device, dtype=dtype))[0]
        self.ball_zero_rot = self.crank_rotation(torch.zeros_like(self.alpha0)).transpose(-1, -2) @ point_z(
            self.mounts - self.crank_end(torch.zeros_like(self.alpha0)))

    def foot_ends(self, q):
        return (rotation(q).unsqueeze(-3) @ self.mounts.unsqueeze(-1)).squeeze(-1)

    def _raw_angles(self, q):
        delta = self.foot_ends(q) - self.pivots
        radius = torch.linalg.vector_norm(delta[..., [0, 2]], dim=-1)
        cosine = (self.rod_lengths.square() - delta.square().sum(-1) - self.crank_lengths.square()) / (
            2 * self.crank_lengths * radius.clamp_min(1e-12))
        valid = (cosine.abs() <= 1 + 2e-6) & torch.isfinite(cosine)
        theta = torch.atan2(delta[..., 2], delta[..., 0]) + cosine.clamp(-1, 1).acos()
        return theta, valid.all(-1)

    def inverse_kinematics(self, q):
        theta, valid = self._raw_angles(q)
        motor = torch.remainder(theta - self.alpha0 + torch.pi, 2*torch.pi) - torch.pi
        return motor, valid

    def crank_end(self, motor):
        theta = self.alpha0 + motor
        return self.pivots + torch.stack((-self.crank_lengths*theta.cos(),
                                         torch.zeros_like(theta), -self.crank_lengths*theta.sin()), -1)

    def crank_rotation(self, motor):
        a = self.alpha0 + motor
        c, s, z, o = a.cos(), a.sin(), torch.zeros_like(a), torch.ones_like(a)
        # Rz(pi) Ry(alpha0 + motor).
        return torch.stack((-c, z, -s, z, -o, z, -s, z, c), -1).reshape(*a.shape, 3, 3)

    def jacobian(self, q, motor=None):
        """dm/d[pitch,roll], differentiated from the actual rod constraint."""
        if motor is None:
            motor, _ = self.inverse_kinematics(q)
        ends = self.foot_ends(q)
        residual = ends - self.crank_end(motor)
        theta = self.alpha0 + motor
        dcrank = torch.stack((self.crank_lengths*theta.sin(), torch.zeros_like(theta),
                              -self.crank_lengths*theta.cos()), -1)
        denom = (residual*dcrank).sum(-1)
        axis_p = torch.zeros_like(ends)
        axis_p[..., 1] = 1
        p = q[..., 0]
        axis_r = torch.stack((p.cos(), torch.zeros_like(p), -p.sin()), -1).unsqueeze(-2).expand_as(ends)
        numer = torch.stack(((residual*torch.linalg.cross(axis_p, ends)).sum(-1),
                             (residual*torch.linalg.cross(axis_r, ends)).sum(-1)), -1)
        safe_denom = torch.where(denom >= 0, denom.clamp_min(1e-9), denom.clamp_max(-1e-9))
        return numer / safe_denom.unsqueeze(-1)

    def rod_relative_rotation(self, q, motor):
        rod = point_z(self.foot_ends(q) - self.crank_end(motor))
        return (self.crank_rotation(motor) @ self.ball_zero_rot).transpose(-1, -2) @ rod

    def reset_velocity(self, q, dq, motor):
        """Motor rates and PhysX spherical rates (angular velocity in child frame).

        The rod's unconstrained axial spin is set to zero. Spherical qpos is an
        exponential rotation vector, but qvel is NOT its time derivative.
        These conventions are checked against the actual GPU link velocities.
        """
        dm = (self.jacobian(q, motor) @ dq.unsqueeze(-1)).squeeze(-1)
        ends = self.foot_ends(q)
        axis_p = torch.zeros_like(ends)
        axis_p[..., 1] = 1
        p = q[..., 0]
        axis_r = torch.stack((p.cos(), torch.zeros_like(p), -p.sin()), -1).unsqueeze(-2).expand_as(ends)
        end_velocity = (torch.linalg.cross(axis_p, ends)*dq[..., 0, None, None]
                        + torch.linalg.cross(axis_r, ends)*dq[..., 1, None, None])
        theta = self.alpha0 + motor
        crank_velocity = torch.stack((self.crank_lengths*theta.sin()*dm, torch.zeros_like(dm),
                                      -self.crank_lengths*theta.cos()*dm), -1)
        direction = (ends - self.crank_end(motor)) / self.rod_lengths.unsqueeze(-1)
        direction_velocity = (end_velocity-crank_velocity) / self.rod_lengths.unsqueeze(-1)
        omega_rod = torch.linalg.cross(direction, direction_velocity)
        omega_crank = torch.stack((torch.zeros_like(dm), -dm, torch.zeros_like(dm)), -1)
        ball_velocity = (point_z(direction).transpose(-1, -2) @ (omega_rod-omega_crank).unsqueeze(-1)).squeeze(-1)
        return dm, ball_velocity


def motor_torque_from_joint(jac, joint_torque):
    """Solve J.T tau_motor=tau_joint, with explicit singularity reporting."""
    a, b, c, d = jac[..., 0, 0], jac[..., 0, 1], jac[..., 1, 0], jac[..., 1, 1]
    det = a*d - b*c
    safe = torch.where(det >= 0, det.clamp_min(1e-6), det.clamp_max(-1e-6))
    tp, tr = joint_torque.unbind(-1)
    torque = torch.stack((d*tp-c*tr, -b*tp+a*tr), -1) / safe.unsqueeze(-1)
    valid = torch.isfinite(jac).all(dim=(-1, -2)) & (det.abs() > 1e-5)
    return torch.where(valid.unsqueeze(-1), torque, 0.0), valid


def limit_motor_torque(torque, velocity, effort=112., speed=26.17993877991494, power=1250.):
    magnitude = torch.minimum(torch.as_tensor(effort, device=torque.device), power / velocity.abs().clamp_min(1e-6))
    upper = torch.where(velocity >= speed, 0., magnitude)
    lower = torch.where(velocity <= -speed, 0., -magnitude)
    return torque.clamp(min=lower, max=upper)
