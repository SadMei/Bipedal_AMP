"""User-confirmed CAD-zero geometry and direct closed-chain MJCF construction.

Geometry now follows the CAD zero-pose bearings and motor axes, per user request.
RobotGeometry preserves the uploaded C++ parameterization for historical checks.
The generalized CAD solver uses the same rigid-rod closure principle; it does
not modify the hardware C++, YAML, encoder signs or the URDF zero.
The tiny auxiliary link masses are numerical placeholders, NOT measured masses.
Original whole-robot inertias/armature remain intact; no extra motor armature.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

if __package__:
    from ._paths import CALIBRATION, CORE, DATA_DIR as HERE, SCRIPT_DIR, URDF, WORKSPACE
else:
    from _paths import CALIBRATION, CORE, DATA_DIR as HERE, SCRIPT_DIR, URDF, WORKSPACE


def vec(x):
    return " ".join(f"{v:.16g}" for v in np.asarray(x).reshape(-1))


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def point_z(direction):
    z = np.asarray(direction) / np.linalg.norm(direction)
    ref = np.array([0., 1., 0.])
    if abs(z @ ref) > .99:
        ref = np.array([1., 0., 0.])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class RobotCore:
    """Compile the actual C++ file, not a replacement implementation."""
    def __init__(self):
        missing = [name for name in ("AnkleKinematics.hpp", "AnkleKinematics.cpp", "bsrl_bridge_bringup.yaml")
                   if not (CORE / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Original BSRL solver/calibration missing from {CORE}: {missing}. "
                "Restore the original files or set BSRL_ANKLE_CORE_DIR; no substitute solver is used."
            )
        self.build = tempfile.TemporaryDirectory(prefix="bsrl-closed-core-")
        build = Path(self.build.name)
        inc = build / "include/elmo_ethercat_sdk/bsrl"
        inc.mkdir(parents=True)
        (inc / "AnkleKinematics.hpp").symlink_to(CORE / "AnkleKinematics.hpp")
        lib = build / "core.so"
        subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC",
                        "-I", str(build / "include"), "-I", str(CORE),
                        str(CORE / "AnkleKinematics.cpp"), str(SCRIPT_DIR / "core_bridge.cpp"),
                        "-o", str(lib)], check=True)
        self.lib = ctypes.CDLL(str(lib))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.ankle_forward.argtypes = [ctypes.c_int, ptr, ptr, ptr]
        self.lib.ankle_forward.restype = ctypes.c_int
        self.lib.ankle_inverse.argtypes = [ctypes.c_int, ptr, ptr, ptr]
        self.lib.ankle_inverse.restype = ctypes.c_int
        self.lib.ankle_geometry.argtypes = [ctypes.c_int, ptr]
        self.lib.ankle_geometry.restype = None

    def forward(self, q, side):
        q = np.ascontiguousarray(q, dtype=np.float64)
        motor, jac = np.empty(2), np.empty((2, 2))
        code = self.lib.ankle_forward(side == "right", q, motor, jac)
        if code:
            raise ValueError(f"Robot C++ forward status={code}, side={side}, q={q}")
        return motor, jac

    def inverse(self, motor, initial, side):
        out = np.empty(2)
        code = self.lib.ankle_inverse(side == "right", np.ascontiguousarray(motor),
                                      np.ascontiguousarray(initial), out)
        if code:
            raise ValueError(f"Robot C++ inverse status={code}, side={side}")
        return out


class RobotGeometry:
    # Independent CAD evidence: the toe is +X, rods are behind the ankle (-X),
    # and the long branch is at -Y on the right / +Y on the left. The uploaded
    # core geometry has all three directions reversed. Keep its parameters
    # intact and apply an explicit proper rotation, not a visual-only mirror.
    installation_rotation = np.diag([-1., -1., 1.])
    joint_to_core_sign = np.array([-1., -1.])

    def __init__(self, side, core):
        if side not in ("left", "right"):
            raise ValueError(side)
        self.side, self.core = side, core
        with CALIBRATION.open() as stream:
            transmission = yaml.safe_load(stream)["bridge"]["ankle_transmissions"][side]
        self.calibration = transmission
        g = transmission["geometry"]
        keys = ("l1", "l2", "l3", "l4", "radius", "d0", "paper_x_robot_y_sign",
                "lateral_sign", "output_direction")
        flattened = np.concatenate([np.asarray(g[k]).reshape(-1) for k in keys])
        actual = np.empty(21)
        core.lib.ankle_geometry(side == "right", actual)
        np.testing.assert_allclose(flattened, actual, rtol=0, atol=1e-14,
                                   err_msg="Calibration YAML and C++ geometry differ")
        for key in keys:
            setattr(self, key, np.asarray(g[key], dtype=float))
        self.s = float(self.paper_x_robot_y_sign)
        self.core_mounts = self.radius[:, None] * self.d0 / np.linalg.norm(self.d0, axis=1)[:, None]
        core_pivots = np.column_stack([np.zeros(2), self.s * self.lateral_sign * self.l2, self.l1])
        self.mounts = self.core_mounts @ self.installation_rotation.T
        self.pivots = core_pivots @ self.installation_rotation.T
        self.alpha0 = self.raw_angles(np.zeros(2))
        self.crank_offsets = np.column_stack([-self.s*self.l3, np.zeros((2, 2))])
        self.motor_axis = np.array([0., 1., 0.])

    def forward(self, q):
        """URDF [pitch, roll] -> core motors and dm/dq_URDF.

        Rz(pi).T @ Ry(p) Rx(r) @ Rz(pi) = Ry(-p) Rx(-r).
        Consequently J_URDF = J_core(-q) @ diag(-1, -1), including in
        motor velocity and inverse-transpose torque conversion.
        """
        motor, jac = self.core.forward(self.joint_to_core_sign * np.asarray(q), self.side)
        return motor, jac * self.joint_to_core_sign[None, :]

    def inverse(self, motor, initial):
        q_core = self.core.inverse(motor, self.joint_to_core_sign * np.asarray(initial), self.side)
        return self.joint_to_core_sign * q_core

    @staticmethod
    def rotation(q):
        p, r = q
        cp, sp, cr, sr = np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        return np.array([[cp, sp*sr, sp*cr], [0, cr, -sr], [-sp, cp*sr, cp*cr]])

    def raw_angles(self, q):
        q_core = self.joint_to_core_sign * np.asarray(q)
        ends = self.core_mounts @ self.rotation(q_core).T
        x = self.s * ends[:, 1] - self.lateral_sign * self.l2
        y = -self.s * ends[:, 0]
        z = ends[:, 2] - self.l1
        a, b = -2*self.l3*y, -2*self.l3*z
        c = x*x+y*y+z*z+self.l3*self.l3-self.l4*self.l4
        discriminant = a*a+b*b-c*c
        if np.any(discriminant < -1e-12):
            raise ValueError("Unreachable real linkage pose")
        return 2*np.arctan((b-np.sqrt(np.maximum(0, discriminant)))/(a-c))

    def crank_rotation(self, branch, motor):
        return self.installation_rotation @ ry(
            self.s * (self.alpha0[branch] + self.output_direction[branch]*motor))

    def crank_end(self, branch, motor):
        return self.pivots[branch] + self.crank_rotation(branch, motor) @ self.crank_offsets[branch]

    def attach(self, root, shank, foot, origin, prefix, auxiliary_mass=.001,
               equality_timeconst=.002, equality_impedance=.9999):
        """Add actuated cranks and spherical-ended rods; pitch/roll stay passive."""
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        for i in range(2):
            name = f"{prefix}{i+1}"
            neutral_rot = self.crank_rotation(i, 0.)
            end = self.crank_end(i, 0.)
            length = np.linalg.norm(self.mounts[i]-end)
            np.testing.assert_allclose(length, self.l4[i], atol=1e-12, rtol=0)
            rod_rot = point_z(self.mounts[i]-end)
            crank = ET.SubElement(shank, "body", name=f"crank_{name}",
                                  pos=vec(origin+self.pivots[i]), quat=vec(quat(neutral_rot)))
            ET.SubElement(crank, "joint", name=f"motor_{name}", type="hinge",
                          # Axis is local to the already Rz(pi)-mounted crank:
                          # local +Y is URDF -Y, not URDF +Y.
                          axis=vec(self.motor_axis), limited="false",
                          damping="0", armature="0")
            offset = self.crank_offsets[i]
            ET.SubElement(crank, "geom", type="capsule", fromto=vec(np.r_[np.zeros(3), offset]),
                          size=".004", mass=str(auxiliary_mass), contype="0", conaffinity="0",
                          group="2", rgba=".15 .55 .95 1" if i==0 else ".98 .48 .12 1")
            rod = ET.SubElement(crank, "body", name=f"rod_{name}", pos=vec(offset),
                                quat=vec(quat(neutral_rot.T @ rod_rot)))
            ET.SubElement(rod, "joint", name=f"rod_ball_{name}", type="ball", damping="0")
            ET.SubElement(rod, "geom", type="capsule", fromto=vec([0,0,0,0,0,length]),
                          size=".003", mass=str(auxiliary_mass), contype="0", conaffinity="0",
                          group="2", rgba=".15 .55 .95 1" if i==0 else ".98 .48 .12 1")
            ET.SubElement(rod, "site", name=f"rod_end_{name}", pos=vec([0,0,length]), size=".003")
            ET.SubElement(foot, "site", name=f"foot_end_{name}", pos=vec(self.mounts[i]), size=".003")
            ET.SubElement(equality, "connect", name=f"closure_{name}", site1=f"rod_end_{name}",
                          site2=f"foot_end_{name}", solref=f"{equality_timeconst} 1",
                          solimp=f"{equality_impedance} {equality_impedance} .0001")
            ET.SubElement(actuator, "motor", name=f"drive_{name}", joint=f"motor_{name}", gear="1",
                          ctrllimited="true", ctrlrange="-112 112", forcelimited="true", forcerange="-112 112")

    def initialize(self, model, data, q, prefix):
        motors, _ = self.forward(q)
        for i, motor in enumerate(motors):
            name = f"{prefix}{i+1}"
            data.joint(f"motor_{name}").qpos[0] = motor
            current = point_z(self.rotation(q) @ self.mounts[i] - self.crank_end(i, motor))
            rod_body = model.body(f"rod_{name}")
            # body quaternion is followed by the ball-joint quaternion.
            body_R = Rotation.from_quat(rod_body.quat[[1,2,3,0]]).as_matrix()
            relative = body_R.T @ self.crank_rotation(i, motor).T @ current
            data.joint(f"rod_ball_{name}").qpos[:] = quat(relative)


class Geometry(RobotGeometry):
    """CAD zero-pose points are authoritative; arbitrary pivot X is supported.

    Existing robot C++ assumes the motor axis is at X=0 and cannot encode this
    CAD's X≈-10 mm motor axis. Do not feed changed points to an incompatible J.
    Solve |R(q)D0-C(m)|^2=Lrod^2 and differentiate that same constraint instead.
    """
    revision = "20260914_cad_zero_connected"

    def __init__(self, side, core):
        # Preserve original hardware provenance/parameter consistency checks.
        # Avoid calling overridden methods before CAD fields exist.
        legacy = RobotGeometry(side, core)
        self.__dict__.update(legacy.__dict__)
        self._load_cad()

    @classmethod
    def from_cad(cls, side):
        """Explicit CAD-only reconstruction; does NOT verify hardware C++/YAML.

        Used to recover generated MJCF when the original external core is lost.
        The normal constructor and its mandatory provenance checks are unchanged.
        """
        if side not in ("left", "right"):
            raise ValueError(side)
        geometry = cls.__new__(cls)
        geometry.side, geometry.core, geometry.calibration = side, None, None
        geometry._load_cad()
        return geometry

    def _load_cad(self):
        if __package__:
            from .cad_assets import load_cad
        else:
            from cad_assets import load_cad
        cad = load_cad(self.side)
        self.mounts = cad["mounts"].copy()
        self.pivots = cad["pivots"].copy()
        self.zero_crank_ends = cad["crank_ends"].copy()
        v = self.zero_crank_ends - self.pivots
        np.testing.assert_allclose(v[:, 1], 0., atol=1e-12)
        self.l3 = np.linalg.norm(v, axis=1)
        self.l4 = np.linalg.norm(self.mounts - self.zero_crank_ends, axis=1)
        self.crank_offsets = np.column_stack([self.l3, np.zeros((2, 2))])
        self.alpha0 = np.arctan2(-v[:, 2], -v[:, 0])
        self.core_mounts = self.mounts @ self.installation_rotation
        self.motor_axis = np.array([0., 1., 0.])
        self.output_direction = np.ones(2)
        np.testing.assert_allclose(self.raw_angles(np.zeros(2)), self.alpha0, atol=1e-12)

    def raw_angles(self, q):
        delta = self.mounts @ self.rotation(q).T - self.pivots
        radius = np.hypot(delta[:, 0], delta[:, 2])
        cosine = (self.l4**2 - np.sum(delta**2, axis=1) - self.l3**2) / (2*self.l3*radius)
        if np.any(np.abs(cosine) > 1 + 1e-12) or not np.isfinite(cosine).all():
            raise ValueError(f"Unreachable CAD ankle pose: {q}")
        # The +acos branch is selected by the existing CAD neutral assembly.
        theta = np.arctan2(delta[:, 2], delta[:, 0]) + np.arccos(np.clip(cosine, -1., 1.))
        return self.alpha0 + (theta - self.alpha0 + np.pi) % (2*np.pi) - np.pi

    def forward(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Expected finite [pitch, roll]")
        theta = self.raw_angles(q)
        motor = theta - self.alpha0
        # Roundoff at the authoritative zero should not redefine motor zero.
        if np.all(q == 0):
            motor = np.zeros(2)
        ends = self.mounts @ self.rotation(q).T
        cranks = self.pivots + np.column_stack([-self.l3*np.cos(theta), np.zeros(2), -self.l3*np.sin(theta)])
        residual = ends - cranks
        dcrank = np.column_stack([self.l3*np.sin(theta), np.zeros(2), -self.l3*np.cos(theta)])
        denominator = np.sum(residual*dcrank, axis=1)
        if np.any(np.abs(denominator) < 1e-9):
            raise ValueError("Singular CAD crank transmission")
        dp = np.cross([0., 1., 0.], ends)
        dr = np.cross([np.cos(q[0]), 0., -np.sin(q[0])], ends)
        jac = np.column_stack([np.sum(residual*dp, axis=1), np.sum(residual*dr, axis=1)]) / denominator[:, None]
        if not np.isfinite(jac).all() or np.linalg.cond(jac) > 100:
            raise ValueError("Singular CAD ankle Jacobian")
        return motor, jac

    def inverse(self, motor, initial):
        target, q = np.asarray(motor, dtype=float), np.array(initial, dtype=float)
        if target.shape != (2,) or q.shape != (2,) or not np.isfinite(np.r_[target, q]).all():
            raise ValueError("Expected finite motor/initial angles")
        for _ in range(30):
            estimate, jac = self.forward(q)
            error = estimate - target
            if np.max(np.abs(error)) < 1e-10:
                return q
            step = np.linalg.solve(jac, error)
            q = np.clip(q-step, [-.6, -.52], [.6, .52])
        raise ValueError("CAD motor-to-joint inverse did not converge")

    def crank_rotation(self, branch, motor):
        return self.installation_rotation @ ry(self.alpha0[branch] + motor)


def closure_gaps(model, data, prefixes):
    mujoco.mj_kinematics(model, data)
    return np.array([np.linalg.norm(data.site(f"rod_end_{p}{i}").xpos -
                                   data.site(f"foot_end_{p}{i}").xpos)
                     for p in prefixes for i in (1,2)])


def xml_string(root):
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def standalone_xml(geometry, auxiliary_mass=.001, equality_timeconst=.002):
    root = ET.Element("mujoco", model=f"BSRL calibrated {geometry.side} ankle")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=".001", integrator="implicitfast",
                  iterations="100", tolerance="1e-10")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="ankle_side", data=geometry.side)
    world = ET.SubElement(root, "worldbody")
    shank = ET.SubElement(world, "body", name="fixed_shank", pos="0 0 .5")
    ET.SubElement(shank, "geom", type="capsule", fromto="0 0 .04 0 0 .32",
                  size=".015", contype="0", conaffinity="0")
    carrier = ET.SubElement(shank, "body", name="pitch_carrier")
    ET.SubElement(carrier, "joint", name="ankle_pitch", axis="0 1 0", range="-.6 .6", armature=".01")
    urdf = ET.parse(URDF).getroot()

    def add_inertial(body, name):
        inertial = urdf.find(f"link[@name='{name}']/inertial")
        tensor = inertial.find("inertia")
        origin = inertial.find("origin")
        np.testing.assert_allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0)
        ET.SubElement(body, "inertial", pos=origin.get("xyz"), mass=inertial.find("mass").get("value"),
                      fullinertia=vec([float(tensor.get(k)) for k in ("ixx","iyy","izz","ixy","ixz","iyz")]))

    add_inertial(carrier, f"link_{geometry.side}_ankle_pitch")
    foot = ET.SubElement(carrier, "body", name="foot")
    ET.SubElement(foot, "joint", name="ankle_roll", axis="1 0 0", range="-.52 .52", armature=".01")
    add_inertial(foot, f"link_{geometry.side}_ankle_roll")
    ET.SubElement(foot, "geom", type="box", pos=".035 0 -.065", size=".125 .065 .012",
                  contype="0", conaffinity="0", rgba=".3 .35 .4 1")
    geometry.attach(root, shank, foot, np.zeros(3), "", auxiliary_mass, equality_timeconst)
    return xml_string(root)
