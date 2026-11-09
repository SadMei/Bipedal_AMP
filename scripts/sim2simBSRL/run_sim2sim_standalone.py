#!/usr/bin/env python3
"""Run the BSRL 12-DoF AMP policy exported from IsaacLab in MuJoCo.

The repository-local XML and URDF snapshots define the robot.  Mesh geometry is
installed beside them for local simulation but deliberately remains untracked.
The XML has no actuators, so this runner injects the twelve implicit position-PD
motors at load time, matching the IsaacLab play config.
"""

from __future__ import annotations

import argparse
import queue
import time
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
BSRL_ASSET_ROOT = (
    REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/BSRL_urdf"
)
LOCAL_BSRL_URDF = BSRL_ASSET_ROOT / "urdf"
DEFAULT_MODEL = BSRL_ASSET_ROOT / "xml" / "bsrl_scene.xml"
DEFAULT_POLICY = REPO_ROOT / (
    "logs/rsl_rl/bsrl_amp/bsrl_amp_no_actor_feet/"
    "2026-08-24_00-16-13_no_actor_feet_formal1_8192/exported/policy.onnx"
)
ROOT_BODY_NAME = "base_link"
START_HEIGHT = 0.8665
RUNNER_LABEL = "bsrl"
DEFAULT_RECORD_DIR = REPO_ROOT / "logs/sim2simBSRL"
RECORD_FILENAME = "bsrl_sim2sim.npz"

JOINT_NAMES = (
    "joint_left_hip_yaw", "joint_right_hip_yaw",
    "joint_left_hip_roll", "joint_right_hip_roll",
    "joint_left_hip_pitch", "joint_right_hip_pitch",
    "joint_left_knee_pitch", "joint_right_knee_pitch",
    "joint_left_ankle_pitch", "joint_right_ankle_pitch",
    "joint_left_ankle_roll", "joint_right_ankle_roll",
)
KEY_BODY_NAMES = ("link_right_ankle_roll", "link_left_ankle_roll")
CONTACT_BODY_NAMES = KEY_BODY_NAMES
NUM_JOINTS = len(JOINT_NAMES)
OBS_FRAME_DIM = 54
OBS_HISTORY = 5
OBS_DIM = OBS_FRAME_DIM * OBS_HISTORY
OBS_FRAME_DIM_NO_FEET = 48
OBS_DIM_NO_FEET = OBS_FRAME_DIM_NO_FEET * OBS_HISTORY
OBS_FRAME_DIM_WITH_HEADING = OBS_FRAME_DIM + 2
OBS_DIM_WITH_HEADING = OBS_FRAME_DIM_WITH_HEADING * OBS_HISTORY
OBS_FRAME_DIM_NO_FEET_WITH_HEADING = OBS_FRAME_DIM_NO_FEET + 2
OBS_DIM_NO_FEET_WITH_HEADING = OBS_FRAME_DIM_NO_FEET_WITH_HEADING * OBS_HISTORY

# Use a 1 kHz physics step while preserving the BSRL policy rate.
PHYSICS_DT = 0.001
DECIMATION = 20
POLICY_DT = PHYSICS_DT * DECIMATION
VIEWER_DECIMATION = 2 * DECIMATION
COMMAND_HUD_CONTROL_DECIMATION = 25
JOINT_LIMIT_SOLREF = np.array([0.02, 1.0], dtype=np.float64)
JOINT_LIMIT_SOLIMP = np.array([0.95, 0.99, 0.001, 0.5, 2.0], dtype=np.float64)
JOINT_LIMIT_MARGIN = 0.0
CONTACT_SOLREF = np.array([0.005, 1.0], dtype=np.float64)
CONTACT_SOLIMP = np.array([0.99, 0.9999, 0.0002, 0.5, 2.0], dtype=np.float64)
CONTACT_FRICTION = np.array([1.0, 0.02, 0.001], dtype=np.float64)
CONTACT_CONDIM = 6

_KEY_LEFT = 263
_KEY_RIGHT = 262
_KEY_KP_1 = 321
_KEY_KP_2 = 322
_KEY_KP_3 = 323
_KEY_KP_4 = 324
_KEY_KP_5 = 325
_SPEED_KEYS = {
    ord("1"): 1,
    ord("2"): 2,
    ord("3"): 3,
    ord("4"): 4,
    ord("5"): 8,
    _KEY_KP_1: 1,
    _KEY_KP_2: 2,
    _KEY_KP_3: 3,
    _KEY_KP_4: 4,
    _KEY_KP_5: 8,
}
COMMAND_MAX = np.array([1.2, 0.8, 0.5], dtype=np.float64)
COMMAND_LOWER = -COMMAND_MAX.copy()
COMMAND_UPPER = COMMAND_MAX.copy()
COMMAND_RATE = 0.2
DEFAULT_COMMAND = (1.0, 0.0, 0.0)

DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, 0.0, 0.0,
    -0.2, -0.2, 0.4, 0.4,
    -0.2, -0.2, 0.0, 0.0,
], dtype=np.float64)
KP = np.array([100, 100, 100, 100, 100, 100, 150, 150, 40, 40, 40, 40], dtype=np.float64)
KD = np.array([2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2], dtype=np.float64)
# A fixed target has no policy feedback to move the center of pressure. These
# stronger gains are reserved for the standalone static-hold diagnostic.
STAND_KP = np.array([150, 150, 150, 150, 150, 150, 300, 300, 200, 200, 200, 200], dtype=np.float64)
STAND_KD = np.array([5, 5, 5, 5, 5, 5, 8, 8, 5, 5, 5, 5], dtype=np.float64)
ARMATURE = np.full(NUM_JOINTS, 0.01, dtype=np.float64)
ACTION_SCALE = np.full(NUM_JOINTS, 0.25, dtype=np.float64)
TARGET_LOWER = None
TARGET_UPPER = None
# Trained env.yaml motor limits: small motors (hip-yaw/ankles) 26.1799 rad/s,
# large motors (hip-roll/hip-pitch/knees) 15.1320 rad/s.
VELOCITY_LIMIT = np.array([
    26.17993877991494, 26.17993877991494,
    15.132004614790837, 15.132004614790837,
    15.132004614790837, 15.132004614790837,
    15.132004614790837, 15.132004614790837,
    26.17993877991494, 26.17993877991494,
    26.17993877991494, 26.17993877991494,
], dtype=np.float64)

# IsaacLab ObservationManager concatenates flattened history term-by-term, not
# complete frame-by-frame. The first six terms are the no-actor-feet policy;
# the seventh term is present only in the original 270-D actor.
OBS_TERM_SLICES_BASE = (
    slice(0, 3),    # base_ang_vel
    slice(3, 9),    # root_local_rot_tan_norm
    slice(9, 12),   # velocity_commands
    slice(12, 24),  # joint_pos
    slice(24, 36),  # joint_vel
    slice(36, 48),  # actions
)
OBS_TERM_SLICES_WITH_FEET = OBS_TERM_SLICES_BASE + (
    slice(48, 54),  # key_body_pos_b
)
OBS_TERM_SLICES_WITH_HEADING = OBS_TERM_SLICES_WITH_FEET + (
    slice(54, 56),  # integrated_heading_error: sin(error), cos(error)
)
OBS_TERM_SLICES_NO_FEET_WITH_HEADING = OBS_TERM_SLICES_BASE + (
    slice(48, 50),  # integrated_heading_error: sin(error), cos(error)
)


def _control_modifier_pressed():
    """Read Ctrl while MuJoCo dispatches its GLFW key callback."""
    try:
        window = glfw.get_current_context()
        if not window:
            return False
        return (
            glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
    except (glfw.GLFWError, TypeError):
        return False


def _integrate_held_command(command, positive_held, negative_held, dt):
    """Integrate held-key commands and zero axes with no active direction."""
    result = np.asarray(command, dtype=np.float64).copy()
    for index, (positive, negative) in enumerate(zip(positive_held, negative_held)):
        direction = int(bool(positive)) - int(bool(negative))
        if direction == 0:
            result[index] = 0.0
        else:
            result[index] = np.clip(
                result[index] + direction * COMMAND_RATE * dt,
                COMMAND_LOWER[index],
                COMMAND_UPPER[index],
            )
    return result


def _make_runtime_xml(model_path: Path) -> str:
    """Resolve external assets and ensure every policy joint has a motor.

    The upstream BSRL XML intentionally has no actuators.  Newer revisions may
    add an actuator section, so append only missing motors instead of treating
    the mere presence of ``<actuator>`` as a complete contract.
    """
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir_text = compiler.get("meshdir")
        meshdir = Path(meshdir_text).expanduser()
        if not meshdir.is_absolute():
            # Older snapshots spelled this from the upstream repository parent.
            # Repository-local snapshots always resolve next to this XML.
            if meshdir_text == "BSRL_urdf/meshes/":
                meshdir = model_path.parent / ".." / "meshes"
            else:
                meshdir = model_path.parent / meshdir
        compiler.set("meshdir", f"{meshdir.resolve()}/")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuated_joints = {element.get("joint") for element in actuator}
    for name in JOINT_NAMES:
        if name not in actuated_joints:
            ET.SubElement(
                actuator, "motor", name=f"motor_{name.removesuffix('_joint')}",
                joint=name, gear="1",
            )
    return ET.tostring(root, encoding="unicode")


def _read_urdf_velocity_limits(model_path: Path) -> np.ndarray | None:
    """Read per-joint velocity limits from a repository-local BSRL URDF.

    MuJoCo's MJCF joint element has no velocity-limit field, while the updated
    BSRL URDF does.  Reading it when available keeps the runtime synchronized
    with future URDF revisions and falls back to the validated policy values
    for standalone/custom XML files.
    """
    candidates = (
        model_path.parent / "export_floating.urdf",
        model_path.parent / "export.urdf",
        model_path.parent.parent / "urdf" / "export_floating.urdf",
        model_path.parent.parent / "urdf" / "export.urdf",
        LOCAL_BSRL_URDF / "export_floating.urdf",
        LOCAL_BSRL_URDF / "export.urdf",
    )
    urdf_path = next((path for path in candidates if path.is_file()), None)
    if urdf_path is None:
        return None
    try:
        root = ET.parse(urdf_path).getroot()
        limits = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name and limit is not None and limit.get("velocity") is not None:
                limits[name] = float(limit.get("velocity"))
        if any(name not in limits for name in JOINT_NAMES):
            return None
        values = np.asarray([limits[name] for name in JOINT_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(values) & (values > 0)):
            return None
        return values
    except (ET.ParseError, OSError, TypeError, ValueError):
        return None


def _quat_to_mat_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = np.asarray([w, x, y, z]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _root_local_rot(q):
    rotation = _quat_to_mat_wxyz(q)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    yaw_inv = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    local = yaw_inv @ rotation
    return np.concatenate((local[:, 0], local[:, 2]))


def _yaw_from_quat_wxyz(q):
    rotation = _quat_to_mat_wxyz(q)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class BSRLMujoco:
    def __init__(self, model_path: Path, headless=False, command=None, delay_steps=0,
                 key_callback=None, follow_camera=True, actor_include_key_body=True,
                 actor_include_heading_error=False, heading_target=0.0):
        self.model_path = Path(model_path).expanduser().resolve()
        runtime_xml = _make_runtime_xml(self.model_path)
        self.model = mujoco.MjModel.from_xml_string(runtime_xml.encode("utf-8"))
        self.model.opt.timestep = PHYSICS_DT
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        missing_joints = [name for name in JOINT_NAMES if self._find_joint_id(name) < 0]
        missing_bodies = [name for name in KEY_BODY_NAMES if self._find_body_id(name) < 0]
        if missing_joints or missing_bodies:
            raise ValueError(
                f"incompatible BSRL XML {self.model_path}: "
                f"missing joints={missing_joints}, bodies={missing_bodies}"
            )
        self._qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES], dtype=int)
        self._qvel_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES], dtype=int)
        self._body_ids = np.array([self.model.body(n).id for n in KEY_BODY_NAMES], dtype=int)
        self._act_ids = np.array([self._actuator_id(n) for n in JOINT_NAMES], dtype=int)
        self.kp = KP.copy()
        self.kd = KD.copy()
        self.effort = self._read_xml_effort_limits()
        self.velocity_limit = _read_urdf_velocity_limits(self.model_path)
        if self.velocity_limit is None:
            self.velocity_limit = VELOCITY_LIMIT.copy()
            self.velocity_limit_source = "validated policy defaults"
        else:
            self.velocity_limit_source = "sibling BSRL URDF"
        raw_command = np.zeros(3, dtype=np.float64) if command is None else np.asarray(command, dtype=np.float64)
        self.command = np.clip(raw_command, COMMAND_LOWER, COMMAND_UPPER)
        self.actor_include_key_body = bool(actor_include_key_body)
        self.actor_include_heading_error = bool(actor_include_heading_error)
        # ``integrated_heading_error`` in the Isaac AMP environment does not
        # compare yaw against a fixed world heading.  It integrates the
        # commanded yaw-rate into a target heading, initialized at reset.
        # Keep the argument for API compatibility as an optional initial
        # offset, but update this state every low-level physics step.
        self.heading_target_offset = float(heading_target)
        self.heading_target_yaw = 0.0
        self.action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.target = DEFAULT_JOINT_POS.copy()
        self.delay_steps = max(0, int(delay_steps))
        self._target_queue = deque(maxlen=self.delay_steps + 1)
        self._history = deque(maxlen=OBS_HISTORY)
        self.follow_camera = bool(follow_camera)
        self._speed_hud = ""
        self._command_hud = ""
        self.viewer = None
        self._configure_model()
        self.reset()
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=key_callback,
            )
            self.viewer.cam.distance = 2.8
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.65]
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -18
            self.viewer.sync()

    def _find_joint_id(self, name):
        try:
            return self.model.joint(name).id
        except KeyError:
            return -1

    def _find_body_id(self, name):
        try:
            return self.model.body(name).id
        except KeyError:
            return -1

    def _actuator_id(self, joint_name):
        """Bind controls by joint, independent of actuator name or XML order."""
        joint_id = self.model.joint(joint_name).id
        matches = np.flatnonzero(
            (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (self.model.actuator_trnid[:, 0] == joint_id)
        )
        if matches.size != 1:
            raise ValueError(f"expected one actuator for {joint_name}, found {matches.tolist()}")
        return int(matches[0])

    def _read_xml_effort_limits(self):
        limits = []
        for name in JOINT_NAMES:
            jid = self.model.joint(name).id
            limits.append(float(np.min(np.abs(self.model.jnt_actfrcrange[jid]))))
        if not np.all(np.asarray(limits) > 0):
            raise ValueError(f"invalid BSRL XML effort limits: {limits}")
        return np.asarray(limits, dtype=np.float64)

    def _configure_model(self):
        for name in JOINT_NAMES:
            jid = self.model.joint(name).id
            if self.model.jnt_limited[jid]:
                self.model.jnt_solref[jid] = JOINT_LIMIT_SOLREF
                self.model.jnt_solimp[jid] = JOINT_LIMIT_SOLIMP
                self.model.jnt_margin[jid] = JOINT_LIMIT_MARGIN

        target_geom_ids = {self.model.geom("floor").id}
        for body_name in CONTACT_BODY_NAMES:
            body_id = self.model.body(body_name).id
            target_geom_ids.update(
                gid for gid in range(self.model.ngeom)
                if self.model.geom_bodyid[gid] == body_id
                and (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid])
            )
        for gid in target_geom_ids:
            self.model.geom_solref[gid] = CONTACT_SOLREF
            self.model.geom_solimp[gid] = CONTACT_SOLIMP
            self.model.geom_friction[gid] = CONTACT_FRICTION
            self.model.geom_condim[gid] = CONTACT_CONDIM

        for i, name in enumerate(JOINT_NAMES):
            jid = self.model.joint(name).id
            did = self.model.jnt_dofadr[jid]
            self.model.dof_armature[did] = ARMATURE[i]
            aid = self._act_ids[i]
            self.model.jnt_actfrclimited[jid] = False
            self.model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_dyntype[aid] = mujoco.mjtDyn.mjDYN_NONE
            self.model.actuator_gainprm[aid] = 0.0
            self.model.actuator_gainprm[aid, 0] = self.kp[i]
            self.model.actuator_biasprm[aid] = 0.0
            self.model.actuator_biasprm[aid, 1] = -self.kp[i]
            self.model.actuator_biasprm[aid, 2] = -self.kd[i]
            self.model.actuator_ctrllimited[aid] = False
            self.model.actuator_forcelimited[aid] = True
            self.model.actuator_forcerange[aid] = (-self.effort[i], self.effort[i])
        mujoco.mj_setConst(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.qpos[self._qpos_adr] = DEFAULT_JOINT_POS
        mujoco.mj_forward(self.model, self.data)
        self.heading_target_yaw = _wrap_to_pi(
            _yaw_from_quat_wxyz(self.data.qpos[3:7]) + self.heading_target_offset
        )
        self.target[:] = DEFAULT_JOINT_POS
        self.action[:] = 0.0
        self._target_queue.clear()
        for _ in range(self.delay_steps + 1):
            self._target_queue.append(self.target.copy())
        frame = self._observation_frame()
        self._history.clear()
        for _ in range(OBS_HISTORY):
            self._history.append(frame.copy())

    def _state(self):
        q = self.data.qpos[self._qpos_adr].copy()
        dq = self.data.qvel[self._qvel_adr].copy()
        quat = self.data.qpos[3:7].copy()
        return q, dq, quat

    def _observation_frame(self):
        q, dq, quat = self._state()
        base_pos = self.data.xpos[self.model.body(ROOT_BODY_NAME).id]
        rotation = _quat_to_mat_wxyz(quat)
        key_pos_b = ((self.data.xpos[self._body_ids] - base_pos) @ rotation).reshape(-1)
        # Ask MuJoCo for body-local velocity explicitly. mj_objectVelocity uses
        # [angular, linear] ordering and avoids relying on free-joint qvel frame
        # conventions.
        base_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.model.body(ROOT_BODY_NAME).id,
            base_velocity,
            1,
        )
        base_ang_vel = base_velocity[:3]
        values = [
            base_ang_vel,
            _root_local_rot(quat),
            self.command,
            q,
            dq,
            self.action,
        ]
        if self.actor_include_key_body:
            values.append(key_pos_b)
        if self.actor_include_heading_error:
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            error = _wrap_to_pi(self.heading_target_yaw - yaw)
            values.append(np.array([np.sin(error), np.cos(error)], dtype=np.float64))
        return np.concatenate(values)

    def observation(self):
        self._history.append(self._observation_frame())
        frames = np.stack(tuple(self._history), axis=0)
        # CircularBuffer.buffer is oldest-to-newest. ObservationManager first
        # flattens each term's five frames, then concatenates the seven terms.
        if self.actor_include_key_body and self.actor_include_heading_error:
            term_slices = OBS_TERM_SLICES_WITH_HEADING
        elif self.actor_include_key_body:
            term_slices = OBS_TERM_SLICES_WITH_FEET
        elif self.actor_include_heading_error:
            term_slices = OBS_TERM_SLICES_NO_FEET_WITH_HEADING
        else:
            term_slices = OBS_TERM_SLICES_BASE
        obs = np.concatenate([frames[:, term_slice].reshape(-1) for term_slice in term_slices])
        return obs.astype(np.float32, copy=False)[None, :]

    def set_action(self, action):
        self.action[:] = np.asarray(action, dtype=np.float64)
        requested = DEFAULT_JOINT_POS + self.action * ACTION_SCALE
        if TARGET_LOWER is not None and TARGET_UPPER is not None:
            requested = np.clip(requested, TARGET_LOWER, TARGET_UPPER)
        if self.delay_steps:
            self._target_queue.append(requested.copy())
            self.target[:] = self._target_queue.popleft()
        else:
            self.target[:] = requested
        self.data.ctrl[self._act_ids] = self.target

    def set_pd_gains(self, kp, kd):
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        if kp.shape != (NUM_JOINTS,) or kd.shape != (NUM_JOINTS,):
            raise ValueError(f"expected {NUM_JOINTS} PD gains, got kp={kp.shape}, kd={kd.shape}")
        self.kp[:] = kp
        self.kd[:] = kd
        for i, aid in enumerate(self._act_ids):
            self.model.actuator_gainprm[aid, 0] = self.kp[i]
            self.model.actuator_biasprm[aid, 1] = -self.kp[i]
            self.model.actuator_biasprm[aid, 2] = -self.kd[i]

    def step(self):
        self.data.ctrl[self._act_ids] = self.target
        # Isaac's integrated_heading_error advances with the low-level
        # command/update period.  This runner uses a 1 ms MuJoCo step, so the
        # commanded yaw-rate contributes exactly wz * 0.001 rad per step.
        if self.actor_include_heading_error:
            self.heading_target_yaw = _wrap_to_pi(
                self.heading_target_yaw + float(self.command[2]) * PHYSICS_DT
            )
        mujoco.mj_step(self.model, self.data)
        self.data.qvel[self._qvel_adr] = np.clip(
            self.data.qvel[self._qvel_adr], -self.velocity_limit, self.velocity_limit
        )

    def render(self):
        if self.viewer is not None:
            if self.follow_camera:
                self.viewer.cam.lookat[:] = self.data.xpos[self.model.body(ROOT_BODY_NAME).id]
            self.viewer.sync()

    def set_speed_hud(self, value):
        self._speed_hud = str(value)
        self._update_hud()

    def set_command_hud(self):
        self._command_hud = (
            f"vx {self.command[0]:+.2f}   vy {self.command[1]:+.2f}   "
            f"wz {self.command[2]:+.2f}"
        )
        self._update_hud()

    def _update_hud(self):
        if self.viewer is None:
            return
        self.viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "SIM SPEED",
                self._speed_hud,
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "COMMAND",
                self._command_hud,
            ),
        ])

    def viewer_alive(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        viewer = self.viewer
        self.viewer = None
        if viewer is None:
            return
        viewer.close()

        # MuJoCo 3.10 only posts an exit request from Handle.close(). Wait for
        # its GLFW thread to destroy the native window before Python teardown.
        simulate_ref = getattr(viewer, "_sim", None)
        if callable(simulate_ref):
            deadline = time.monotonic() + 2.0
            while simulate_ref() is not None and time.monotonic() < deadline:
                time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description=f"Run the {RUNNER_LABEL} AMP ONNX policy in MuJoCo")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-target-pd",
        action="store_true",
        help="do not load/run ONNX; hold the nominal default joint pose with fixed PD",
    )
    parser.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True,
        help="limit simulation to wall-clock speed; --no-realtime runs unlimited",
    )
    parser.add_argument("--no-follow-camera", "--no_follow_camera", action="store_true")
    parser.add_argument("--max_control_steps", type=int, default=0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record_dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--video", action="store_true", help="record an offscreen MuJoCo MP4")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--video_width", type=int, default=1280)
    parser.add_argument("--video_height", type=int, default=720)
    parser.add_argument("--video_fps", type=int, default=50)
    parser.add_argument(
        "--command", nargs=3, type=float, default=DEFAULT_COMMAND, metavar=("VX", "VY", "WZ")
    )
    parser.add_argument(
        "--keyboard-control", action="store_true",
        help="enable held-key WASD/arrow velocity control; starts from zero command",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=0,
        help=f"position-target delay in physics steps ({PHYSICS_DT * 1000:g} ms each); default is 0",
    )
    args = parser.parse_args()
    if args.video and (
        args.video_width <= 0 or args.video_height <= 0 or not 0 < args.video_fps <= 1 / PHYSICS_DT
    ):
        parser.error("video dimensions must be positive; video_fps must be between 1 and the physics rate")
    if not args.fixed_target_pd and not args.policy.is_file():
        raise FileNotFoundError(f"{RUNNER_LABEL} ONNX policy not found: {args.policy}")
    if not args.model.is_file():
        raise FileNotFoundError(f"{RUNNER_LABEL} MuJoCo XML not found: {args.model}")

    session = None
    inputs = []
    output_name = None
    actor_obs_dim = OBS_DIM
    actor_include_key_body = True
    actor_include_heading_error = False
    if not args.fixed_target_pd:
        session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"expected one {RUNNER_LABEL} policy input, got {[(x.name, x.shape) for x in inputs]}")
        actor_obs_dim = int(inputs[0].shape[-1])
        if actor_obs_dim == OBS_DIM:
            actor_include_key_body, actor_include_heading_error = True, False
        elif actor_obs_dim == OBS_DIM_NO_FEET:
            actor_include_key_body, actor_include_heading_error = False, False
        elif actor_obs_dim == OBS_DIM_WITH_HEADING:
            actor_include_key_body, actor_include_heading_error = True, True
        elif actor_obs_dim == OBS_DIM_NO_FEET_WITH_HEADING:
            actor_include_key_body, actor_include_heading_error = False, True
        else:
            raise ValueError(
                f"unsupported {RUNNER_LABEL} policy input [1,{actor_obs_dim}]; "
                f"expected [1,{OBS_DIM}] / [1,{OBS_DIM_NO_FEET}] without heading or "
                f"[1,{OBS_DIM_WITH_HEADING}] / [1,{OBS_DIM_NO_FEET_WITH_HEADING}] with heading_error, "
                f"got {[(x.name, x.shape) for x in inputs]}"
            )
        output_name = session.get_outputs()[0].name
    state = {
        "paused": False,
        "realtime_limited": bool(args.realtime),
        "speed_multiplier": 1,
        "wall_deadline": time.perf_counter(),
    }
    key_events = queue.SimpleQueue()
    keyboard_state = {
        "window": None,
        "movement_enabled": False,
        "movement_held": False,
        "hud_control_steps": 0,
    }

    def key_callback(keycode):
        try:
            window = glfw.get_current_context()
            if window:
                keyboard_state["window"] = window
        except (glfw.GLFWError, TypeError):
            pass
        key_events.put((int(keycode), _control_modifier_pressed()))

    sim = BSRLMujoco(
        args.model,
        headless=args.headless,
        command=np.zeros(3) if args.keyboard_control else args.command,
        delay_steps=args.delay_steps,
        key_callback=key_callback,
        follow_camera=not args.no_follow_camera,
        actor_include_key_body=actor_include_key_body,
        actor_include_heading_error=actor_include_heading_error,
    )
    video_renderer = None
    video_writer = None
    video_camera = None
    video_path = args.video_path

    def set_speed_multiplier(multiplier):
        state["speed_multiplier"] = int(multiplier)
        state["realtime_limited"] = True
        state["wall_deadline"] = time.perf_counter()
        sim.set_speed_hud(f"{state['speed_multiplier']}x")
        print(f"[speed] limited to {state['speed_multiplier']}x real time")

    def stop_command():
        sim.command[:] = 0.0
        sim.set_command_hud()
        print("[command] stopped: vx=+0.00 m/s, vy=+0.00 m/s, wz=+0.00 rad/s")

    def update_held_command():
        window = keyboard_state["window"]
        if window is None:
            return
        try:
            positive = (
                glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS,
                glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS,
                glfw.get_key(window, glfw.KEY_COMMA) == glfw.PRESS,
            )
            negative = (
                glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS,
                glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS,
                glfw.get_key(window, glfw.KEY_PERIOD) == glfw.PRESS,
            )
        except (glfw.GLFWError, TypeError):
            return
        movement_held = any(positive) or any(negative)
        if movement_held:
            keyboard_state["movement_enabled"] = True
        if not keyboard_state["movement_enabled"]:
            return
        sim.command[:] = _integrate_held_command(
            sim.command, positive, negative, POLICY_DT
        )
        movement_changed = movement_held != keyboard_state["movement_held"]
        if movement_changed:
            if movement_held:
                print("[command] movement key held")
            else:
                print("[command] movement key released; active axes reset to zero")
        keyboard_state["movement_held"] = movement_held
        keyboard_state["hud_control_steps"] += 1
        if (
            movement_changed
            or keyboard_state["hud_control_steps"] % COMMAND_HUD_CONTROL_DECIMATION == 0
        ):
            sim.set_command_hud()

    def process_key_events():
        while True:
            try:
                keycode, ctrl_pressed = key_events.get_nowait()
            except queue.Empty:
                return

            speed_multiplier = _SPEED_KEYS.get(keycode) if ctrl_pressed else None
            if speed_multiplier is not None:
                set_speed_multiplier(speed_multiplier)
            elif keycode == 32:
                state["paused"] = not state["paused"]
                print("[viewer] PAUSED (space to resume)" if state["paused"] else "[viewer] running")
            else:
                try:
                    key = chr(keycode).lower()
                except ValueError:
                    continue
                if key == "0" and args.keyboard_control:
                    stop_command()

    sim.set_speed_hud("1x" if state["realtime_limited"] else "UNLIMITED")
    sim.set_command_hud()
    if args.fixed_target_pd:
        sim.set_pd_gains(STAND_KP, STAND_KD)
    print(f"[{RUNNER_LABEL}] model={args.model}")
    if args.fixed_target_pd:
        print(f"[{RUNNER_LABEL}] mode=fixed-target-pd (ONNX disabled, holding nominal pose)")
    else:
        print(
            f"[{RUNNER_LABEL}] policy={args.policy} obs={actor_obs_dim} history={OBS_HISTORY} "
            f"actor_key_body={'on' if actor_include_key_body else 'off'} "
            f"integrated_heading={'on' if actor_include_heading_error else 'off'} "
            f"dt={PHYSICS_DT * 1000:g}ms decimation={DECIMATION}"
        )
    print(
        f"[{RUNNER_LABEL}] contact: friction={CONTACT_FRICTION.tolist()}, condim={CONTACT_CONDIM}, "
        f"solref={CONTACT_SOLREF.tolist()}, solimp={CONTACT_SOLIMP.tolist()}"
    )
    print(f"[{RUNNER_LABEL}] action_scale={ACTION_SCALE.tolist()}, kp={sim.kp.tolist()}, kd={sim.kd.tolist()}")
    print(f"[{RUNNER_LABEL}] effort limits from XML={sim.effort.tolist()}, armature={ARMATURE.tolist()}, command={sim.command.tolist()}")
    print(f"[{RUNNER_LABEL}] velocity limits from {sim.velocity_limit_source}={sim.velocity_limit.tolist()}")
    print(
        f"[{RUNNER_LABEL}] command limits: lower={COMMAND_LOWER.tolist()}, "
        f"upper={COMMAND_UPPER.tolist()}"
    )
    if not args.headless:
        if args.keyboard_control:
            print(
                f"[{RUNNER_LABEL}] keyboard control enabled: up/down = forward/back; "
                "left/right = left/right; ,/. = wz; 0 = stop"
            )
        print(
            f"[{RUNNER_LABEL}] viewer keys: space = pause/resume; "
            "Ctrl+1/2/3/4/5 = 1x/2x/3x/4x/8x speed"
        )
    record = {
        "t": [], "q": [], "dq": [], "action": [], "target": [], "tau": [], "command": [],
        "base_pos": [], "base_quat_wxyz": [],
    } if args.record else None
    control_step = 0
    physics_step = 0
    video_frame = 0
    try:
        if args.video:
            import imageio.v2 as imageio

            if video_path is None:
                video_path = args.record_dir / f"{RUNNER_LABEL}_sim2sim.mp4"
            video_path = video_path.expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, args.video_width)
            sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, args.video_height)
            video_renderer = mujoco.Renderer(
                sim.model, height=args.video_height, width=args.video_width
            )
            video_writer = imageio.get_writer(
                str(video_path), fps=args.video_fps, codec="libx264", quality=8
            )
            video_camera = mujoco.MjvCamera()
            video_camera.distance = 3.0
            video_camera.azimuth = 125
            video_camera.elevation = -18

        while sim.viewer_alive():
            process_key_events()
            waited_while_paused = state["paused"]
            while state["paused"] and sim.viewer_alive():
                process_key_events()
                sim.render()
                time.sleep(0.01)
            if not sim.viewer_alive():
                break
            if waited_while_paused:
                state["wall_deadline"] = time.perf_counter()

            is_control_step = physics_step % DECIMATION == 0
            if is_control_step:
                if args.max_control_steps and control_step >= args.max_control_steps:
                    break
                if args.keyboard_control:
                    update_held_command()
                if args.fixed_target_pd:
                    action = np.zeros(NUM_JOINTS, dtype=np.float32)
                    sim.set_action(action)
                else:
                    obs = sim.observation()
                    action = session.run([output_name], {inputs[0].name: obs})[0].squeeze(0)
                    sim.set_action(action)
                if record is not None:
                    q, dq, _ = sim._state()
                    tau = np.clip((sim.target - q) * sim.kp - dq * sim.kd, -sim.effort, sim.effort)
                    record["t"].append(physics_step * PHYSICS_DT)
                    record["q"].append(q)
                    record["dq"].append(dq)
                    record["action"].append(action)
                    record["target"].append(sim.target.copy())
                    record["tau"].append(tau)
                    record["command"].append(sim.command.copy())
                    record["base_pos"].append(sim.data.qpos[:3].copy())
                    record["base_quat_wxyz"].append(sim.data.qpos[3:7].copy())
                control_step += 1
            # Schedule by simulated time, not by the policy rate. Changing
            # output FPS must not speed up or slow down the recorded motion.
            if video_renderer is not None and physics_step * PHYSICS_DT + 1e-10 >= video_frame / args.video_fps:
                video_camera.lookat[:] = sim.data.xpos[sim.model.body(ROOT_BODY_NAME).id]
                video_renderer.update_scene(sim.data, camera=video_camera)
                video_writer.append_data(video_renderer.render())
                video_frame += 1
            sim.step()
            if physics_step % VIEWER_DECIMATION == 0:
                sim.render()
            physics_step += 1
            if (
                state["realtime_limited"]
                and not args.headless
                and physics_step % DECIMATION == 0
            ):
                state["wall_deadline"] += POLICY_DT / state["speed_multiplier"]
                remaining = state["wall_deadline"] - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                elif remaining < -POLICY_DT:
                    state["wall_deadline"] = time.perf_counter()
    except KeyboardInterrupt:
        print(f"\n[{RUNNER_LABEL}] stopped.")
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[{RUNNER_LABEL}] wrote {video_path} ({video_frame} frames at {args.video_fps} fps)")
        if video_renderer is not None:
            video_renderer.close()
        if record is not None:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            out = args.record_dir / RECORD_FILENAME
            np.savez_compressed(out, **{k: np.asarray(v) for k, v in record.items()})
            print(f"[{RUNNER_LABEL}] wrote {out}")
        sim.close()


if __name__ == "__main__":
    main()
