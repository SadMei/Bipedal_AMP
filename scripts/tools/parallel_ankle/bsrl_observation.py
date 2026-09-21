"""Self-contained BSRL MuJoCo observation contract for closed-ankle replay.

This is the deploy-time contract used by the BSRL AMP policy.  It is kept next
to the replay tool so the Bipedal_AMP repository does not import sim2sim code
from another repository at runtime.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT_BODY_NAME = "base_link"
KEY_BODY_NAMES = ("link_right_ankle_roll", "link_left_ankle_roll")
CONTACT_BODY_NAMES = KEY_BODY_NAMES
JOINT_NAMES = (
    "joint_left_hip_yaw", "joint_right_hip_yaw",
    "joint_left_hip_roll", "joint_right_hip_roll",
    "joint_left_hip_pitch", "joint_right_hip_pitch",
    "joint_left_knee_pitch", "joint_right_knee_pitch",
    "joint_left_ankle_pitch", "joint_right_ankle_pitch",
    "joint_left_ankle_roll", "joint_right_ankle_roll",
)
OBS_HISTORY = 5
OBS_TERM_SLICES_BASE = (
    slice(0, 3), slice(3, 9), slice(9, 12),
    slice(12, 24), slice(24, 36), slice(36, 48),
)
OBS_TERM_SLICES_WITH_FEET = OBS_TERM_SLICES_BASE + (slice(48, 54),)
OBS_TERM_SLICES_NO_FEET_WITH_HEADING = OBS_TERM_SLICES_BASE + (slice(48, 50),)
OBS_TERM_SLICES_WITH_HEADING = OBS_TERM_SLICES_WITH_FEET + (slice(54, 56),)
PHYSICS_DT = 0.001
START_HEIGHT = 0.8665
JOINT_LIMIT_SOLREF = np.array([0.02, 1.0], dtype=np.float64)
JOINT_LIMIT_SOLIMP = np.array([0.95, 0.99, 0.001, 0.5, 2.0], dtype=np.float64)
CONTACT_SOLREF = np.array([0.005, 1.0], dtype=np.float64)
CONTACT_SOLIMP = np.array([0.99, 0.9999, 0.0002, 0.5, 2.0], dtype=np.float64)
CONTACT_FRICTION = np.array([1.0, 0.02, 0.001], dtype=np.float64)
CONTACT_CONDIM = 6


def _make_runtime_xml(model_path: Path) -> str:
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir")).expanduser()
        if not meshdir.is_absolute():
            meshdir = model_path.parent / meshdir
        compiler.set("meshdir", f"{meshdir.resolve()}/")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {element.get("joint") for element in actuator}
    for name in JOINT_NAMES:
        if name not in existing:
            ET.SubElement(actuator, "motor", name=f"motor_{name.removesuffix('_joint')}",
                          joint=name, gear="1")
    return ET.tostring(root, encoding="unicode")


def _quat_to_mat_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _root_local_rot(q):
    rotation = _quat_to_mat_wxyz(q)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    local = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ rotation
    return np.concatenate((local[:, 0], local[:, 2]))


class BSRLMujoco:
    """Minimal observation-only backend matching the BSRL actor layout."""

    def __init__(self, model_path: Path, command, history_length=OBS_HISTORY):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_string(_make_runtime_xml(self.model_path))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self._qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self._qvel_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self._body_ids = np.array([self.model.body(n).id for n in KEY_BODY_NAMES])
        self.command = np.asarray(command, dtype=np.float64)
        self.action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.actor_include_key_body = True
        self.actor_include_heading_error = False
        self.heading_target_yaw = 0.0
        self.history_length = int(history_length)
        self._history = deque(maxlen=self.history_length)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, START_HEIGHT]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._qpos_adr] = [0.0, 0.0, 0.0, 0.0, -0.2, -0.2, 0.4, 0.4, -0.2, -0.2, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        frame = self._observation_frame()
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(frame.copy())

    def _state(self):
        return (self.data.qpos[self._qpos_adr].copy(),
                self.data.qvel[self._qvel_adr].copy(), self.data.qpos[3:7].copy())

    def _observation_frame(self):
        q, dq, quat = self._state()
        base_id = self.model.body(ROOT_BODY_NAME).id
        base_pos = self.data.xpos[base_id]
        rotation = _quat_to_mat_wxyz(quat)
        key_pos_b = ((self.data.xpos[self._body_ids] - base_pos) @ rotation).reshape(-1)
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                  base_id, velocity, 1)
        values = [velocity[:3], _root_local_rot(quat), self.command, q, dq, self.action]
        if self.actor_include_key_body:
            values.append(key_pos_b)
        if self.actor_include_heading_error:
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            error = (self.heading_target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
            values.append(np.array([np.sin(error), np.cos(error)], dtype=np.float64))
        return np.concatenate(values)

    def observation(self):
        self._history.append(self._observation_frame())
        frames = np.stack(tuple(self._history), axis=0)
        if self.actor_include_key_body and self.actor_include_heading_error:
            slices = OBS_TERM_SLICES_WITH_HEADING
        elif self.actor_include_key_body:
            slices = OBS_TERM_SLICES_WITH_FEET
        elif self.actor_include_heading_error:
            slices = OBS_TERM_SLICES_NO_FEET_WITH_HEADING
        else:
            slices = OBS_TERM_SLICES_BASE
        return np.concatenate([frames[:, part].reshape(-1) for part in slices]) \
            .astype(np.float32, copy=False)[None, :]
