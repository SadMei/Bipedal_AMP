"""Pure-Python tests for simulator-only PD candidates (no Isaac launch)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "scripts/rsl_rl/bsrl_pd_profiles.py"
SPEC = importlib.util.spec_from_file_location("bsrl_pd_profiles", SOURCE)
PROFILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILES)


@pytest.mark.parametrize("profile", PROFILES.PROFILES)
def test_gains_follow_joint_order_and_are_symmetric(profile):
    names = [f"joint_{side}_{axis}" for axis in reversed(PROFILES.AXES) for side in ("left", "right")]
    actual = PROFILES.gain_vectors(profile, names)
    expected = tuple([value for value in reversed(values) for _ in range(2)]
                     for values in PROFILES.PROFILES[profile])
    assert actual == expected


def test_nominal_preserves_saved_gains():
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={})))
    PROFILES.apply_bsrl_pd_profile(cfg, "BSRL", "nominal")
    assert cfg.scene.robot.actuators == {}


def test_override_changes_only_pd():
    actuators = {key: SimpleNamespace(stiffness=None, damping=None, effort_limit=112,
                                     min_delay=0, max_delay=4)
                 for key in ("large_motors", "small_motors", "parallel_ankles")}
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators=actuators)))
    PROFILES.apply_bsrl_pd_profile(cfg, "BSRL", "damped2")
    assert actuators["large_motors"].stiffness["joint_.*_knee_pitch"] == 200
    assert actuators["large_motors"].damping["joint_.*_hip_roll"] == 4
    assert actuators["small_motors"].damping == {"joint_.*_hip_yaw": 4}
    assert actuators["parallel_ankles"].damping == {
        "joint_.*_ankle_pitch": 2, "joint_.*_ankle_roll": 2}
    assert all((a.effort_limit, a.min_delay, a.max_delay) == (112, 0, 4) for a in actuators.values())


def test_rejects_t1_override():
    with pytest.raises(ValueError, match="restricted to BSRL"):
        PROFILES.apply_bsrl_pd_profile(None, "T1", "damped2")


def test_unknown_joint_is_not_silently_assigned_a_gain():
    with pytest.raises(ValueError):
        PROFILES.gain_vectors("nominal", ["joint_left_unknown"])


@pytest.mark.parametrize("profile", ("proximal15", "proximal2", "hip6_anklefixed", "knee4_anklefixed"))
def test_hardware_constrained_profiles_keep_ankle_pd(profile):
    kp, kd = PROFILES.PROFILES[profile]
    assert kp[-2:] == [100, 100]
    assert kd[-2:] == [1, 1]


def test_default_sweep_respects_ankle_constraint():
    for profile in PROFILES.DEFAULT_PROFILES:
        kp, kd = PROFILES.PROFILES[profile]
        assert kp[-2:] == [100, 100]
        assert kd[-2:] == [1, 1]
