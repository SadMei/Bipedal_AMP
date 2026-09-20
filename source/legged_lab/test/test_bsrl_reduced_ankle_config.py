"""Kit imports/configuration and real DelayBuffer; no environment or PPO smoke."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _isaaclab_test_app import simulation_app
from isaaclab.utils.types import ArticulationActions
from legged_lab.assets.bsrl import BSRL_CFG
from legged_lab.assets.bsrl_reduced_ankle import (
    ReducedAnkleActuator, with_reduced_ankles, randomize_ankle_transmission,
)
from legged_lab.assets.bsrl_closed_ankle import with_closed_ankles

assert simulation_app is not None


def test_factory_has_no_explicit_constraints_and_no_motor_caps_on_joint_axes():
    cfg = with_reduced_ankles(BSRL_CFG)
    assert cfg.spawn.func is BSRL_CFG.spawn.func
    assert cfg.spawn.asset_path == BSRL_CFG.spawn.asset_path
    assert cfg.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 4
    assert cfg.spawn.articulation_props.enabled_self_collisions
    assert set(cfg.actuators) == {"large_motors", "small_motors", "parallel_ankles"}
    ankle = cfg.actuators["parallel_ankles"]
    assert set(ankle.stiffness.values()) == {100} and set(ankle.damping.values()) == {1}
    assert ankle.armature == .01
    assert ankle.effort_limit_sim == 1e9 and ankle.velocity_limit_sim == 1000
    # Factory must not mutate the template used by historical playback.
    assert "parallel_ankles" not in BSRL_CFG.actuators
    assert len(BSRL_CFG.actuators["small_motors"].joint_names_expr) == 3


def test_factory_respects_future_pipeline_pd_edits():
    changed = BSRL_CFG.copy()
    changed.actuators["small_motors"].stiffness["joint_.*_ankle_roll"] = 95.
    changed.actuators["small_motors"].damping["joint_.*_ankle_pitch"] = 1.5
    cfg = with_reduced_ankles(changed)
    assert cfg.actuators["parallel_ankles"].stiffness["joint_right_ankle_roll"] == 95.
    assert cfg.actuators["parallel_ankles"].damping["joint_left_ankle_pitch"] == 1.5
    assert cfg.actuators["small_motors"].stiffness["joint_left_hip_yaw"] == 150.


def test_actual_delayed_actuator_and_startup_event():
    cfg = with_reduced_ankles(BSRL_CFG).actuators["parallel_ankles"]
    cfg.min_delay, cfg.max_delay = 2, 2
    names = [f"joint_{s}_ankle_{a}" for s in ("right", "left") for a in ("roll", "pitch")]
    ac = ReducedAnkleActuator(cfg, names, slice(None), 4, "cpu")
    ac.reset(None)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(actuators={"parallel_ankles": ac})})
    randomize_ankle_transmission(env, None, ratio_range=(1.03, 1.03),
                                strength_range=(1., 1.), response_time_range=(0., 0.))
    pos = torch.zeros(4, 4)
    def compute(target):
        return ac.compute(ArticulationActions(joint_positions=target,
                          joint_velocities=torch.zeros_like(pos), joint_efforts=torch.zeros_like(pos)),
                          pos, torch.zeros_like(pos)).joint_efforts.clone()
    # First push fills history with CURRENT setpoint, not zero targets.
    torch.testing.assert_close(compute(pos+.01), torch.ones_like(pos)*1.03)
    torch.testing.assert_close(compute(pos+.02), torch.ones_like(pos)*1.03)
    torch.testing.assert_close(compute(pos+.03), torch.ones_like(pos)*1.03)
    torch.testing.assert_close(compute(pos+.04), torch.ones_like(pos)*2.06)
    old_other = ac.transmission.motor_torque[1].clone()
    ac.reset(torch.tensor([0]))
    assert ac.transmission.motor_torque[0].abs().max() == 0
    torch.testing.assert_close(ac.transmission.motor_torque[1], old_other)


@pytest.mark.parametrize("saved_kind", ["legacy", "explicit", "reduced", "reduced_dr"])
def test_play_restores_all_three_dynamics_contracts(saved_kind):
    path = Path(__file__).resolve().parents[3] / "scripts/rsl_rl/bsrl_dynamics_restore.py"
    spec = importlib.util.spec_from_file_location("bsrl_restore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = {"legacy": lambda c: c.copy(), "explicit": lambda c: with_closed_ankles(c, 5),
               "reduced": with_reduced_ankles, "reduced_dr": with_reduced_ankles}[saved_kind]
    saved_robot = factory(BSRL_CFG)
    saved = {"scene": {"robot": saved_robot.to_dict()}, "sim": {
        "dt": .001 if saved_kind == "explicit" else .005,
        "physx": {"solver_type": 0 if saved_kind == "explicit" else 1}},
        "decimation": 20 if saved_kind == "explicit" else 4,
        "events": {"ankle_transmission": None}}
    if saved_kind == "reduced_dr":
        saved["events"]["ankle_transmission"] = {"mode": "startup", "params": {
            "ratio_range": [.98, 1.02], "strength_range": [.95, 1.], "response_time_range": [0., .003]}}
    env = SimpleNamespace(scene=SimpleNamespace(
        robot=with_reduced_ankles(BSRL_CFG).replace(prim_path="robot"),
        contact_forces=SimpleNamespace()), sim=SimpleNamespace(physx=SimpleNamespace()),
        events=SimpleNamespace(ankle_transmission="must_not_leak"))
    module.restore_bsrl_dynamics(env, saved)
    assert set(env.scene.robot.actuators) == set(saved_robot.actuators)
    assert env.sim.dt == saved["sim"]["dt"] and env.decimation == saved["decimation"]
    if saved_kind == "reduced_dr":
        assert env.events.ankle_transmission.params["ratio_range"] == (.98, 1.02)
    else:
        assert env.events.ankle_transmission is None
