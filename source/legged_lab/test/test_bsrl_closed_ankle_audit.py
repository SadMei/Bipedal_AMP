"""Run the production audit with tensor-only stand-ins, without Kit/PPO."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def robot_for_audit(gap=.05, torque=112., valid=True):
    path = ROOT / "source/legged_lab/legged_lab/assets/bsrl_closed_ankle_articulation.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "update"]

    class Base:
        def update(self, dt):
            pass

    ns = {"Articulation": Base, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    robot = ns["ClosedAnkleArticulation"]()
    robot._ankle_ready = True
    robot.num_instances = 2
    robot.ankle_q_ids = [[0, 1], [2, 3]]
    robot.joint_names = [str(i) for i in range(8)]
    robot.data = SimpleNamespace(joint_pos=torch.zeros(2, 8), joint_vel=torch.zeros(2, 8),
                                 root_pose_w=torch.zeros(2, 7), applied_torque=torch.zeros(2, 8))
    robot.actuators = {"closed_ankles": SimpleNamespace(
        motor_torque=torch.full((2, 2, 2), torque), motor_velocity=torch.zeros(2, 2, 2),
        jacobian_valid=torch.full((2, 2), valid, dtype=torch.bool))}
    robot.closure_gaps = lambda: torch.full((2, 2, 2), gap)
    robot._audit_step = 479
    robot._audit_reset_count = 0
    robot._audit = torch.zeros(7)
    robot._audit_gap_totals = torch.zeros(2)
    robot._audit_total_closure_max = 0.
    robot._audit_peak_state = torch.zeros(24)
    return robot


def test_large_finite_closure_gap_is_logged_not_rejected(capsys):
    robot = robot_for_audit(gap=.5)
    robot.update(.001)
    assert "gap_monitor_only=1" in capsys.readouterr().out
    assert robot._audit_total_closure_max == .5
    assert robot._audit_gap_totals.sum() == 0


@pytest.mark.parametrize("kwargs", [{"gap": float("nan")}, {"torque": 113.}, {"valid": False}])
def test_real_failure_guards_remain(kwargs):
    with pytest.raises(RuntimeError, match="physical audit failed"):
        robot_for_audit(**kwargs).update(.001)


def test_zero_dt_does_not_advance_audit():
    robot = robot_for_audit()
    robot.update(0.)
    assert robot._audit_step == 479


def test_acceptance_monitor_respects_new_gap_policy_and_old_runs():
    path = ROOT / "scripts/rsl_rl/watch_closed_ankle_training.py"
    spec = importlib.util.spec_from_file_location("closed_training_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = ("[ClosedAnkleAudit] physics_steps=480 closure_max_mm=500 motor_peak_Nm=112 "
           "passive_direct_Nm=0 motor_power_peak_W=1250 singular=0 nonfinite=0")
    assert module.parse_log(log)["failures"]
    assert not module.parse_log(log + " gap_monitor_only=1")["failures"]
    assert module.parse_log(log.replace("nonfinite=0", "nonfinite=1") + " gap_monitor_only=1")["failures"]
