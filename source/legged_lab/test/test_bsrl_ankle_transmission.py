"""Production transmission/controller tests without launching Kit or PPO."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from test_bsrl_closed_ankle_actuator import Base, action, geometry

ASSETS = Path(__file__).resolve().parents[1] / "legged_lab/assets"


def load_transmission():
    path = ASSETS / "bsrl_ankle_transmission.py"
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    ns = {"torch": torch, "CADAnkleGeometry": geometry.CADAnkleGeometry,
          "limit_motor_torque": geometry.limit_motor_torque,
          "motor_torque_from_joint": geometry.motor_torque_from_joint}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["AnkleTransmission"]


Transmission = load_transmission()


def test_nominal_identity_full_urdf_range_and_virtual_power():
    tr = Transmission(8192, dtype=torch.float64)
    rng = torch.Generator().manual_seed(42)
    q = (torch.rand((8192, 2, 2), generator=rng, dtype=torch.float64)*2-1)*torch.tensor([.6, .52])
    dq = torch.rand(q.shape, generator=rng, dtype=torch.float64)*2-1
    request = torch.rand(q.shape, generator=rng, dtype=torch.float64)*20-10
    actual = tr.compute(q, dq, request)
    assert tr.valid.all()
    torch.testing.assert_close(actual, request, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((actual*dq).sum(-1), (tr.motor_torque*tr.motor_velocity).sum(-1))


def test_motor_limits_are_coupled_not_joint_boxes():
    tr = Transmission(1, dtype=torch.float64)
    q = torch.tensor([[[-.2, 0.], [-.2, 0.]]], dtype=torch.float64)
    request = torch.tensor([[[1000., 0.], [1000., 1000.]]], dtype=torch.float64)
    actual = tr.compute(q, torch.zeros_like(q), request)
    assert tr.motor_torque.abs().max() <= 112
    assert actual[..., 0].max() > 112  # physical ankle pitch can exceed one motor's torque
    assert not torch.allclose(actual, request.clamp(-112, 112))
    recovered, valid = geometry.motor_torque_from_joint(tr.geometry.jacobian(q), actual)
    assert valid.all()
    torch.testing.assert_close(recovered, tr.motor_torque)


def test_model_error_does_not_cancel_and_plant_power_is_consistent():
    tr = Transmission(4, dtype=torch.float64)
    tr.randomize(ratio_range=(1.03, 1.03))
    q = torch.zeros(4, 2, 2, dtype=torch.float64)
    request = torch.full_like(q, 2.)
    dq = torch.full_like(q, .1)
    actual = tr.compute(q, dq, request)
    torch.testing.assert_close(actual, 1.03*request)
    torch.testing.assert_close((actual*dq).sum(-1), (tr.motor_torque*tr.motor_velocity).sum(-1))
    tr.randomize(ratio_range=(1.03, 1.03))
    torch.testing.assert_close(tr.ratio, torch.full_like(q, 1.03))  # no accumulating scale


def test_randomization_independent_bounded_persistent_and_partial_reset():
    tr = Transmission(128)
    tr.randomize(ratio_range=(.97, 1.03), strength_range=(.9, 1.), response_time_range=(0., .005))
    assert .97 <= tr.ratio.min() <= tr.ratio.max() <= 1.03
    assert .9 <= tr.strength.min() <= tr.strength.max() <= 1.
    assert (tr.ratio[:, 0] != tr.ratio[:, 1]).any()
    tr.motor_torque[:] = 10
    old_ratio = tr.ratio.clone()
    tr.reset(torch.tensor([1, 5]))
    assert tr.motor_torque[[1, 5]].abs().max() == 0
    assert tr.motor_torque[0].min() == 10
    torch.testing.assert_close(tr.ratio, old_ratio)
    tr.randomize(torch.tensor([1]), ratio_range=(1., 1.))
    torch.testing.assert_close(tr.ratio[0], old_ratio[0])
    torch.testing.assert_close(tr.ratio[1], torch.ones_like(tr.ratio[1]))


def test_response_lag_and_envelope_reapplication():
    tr = Transmission(1, dtype=torch.float64)
    tr.randomize(strength_range=(.9, .9), response_time_range=(.005, .005))
    q = torch.zeros(1, 2, 2, dtype=torch.float64)
    jac = tr.geometry.jacobian(q)
    req_motor = torch.full_like(q, 50.)
    request = (jac.transpose(-1, -2) @ req_motor.unsqueeze(-1)).squeeze(-1)
    tr.compute(q, torch.zeros_like(q), request)
    torch.testing.assert_close(tr.motor_torque, 45*(1-torch.exp(torch.tensor(-1., dtype=torch.float64)))*torch.ones_like(q))
    # Find joint rates corresponding to positive overspeed of BOTH motors.
    dq = torch.linalg.solve(jac, torch.full_like(q, 40.).unsqueeze(-1)).squeeze(-1)
    tr.compute(q, dq, request)
    assert tr.motor_torque.max() <= 0  # lag must not retain prior accelerating effort
    for _ in range(10):
        tr.compute(q, dq, -100*request)
    assert tr.motor_torque.abs().max() <= 112*.9
    assert (tr.motor_torque*tr.motor_velocity).abs().max() <= 1250*.9+1e-8


def test_invalid_state_is_reported_without_injecting_nan_force():
    tr = Transmission(2)
    q = torch.zeros(2, 2, 2)
    q[0, 0, 0] = float("nan")
    request = torch.ones_like(q)
    request[1, 1, 0] = float("inf")
    actual = tr.compute(q, torch.zeros_like(q), request)
    assert not tr.valid[0, 0] and not tr.valid[1, 1]
    assert torch.isfinite(actual).all()
    assert actual[0, 0].abs().max() == 0
    assert actual[1, 1].abs().max() == 0


def controller():
    path = ASSETS / "bsrl_reduced_ankle.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
                and n.name == "ReducedAnkleActuator")
    ns = {"torch": torch, "DelayedPDActuator": Base, "AnkleTransmission": Transmission}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    names = ["joint_right_ankle_roll", "joint_left_ankle_pitch", "joint_left_ankle_roll", "joint_right_ankle_pitch"]
    cfg = SimpleNamespace(physics_dt=.005, motor_effort=112., motor_speed=26.17993878, motor_power=1250.)
    return ns["ReducedAnkleActuator"](cfg, names)


def test_production_actuator_joint_order_pd_and_unclipped_target():
    ac = controller()
    q = torch.tensor([[0., -.2, 0., -.2], [.1, -.2, .1, -.2]])
    vel = torch.zeros_like(q)
    target = q + torch.tensor([.01, .02, -.02, -.03])
    result = ac.compute(action(target), q, vel)
    torch.testing.assert_close(result.joint_efforts, 100*(target-q), atol=1e-5, rtol=1e-5)
    assert result.joint_positions is None and result.joint_velocities is None
    ac.compute(action(q+200.), q, vel)  # target far outside limits still NOT clipped
    torch.testing.assert_close(ac.computed_effort, torch.full_like(q, 20000.))
    assert ac.transmission.motor_torque.abs().max() <= 112
    assert ac.positions_delay_buffer.calls == 2
    ac.reset(torch.tensor([0]))
    assert ac.applied_effort[0].abs().max() == 0
    assert ac.applied_effort[1].abs().max() > 0


def test_reduced_clock_does_not_tick_during_reset_forward():
    path = ASSETS / "bsrl_reduced_ankle.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
                and n.name == "ReducedAnkleArticulation")
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("begin_physics_step", "_apply_actuator_model")]
    class FakeArticulation:
        count = 0
        def _apply_actuator_model(self):
            self.count += 1
    ns = {"Articulation": FakeArticulation}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    robot = ns["ReducedAnkleArticulation"]()
    for i in range(10):
        robot._apply_actuator_model()
        assert robot.count == i
        robot.begin_physics_step()
        robot._apply_actuator_model()
    assert robot.count == 10


def test_initialization_update_zero_is_not_a_physics_step():
    path = ASSETS / "bsrl_reduced_ankle.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
                and n.name == "ReducedAnkleArticulation")
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "update"]
    class FakeArticulation:
        def update(self, dt):
            self.last_dt = dt
    ns = {"Articulation": FakeArticulation, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    robot = ns["ReducedAnkleArticulation"]()
    robot.update(0.)  # no actuator initialization required, no audit/delay tick
    assert robot.last_dt == 0.
    assert not hasattr(robot, "_transmission_step")
