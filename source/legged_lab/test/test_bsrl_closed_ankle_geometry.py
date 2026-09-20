"""CPU tests of the production batched geometry; GPU checks use the check script."""
import importlib.util
import ast
from pathlib import Path
import torch


path = Path(__file__).resolve().parents[1] / "legged_lab/assets/bsrl_closed_ankle_geometry.py"
spec = importlib.util.spec_from_file_location("closed_geometry", path)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_neutral_and_8192_reachable_limit_poses():
    a = g.CADAnkleGeometry(dtype=torch.float64)
    motor, valid = a.inverse_kinematics(torch.zeros((2, 2), dtype=torch.float64))
    torch.testing.assert_close(motor, torch.zeros_like(motor), atol=1e-12, rtol=0)
    gen = torch.Generator().manual_seed(42)
    q = (2*torch.rand((8192, 2, 2), generator=gen, dtype=torch.float64)-1)*torch.tensor([.6, .52])
    motor, valid = a.inverse_kinematics(q)
    assert valid.all()
    lengths = (a.foot_ends(q)-a.crank_end(motor)).norm(dim=-1)
    torch.testing.assert_close(lengths, a.rod_lengths.expand_as(lengths), atol=1e-12, rtol=0)
    assert torch.linalg.cond(a.jacobian(q, motor)).max() < 4


def test_jacobian_finite_difference_and_virtual_work():
    a = g.CADAnkleGeometry(dtype=torch.float64)
    q = torch.tensor([[[.3, .25], [-.4, -.35]], [[-.6, .52], [.6, -.52]]], dtype=torch.float64)
    motor, valid = a.inverse_kinematics(q)
    jac = a.jacobian(q, motor)
    for axis in range(2):
        delta = torch.zeros_like(q)
        delta[..., axis] = 1e-6
        plus, _ = a.inverse_kinematics(q+delta)
        minus, _ = a.inverse_kinematics(q-delta)
        torch.testing.assert_close((plus-minus)/2e-6, jac[..., axis], atol=2e-9, rtol=0)
    requested = torch.tensor([[[500., -300.], [-200., 400.]]], dtype=torch.float64).expand(2, -1, -1)
    tm, valid = g.motor_torque_from_joint(jac, requested)
    assert valid.all() and tm.abs().max() > 112  # joint request is not prematurely clipped
    torch.testing.assert_close((jac.transpose(-1, -2)@tm.unsqueeze(-1)).squeeze(-1), requested)
    dq = torch.tensor([[[1.2, -.8], [-.7, 1.1]]], dtype=torch.float64).expand(2, -1, -1)
    dm, _ = a.reset_velocity(q, dq, motor)
    torch.testing.assert_close((tm*dm).sum(-1), (requested*dq).sum(-1))


def test_motor_envelope_and_braking():
    velocity = torch.tensor([0., 20., 30., -30., 30., -30.])
    request = torch.tensor([1000., 1000., 1000., -1000., -1000., 1000.])
    actual = g.limit_motor_torque(request, velocity)
    torch.testing.assert_close(actual, torch.tensor([112., 62.5, 0., 0., -1250/30, 1250/30]))
    assert actual.abs().max() <= 112
    assert (actual*velocity).abs().max() <= 1250.001


def test_reset_analytic_rod_endpoint_velocities():
    a = g.CADAnkleGeometry(dtype=torch.float64)
    q = torch.tensor([[[.3, .25], [-.4, -.35]]], dtype=torch.float64)
    dq = torch.tensor([[[1.2, -.8], [-.7, 1.1]]], dtype=torch.float64)
    motor, _ = a.inverse_kinematics(q)
    dm, ball_velocity = a.reset_velocity(q, dq, motor)
    eps = 1e-6
    plus, _ = a.inverse_kinematics(q+eps*dq)
    minus, _ = a.inverse_kinematics(q-eps*dq)
    vc = (a.crank_end(plus)-a.crank_end(minus))/(2*eps)
    vf = (a.foot_ends(q+eps*dq)-a.foot_ends(q-eps*dq))/(2*eps)
    rod = a.foot_ends(q)-a.crank_end(motor)
    omega_c = torch.stack((torch.zeros_like(dm), -dm, torch.zeros_like(dm)), -1)
    omega_r = omega_c + (g.point_z(rod)@ball_velocity.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(vc+torch.linalg.cross(omega_r, rod), vf, atol=2e-9, rtol=0)


def test_explicit_experiment_is_opt_in_and_urdf_stays_serial(monkeypatch):
    package = path.parent.parent
    config = package / "tasks/locomotion/amp/config/bsrl/bsrl_amp_env_cfg.py"
    tree = ast.parse(config.read_text())
    switch = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "BSRL_EXPLICIT_CLOSED_ANKLES" for t in n.targets))
    import os
    monkeypatch.delenv("BSRL_EXPLICIT_CLOSED_ANKLES", raising=False)
    expression = compile(ast.Expression(switch.value), str(config), "eval")
    assert eval(expression, {"os": os}) is False
    monkeypatch.setenv("BSRL_EXPLICIT_CLOSED_ANKLES", "1")
    assert eval(expression, {"os": os}) is True
    # No edits to the authoritative original topology/zero/limits are needed.
    import xml.etree.ElementTree as ET
    root = ET.parse(package / "data/Robots/BSRL_urdf/urdf/export.urdf").getroot()
    assert len(root.findall("joint")) == 12
    assert not root.findall("loop_joint")
    for side, offset in (("right", -.355), ("left", -.3549)):
        pitch = root.find(f"joint[@name='joint_{side}_ankle_pitch']")
        assert float(pitch.find("origin").get("xyz").split()[2]) == offset
        assert pitch.find("axis").get("xyz") == "0 1 0"
