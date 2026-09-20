"""Test production controller scheduling without Kit (storage/buffers stubbed)."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

ASSETS = Path(__file__).resolve().parents[1] / "legged_lab/assets"
spec = importlib.util.spec_from_file_location("geometry", ASSETS / "bsrl_closed_ankle_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class Buffer:
    def __init__(self):
        self.calls = 0
    def compute(self, value):
        self.calls += 1
        return value


class Base:
    def __init__(self, cfg, joint_names, **kwargs):
        self.cfg, self.joint_names = cfg, joint_names
        self._device, self._num_envs = "cpu", 2
        self.computed_effort = torch.zeros(2, len(joint_names))
        self.applied_effort = torch.zeros_like(self.computed_effort)
        self.stiffness = torch.full_like(self.computed_effort, 100.)
        self.damping = torch.ones_like(self.computed_effort)
        self.positions_delay_buffer = Buffer()
        self.velocities_delay_buffer = Buffer()
        self.efforts_delay_buffer = Buffer()
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self.super_calls = 0
    def reset(self, env_ids):
        pass
    def _clip_effort(self, effort):
        return geometry.limit_motor_torque(effort, self._joint_vel)
    def compute(self, control_action, joint_pos, joint_vel):
        self.super_calls += 1
        self._joint_vel[:] = joint_vel
        self.computed_effort = 100*(control_action.joint_positions-joint_pos)-joint_vel
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = control_action.joint_velocities = None
        return control_action


def controller(name):
    path = ASSETS / "bsrl_closed_ankle_actuator.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    ns = {"torch": torch, "DelayedPDActuator": Base, "DelayedPowerLimitedPDActuator": Base,
          "CADAnkleGeometry": geometry.CADAnkleGeometry,
          "motor_torque_from_joint": geometry.motor_torque_from_joint,
          "limit_motor_torque": geometry.limit_motor_torque}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def action(target):
    return SimpleNamespace(joint_positions=target.clone(), joint_velocities=torch.zeros_like(target),
                           joint_efforts=torch.zeros_like(target))


def test_ankle_pd_and_delay_buffers_tick_at_200hz_and_passive_effort_zero():
    names = [f"joint_{s}_ankle_{a}" for s in ("left", "right") for a in ("pitch", "roll")]
    names += [f"motor_{s}_{b}" for s in ("left", "right") for b in ("long", "short")]
    ac = controller("ClosedAnkleActuator")(
        SimpleNamespace(control_decimation=5, motor_effort=112., motor_speed=26.17993878, motor_power=1250.), names)
    pos, vel = torch.zeros(2, 8), torch.zeros(2, 8)
    last_request = None
    for i in range(11):
        target = torch.zeros_like(pos)
        target[:, ac.q_ids] = -.01*(i+1)
        result = ac.compute(action(target), pos, vel)
        assert result.joint_positions is None and result.joint_velocities is None
        assert result.joint_efforts[:, ac.q_ids].abs().max() == 0
        if i % 5:
            torch.testing.assert_close(ac._requested_motor, last_request)
        else:
            last_request = ac._requested_motor.clone()
        assert ac.positions_delay_buffer.calls == i//5+1
        assert ac.velocities_delay_buffer.calls == i//5+1
        assert ac.efforts_delay_buffer.calls == i//5+1
    # Envelope is enforced at every physics step, even between held PD updates.
    vel[:, ac.m_ids] = 40.
    ac.compute(action(torch.zeros_like(pos)), pos, vel)
    assert ac.motor_torque.max() <= 0
    old_other = ac._requested_motor[1].clone()
    tick = ac._control_tick
    ac.reset(torch.tensor([0]))
    assert ac._requested_motor[0].abs().max() == 0
    torch.testing.assert_close(ac._requested_motor[1], old_other)
    assert ac._control_tick == tick  # partial resets do not rephase other envs


def test_direct_joint_pd_also_retains_200hz_and_substep_motor_envelope():
    ac = controller("SubsteppedPowerLimitedPDActuator")(SimpleNamespace(control_decimation=5), ["joint"])
    pos, vel = torch.zeros(2, 1), torch.zeros(2, 1)
    for i in range(10):
        ac.compute(action(torch.ones_like(pos)*(i+1)), pos, vel)
        assert ac.super_calls == i//5+1
        torch.testing.assert_close(ac.computed_effort, torch.full_like(pos, 100*(1+5*(i//5))))
        if i == 1:
            vel[:] = 40
        if i == 2:
            assert ac.applied_effort.max() == 0
            vel[:] = 0


def test_reset_forward_writes_do_not_tick_substepped_pd_or_delay():
    path = ASSETS / "bsrl_closed_ankle_articulation.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
               and n.name in ("begin_physics_step", "_apply_actuator_model")]
    class BaseArticulation:
        evaluations = 0
        def _apply_actuator_model(self):
            self.evaluations += 1
    ns = {"Articulation": BaseArticulation}
    node.body = methods
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    robot = ns["ClosedAnkleArticulation"]()
    robot._apply_actuator_model()  # startup env.reset -> scene.write -> sim.forward
    assert robot.evaluations == 0
    for i in range(40):
        robot.begin_physics_step()
        robot._apply_actuator_model()
        assert robot.evaluations == i+1
        # A partial episode reset/visualization write cannot consume another
        # physical tick, even when it lands exactly on the PD update boundary.
        robot._apply_actuator_model()
        assert robot.evaluations == i+1
