"""CPU tests of production heading code and actual uniform sampling methods.

Only the simulator-dependent command-manager shell is replaced. Production
heading code and Isaac UniformVelocityCommand sampling/postprocessing execute.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from bsrl_heading_ablation import (
    check_heading_configuration, expected_heading_configuration, validate_reduced_heading_baseline,
)
from t1_heading_ablation import load_saved_config

PRODUCTION = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/heading_frame.py"
ISAAC_UNIFORM = Path("/root/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py")


def wrap(value):
    return torch.remainder(value + torch.pi, 2 * torch.pi) - torch.pi


class UniformShell:
    def __init__(self, cfg, env):
        self.cfg, self._env, self.robot = cfg, env, env.scene[cfg.asset_name]
        self.num_envs, self.device = env.num_envs, env.device
        self.vel_command_b = torch.zeros(env.num_envs, 3)
        self.heading_target = torch.zeros(env.num_envs)
        self.is_heading_env = torch.zeros(env.num_envs, dtype=torch.bool)
        self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool)
        self.metrics = {key: torch.zeros(env.num_envs) for key in ("error_vel_xy", "error_vel_yaw")}
        self.resample_next = False

    def reset(self, env_ids):
        self._resample_command(env_ids)
        self._update_command()
        return {}

    def compute(self, dt):
        self._update_metrics()
        if self.resample_next:
            self._resample_command(torch.arange(self.num_envs))
            self.resample_next = False
        self._update_command()


def load_production():
    if not ISAAC_UNIFORM.is_file():
        pytest.skip("Local Isaac source needed to verify actual sampling")
    tree = ast.parse(ISAAC_UNIFORM.read_text())
    uniform = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UniformVelocityCommand")
    namespace = {"torch": torch, "Sequence": list, "math_utils": SimpleNamespace(wrap_to_pi=wrap)}
    for name in ("_resample_command", "_update_command", "_update_metrics"):
        node = next(n for n in uniform.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ISAAC_UNIFORM), "exec"), namespace)
        setattr(UniformShell, name, namespace[name])
    namespace["UniformVelocityCommand"] = UniformShell
    tree = ast.parse(PRODUCTION.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PRODUCTION), "exec"), namespace)
    return SimpleNamespace(**namespace)


def make_env(num_envs=4, reference=True):
    module = load_production()
    data = SimpleNamespace(heading_w=torch.zeros(num_envs), root_lin_vel_w=torch.zeros(num_envs, 3),
                           root_lin_vel_b=torch.zeros(num_envs, 3), root_ang_vel_b=torch.zeros(num_envs, 3))
    env = SimpleNamespace(num_envs=num_envs, device="cpu", _sim_step_counter=0, physics_dt=.005,
                          step_dt=.02, scene={"robot": SimpleNamespace(data=data)})
    cfg = SimpleNamespace(asset_name="robot", heading_command=False, rel_heading_envs=0.,
                          heading_control_stiffness=.5, rel_standing_envs=.02,
                          resampling_time_range=(10., 10.), ranges=SimpleNamespace(
                              lin_vel_x=(-1.2, 1.2), lin_vel_y=(-.8, .8), ang_vel_z=(-.5, .5), heading=None))
    term = (module.HeadingReferenceVelocityCommand if reference else UniformShell)(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda _: term, get_command=lambda _: term.vel_command_b)
    return module, env, term


def test_uniform_draws_commands_and_rng_state_are_unchanged():
    _, base_env, base = make_env(8192, reference=False)
    _, env, term = make_env(8192)
    assert "_resample_command" not in vars(type(term))
    for counter in range(3):
        torch.manual_seed(42 + counter)
        base.reset(torch.arange(8192))
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(42 + counter)
        term.reset(torch.arange(8192))
        torch.testing.assert_close(term.vel_command_b, base.vel_command_b)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng)
        torch.testing.assert_close(term.is_standing_env, base.is_standing_env)


def test_physics_clock_reward_observation_and_compute_do_not_double_integrate():
    module, env, term = make_env()
    term.vel_command_b[:, 2] = torch.tensor([0., .5, -.5, .1])
    env._sim_step_counter = 4
    expected = torch.tensor([0., .01, -.01, .002])
    for _ in range(4):
        torch.testing.assert_close(term.get_heading_reference(), expected, atol=2e-7, rtol=0)
        module.heading_frame_error(env, "base_velocity")
        module.heading_frame_lin_vel_xy_exp(env, "base_velocity", .4)
        term.compute(.02)
    env._sim_step_counter = 8
    torch.testing.assert_close(term.get_heading_reference(), expected * 2, atol=3e-7, rtol=0)


def test_partial_reference_reset_never_integrates_prior_episode_interval():
    _, env, term = make_env()
    term.vel_command_b[:, 2] = .5
    env._sim_step_counter = 4
    env.scene["robot"].data.heading_w[:] = torch.tensor([.7, -.8, 2., -2.])
    term.reset(torch.tensor([0, 2]))
    target = term.get_heading_reference()
    torch.testing.assert_close(target, torch.tensor([.7, .01, 2., .01]), atol=2e-7, rtol=0)
    term.compute(.02)
    torch.testing.assert_close(target[[0, 2]], torch.tensor([.7, 2.]))
    term.reset()
    torch.testing.assert_close(term.heading_target, env.scene["robot"].data.heading_w)


def test_resampling_uses_old_command_for_elapsed_interval_and_preserves_reference():
    _, env, term = make_env()
    term.vel_command_b[:, 2] = .5
    term.resample_next = True
    env._sim_step_counter = 4
    term.compute(.02)
    torch.testing.assert_close(term.heading_target, torch.full((4,), .01), atol=2e-7, rtol=0)
    new_yaw_command = term.vel_command_b[:, 2].clone()
    env._sim_step_counter = 8
    torch.testing.assert_close(term.get_heading_reference(), .01 + new_yaw_command * .02, atol=3e-7, rtol=0)


def test_reference_does_not_follow_measured_yaw_drift():
    module, env, term = make_env(1)
    term.vel_command_b[0] = torch.tensor([.6, 0., 0.])
    env.scene["robot"].data.root_lin_vel_w[0] = torch.tensor([.6, 0., 0.])
    torch.testing.assert_close(module.heading_frame_lin_vel_xy_exp(env, "base_velocity", .4), torch.ones(1))
    env.scene["robot"].data.heading_w.fill_(torch.pi / 2)
    env.scene["robot"].data.root_lin_vel_w[0] = torch.tensor([0., .6, 0.])
    env._sim_step_counter = 4
    assert module.heading_frame_lin_vel_xy_exp(env, "base_velocity", .4).item() < .02
    torch.testing.assert_close(module.heading_frame_error(env, "base_velocity"), torch.tensor([[-1., 0.]]), atol=2e-7, rtol=0)
    assert term.heading_target.item() == 0.


def test_commanded_turn_rotates_reference_and_critic_explains_reward():
    module, env, term = make_env(1)
    term.vel_command_b[0] = torch.tensor([.6, .1, .5])
    env._sim_step_counter = 40
    reference = .1
    cosine, sine = torch.cos(torch.tensor(reference)), torch.sin(torch.tensor(reference))
    env.scene["robot"].data.root_lin_vel_w[0] = torch.tensor([cosine*.6-sine*.1, sine*.6+cosine*.1, 0.])
    env.scene["robot"].data.heading_w.fill_(reference)
    velocity = module.heading_frame_lin_vel_xy(env, "base_velocity")
    torch.testing.assert_close(velocity, torch.tensor([[.6, .1]]))
    torch.testing.assert_close(module.heading_frame_error(env, "base_velocity"), torch.tensor([[0., 1.]]), atol=2e-7, rtol=0)
    expected = torch.exp(-((term.vel_command_b[:, :2]-velocity)**2).sum(-1)/.4**2)
    torch.testing.assert_close(module.heading_frame_lin_vel_xy_exp(env, "base_velocity", .4), expected)
    with pytest.raises(ValueError, match="positive"):
        module.heading_frame_lin_vel_xy_exp(env, "base_velocity", 0.)


def saved():
    path = ROOT / "logs/rsl_rl/bsrl_amp/2026-09-15_15-14-55_reduced_parallelankle_nominal_pd200_noestimator_mirror01_formal1_8192/params"
    if not path.is_dir():
        pytest.skip("Local frozen reduced baseline required")
    return load_saved_config(path / "env.yaml"), load_saved_config(path / "agent.yaml")


def test_whitelisted_change_keeps_actor_mirror_sampling_and_dynamics():
    env, agent = saved()
    validate_reduced_heading_baseline(env, agent)
    e, a = expected_heading_configuration(env, agent, "new_heading_run")
    assert e["observations"]["policy"] == env["observations"]["policy"]
    assert a["algorithm"] == agent["algorithm"]
    assert e["scene"] == env["scene"] and e["sim"] == env["sim"]
    assert e["commands"]["base_velocity"]["ranges"] == env["commands"]["base_velocity"]["ranges"]
    check_heading_configuration(e, a, e, a)
    actual = copy.deepcopy(e)
    actual["log_dir"] = "new_output"
    check_heading_configuration(e, a, actual, a)


@pytest.mark.parametrize("change", ["actor", "mirror", "pd", "sampling", "smoothness"])
def test_unrequested_change_rejected(change):
    env, agent = saved()
    expected_e, expected_a = expected_heading_configuration(env, agent, "new_run")
    e, a = copy.deepcopy(expected_e), copy.deepcopy(expected_a)
    if change == "actor":
        e["observations"]["policy"]["heading_error"] = {"func": "unexpected"}
    elif change == "mirror":
        a["algorithm"]["symmetry_cfg"]["mirror_loss_coeff"] = .2
    elif change == "pd":
        e["scene"]["robot"]["actuators"]["parallel_ankles"]["stiffness"] = 200
    elif change == "sampling":
        e["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = (0., 0.)
    else:
        e["rewards"]["action_rate_l2"]["weight"] = -.005
    with pytest.raises(ValueError, match="Unexpected change"):
        check_heading_configuration(expected_e, expected_a, e, a)


@pytest.mark.parametrize("change", ["topology", "mirror", "actor_heading", "sampling"])
def test_wrong_baseline_rejected(change):
    env, agent = saved()
    if change == "topology":
        env["scene"]["robot"]["actuators"].pop("parallel_ankles")
    elif change == "mirror":
        agent["algorithm"]["symmetry_cfg"] = None
    elif change == "actor_heading":
        env["observations"]["policy"]["heading_error"] = {"func": "unexpected"}
    else:
        env["commands"]["base_velocity"]["heading_command"] = True
    with pytest.raises(ValueError):
        validate_reduced_heading_baseline(env, agent)
