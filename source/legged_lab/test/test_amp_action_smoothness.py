"""CPU-only regression of the production reward kernel, without starting Kit.

Extract the actual class AST to avoid importing the simulator-dependent MDP
package. Only ManagerTermBase's storage-only constructor is substituted; the
production init, reward computation and reset methods execute unchanged.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PACKAGE = Path(__file__).resolve().parents[1] / "legged_lab"
AMP = PACKAGE / "tasks/locomotion/amp"


class _TermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def _load_reward_class():
    path = AMP / "mdp/rewards.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionSecondOrderL2")
    namespace = {
        "torch": torch,
        "ManagerTermBase": _TermBase,
        "RewardTermCfg": SimpleNamespace,
        "ManagerBasedRLEnv": SimpleNamespace,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ActionSecondOrderL2"]


def _make_term(num_envs, action_dim):
    env = SimpleNamespace(action_manager=SimpleNamespace(
        action=torch.zeros(num_envs, action_dim),
        prev_action=torch.zeros(num_envs, action_dim),
    ))
    return _load_reward_class()(SimpleNamespace(weight=-0.001), env), env


@pytest.mark.parametrize("action_dim", [12, 25])
def test_linear_ramp_and_constant_large_target(action_dim):
    term, env = _make_term(3, action_dim)
    unit = torch.arange(1, 4, dtype=torch.float32).unsqueeze(-1).expand(3, action_dim)
    for multiplier, expected_factor in [(1, 1), (2, 0), (3, 0), (3, 1), (3, 0)]:
        env.action_manager.prev_action.copy_(env.action_manager.action)
        env.action_manager.action.copy_(multiplier * unit)
        action_before = env.action_manager.action.clone()
        previous_before = env.action_manager.prev_action.clone()
        result = term(env)
        torch.testing.assert_close(result, expected_factor * unit.square().sum(-1))
        torch.testing.assert_close(env.action_manager.action, action_before)
        torch.testing.assert_close(env.action_manager.prev_action, previous_before)
        assert term._prev_prev_action.data_ptr() != env.action_manager.prev_action.data_ptr()


@pytest.mark.parametrize("action_dim", [12, 25])
def test_random_8192_env_sequence_and_partial_episode_resets(action_dim):
    term, env = _make_term(8192, action_dim)
    generator = torch.Generator().manual_seed(42)
    expected_prev = torch.zeros_like(env.action_manager.action)
    expected_prev_prev = torch.zeros_like(expected_prev)
    for step in range(10):
        if step in (3, 7):
            env_ids = torch.arange(step, 8192, 7)
            untouched = term._prev_prev_action.clone()
            env.action_manager.action[env_ids] = 0
            env.action_manager.prev_action[env_ids] = 0
            term.reset(env_ids)
            untouched[env_ids] = 0
            torch.testing.assert_close(term._prev_prev_action, untouched)
            expected_prev[env_ids] = 0
            expected_prev_prev[env_ids] = 0
        action = torch.randn(expected_prev.shape, generator=generator)
        env.action_manager.prev_action.copy_(env.action_manager.action)
        env.action_manager.action.copy_(action)
        expected = (action - 2 * expected_prev + expected_prev_prev).square().sum(-1)
        actual = term(env)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (8192,) and torch.isfinite(actual).all()
        expected_prev_prev.copy_(expected_prev)
        expected_prev.copy_(action)


@pytest.mark.parametrize("env_ids", [None, slice(None), [], [0, 2]])
def test_reset_clears_only_selected_history(env_ids):
    term, env = _make_term(3, 12)
    env.action_manager.prev_action.fill_(7)
    term(env)
    expected = term._prev_prev_action.clone()
    expected[slice(None) if env_ids is None else env_ids] = 0
    term.reset(env_ids)
    torch.testing.assert_close(term._prev_prev_action, expected)


def test_different_reward_instances_do_not_share_history():
    first, env = _make_term(3, 12)
    second, _ = _make_term(3, 12)
    env.action_manager.prev_action.fill_(7)
    first(env)
    assert first._prev_prev_action.data_ptr() != second._prev_prev_action.data_ptr()
    assert torch.count_nonzero(second._prev_prev_action) == 0


@pytest.mark.parametrize("robot", ["t1", "bsrl"])
def test_configuration_weights_and_no_action_clipping(robot):
    path = AMP / f"config/{robot}/{robot}_amp_env_cfg.py"
    tree = ast.parse(path.read_text())
    compile(tree, str(path), "exec")
    assignments = {ast.unparse(n.targets[0]): n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)}
    expected = {"action_rate_l2": -0.1, "dof_acc_l2": -1.0e-6, "dof_pos_limits": -5.0}
    for name, weight in expected.items():
        if robot == "bsrl":
            value = assignments[f"self.rewards.{name}.weight"]
        else:
            value = next(k.value for k in assignments[name].keywords if k.arg == "weight")
        assert ast.literal_eval(value) == weight
    smoothness = assignments["self.rewards.action_smoothness" if robot == "bsrl" else "action_smoothness"]
    kwargs = {k.arg: k.value for k in smoothness.keywords}
    assert ast.literal_eval(kwargs["weight"]) == -0.001
    assert ast.unparse(kwargs["func"]) == ("amp_mdp" if robot == "bsrl" else "mdp") + ".ActionSecondOrderL2"
    assert ast.literal_eval(assignments["self.actions.joint_pos.clip"]) is None


def test_g1_and_shared_base_do_not_enable_the_new_reward():
    for path in (AMP / "amp_env_cfg.py", AMP / "config/g1/g1_amp_env_cfg.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(n, ast.Attribute) and n.attr == "ActionSecondOrderL2" for n in ast.walk(tree))
