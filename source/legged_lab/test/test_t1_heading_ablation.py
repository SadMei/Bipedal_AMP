"""CPU-only tests of the real T1 symmetry code and saved-baseline guard."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYMMETRY = _load("t1_symmetry_under_test", ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/t1.py")
ABLATION = _load("t1_ablation_under_test", ROOT / "scripts/rsl_rl/t1_heading_ablation.py")
BASELINE = ROOT / "logs/rsl_rl/men_t1_amp/2026-09-14_21-52-57_resetmatch_noestimator_mirror01_noclip_jointlimit5_formal1_8192"


@pytest.mark.parametrize("width", [435, 445])
def test_mirror_involution_and_privileged_state_unchanged(width):
    obs = TensorDict({"policy": torch.randn(8, width), "critic": torch.randn(8, 550)}, batch_size=[8])
    actions = torch.randn(8, 25)
    obs_before, actions_before = obs.clone(), actions.clone()
    augmented, augmented_actions = SYMMETRY.compute_symmetric_states(None, obs, actions)
    twice, twice_actions = SYMMETRY.compute_symmetric_states(None, augmented[8:], augmented_actions[8:])
    torch.testing.assert_close(twice["policy"][8:], obs["policy"])
    torch.testing.assert_close(twice_actions[8:], actions)
    torch.testing.assert_close(augmented["critic"][8:], obs["critic"])
    torch.testing.assert_close(obs["policy"], obs_before["policy"])
    torch.testing.assert_close(actions, actions_before)


def test_no_heading_is_exact_legacy_prefix():
    observation = torch.randn(4, 445)
    full = SYMMETRY._mirror_policy_observation(observation)
    short = SYMMETRY._mirror_policy_observation(observation[:, :435])
    torch.testing.assert_close(full[:, :435], short)
    torch.testing.assert_close(full[:, 435::2], -observation[:, 435::2])
    torch.testing.assert_close(full[:, 436::2], observation[:, 436::2])
    torch.testing.assert_close(short[:, 45:60].reshape(4, 5, 3),
                               observation[:, 45:60].reshape(4, 5, 3) * torch.tensor([1., -1., -1.]))


@pytest.mark.parametrize("width", [434, 436, 444, 446, 525])
def test_bad_layout_rejected(width):
    with pytest.raises(ValueError, match="policy observation dim"):
        SYMMETRY._mirror_policy_observation(torch.randn(2, width))


def test_optional_none_and_sequence_representation_are_equivalent():
    assert ABLATION.configuration_differences({"a": (1, 2)}, {"a": [1, 2], "optional": None}) == []
    assert ABLATION.configuration_differences({"a": None}, {"a": {"weight": 0.1}})[0]["path"] == "a"


def test_scene_substitutions_match_isaac_without_mutating_config():
    data = {"scene": {"num_envs": 8192, "env_spacing": 2.5,
                      "robot": {"prim_path": "{ENV_REGEX_NS}/Robot"},
                      "terrain": {"num_envs": 1, "env_spacing": None}}}
    before = copy.deepcopy(data)
    result = ABLATION.resolved_scene_config(data)
    assert result["scene"]["robot"]["prim_path"] == "/World/envs/env_.*/Robot"
    assert result["scene"]["terrain"] == {"num_envs": 8192, "env_spacing": 2.5}
    assert data == before


class _Config(SimpleNamespace):
    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, dict):
            return cls(**{key: cls.from_mapping(item) for key, item in value.items()})
        return copy.deepcopy(value)

    def to_dict(self):
        return {key: value.to_dict() if isinstance(value, _Config) else copy.deepcopy(value)
                for key, value in vars(self).items()}


def _saved_configs():
    if not (BASELINE / "params/env.yaml").is_file():
        pytest.skip("The frozen local T1 baseline is not distributed with every checkout.")
    env = _Config.from_mapping(ABLATION.load_saved_config(BASELINE / "params/env.yaml"))
    agent = _Config.from_mapping(ABLATION.load_saved_config(BASELINE / "params/agent.yaml"))
    agent.run_name = "test_no_actor_heading"
    env.log_dir = None
    env.rewards.action_rate_l2.weight = -0.1
    env.rewards.dof_acc_l2.weight = -1.0e-6
    env.rewards.action_smoothness = _Config(weight=-0.001)
    return env, agent


def test_only_heading_changes_and_old_smoothness_restored():
    env, agent = _saved_configs()
    audit = ABLATION.apply_no_actor_heading_ablation(env, agent, str(BASELINE))
    assert env.observations.policy.heading_error is None
    assert env.observations.critic.heading_error.func.endswith(":integrated_heading_error")
    assert env.rewards.action_rate_l2.weight == -0.005
    assert env.rewards.dof_acc_l2.weight == -1.0e-7
    assert env.rewards.action_smoothness is None
    assert audit["unrequested_configuration_differences"] == []
    assert audit["policy_observation_dim"] == 435


@pytest.mark.parametrize("change", ["mirror", "reward", "num_envs", "critic_heading"])
def test_unrequested_changes_abort(change):
    env, agent = _saved_configs()
    if change == "mirror":
        agent.algorithm.symmetry_cfg.mirror_loss_coeff = 0.2
    elif change == "reward":
        env.rewards.track_ang_vel_z_exp.weight = 2.0
    elif change == "num_envs":
        env.scene.num_envs = 4096
    else:
        env.observations.critic.heading_error = None
    with pytest.raises(ValueError, match="Unexpected changes"):
        ABLATION.apply_no_actor_heading_ablation(env, agent, str(BASELINE))
