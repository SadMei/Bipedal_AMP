"""CPU-only tests for exact restoration rather than config-map merging."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from frozen_run_config import _restore_value, disable_training_command_sampling
from t1_heading_ablation import configuration_differences, load_saved_config


class Config(SimpleNamespace):
    def to_dict(self):
        def encode(value):
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return copy.deepcopy(value)
        return {key: encode(value) for key, value in vars(self).items()}


def test_restore_optional_clip_and_replace_parameter_map():
    cfg = Config(clip=None, params={"old": 1, "new_source_only": 4}, extra_reward=Config(weight=-1))
    saved = {"clip": {"joint": (-1., 1.)}, "params": {"old": 2}}
    _restore_value(cfg, saved)
    assert cfg.clip == {"joint": (-1., 1.)}
    assert cfg.params == {"old": 2}
    assert cfg.extra_reward is None
    assert configuration_differences(saved, cfg.to_dict()) == []


def test_nested_config_objects_keep_types():
    child = Config(weight=-0.1, params={"a": 2})
    cfg = Config(child=child)
    _restore_value(cfg, {"child": {"weight": -0.005, "params": {}}})
    assert cfg.child is child
    assert child.weight == -0.005 and child.params == {}


def test_unknown_typed_field_fails_closed():
    with pytest.raises(KeyError, match="typed template"):
        _restore_value(Config(known=1), {"missing": 2})


def test_no_mutable_alias_to_saved_snapshot():
    saved = {"values": [1, 2], "params": {"nested": [3, 4]}}
    cfg = Config(values=[], params={})
    _restore_value(cfg, saved)
    cfg.values.append(7)
    cfg.params["nested"].append(9)
    assert saved == {"values": [1, 2], "params": {"nested": [3, 4]}}


def test_fixed_command_disables_all_training_sampling_layers():
    cfg = Config(motion_speed_animation="animation", single_axis_commands=True,
                 straight_command_probability=0.5, ranges={"lin_vel_x": (0.6, 0.6)})
    disable_training_command_sampling(cfg)
    assert cfg.motion_speed_animation is None
    assert cfg.single_axis_commands is False
    assert cfg.straight_command_probability == 0.0
    assert cfg.ranges == {"lin_vel_x": (0.6, 0.6)}
    plain = Config(ranges={"lin_vel_x": (0.6, 0.6)})
    disable_training_command_sampling(plain)
    assert not hasattr(plain, "motion_speed_animation")


def test_real_bsrl_baseline_is_direct_serial_mirror_without_heading_reward():
    path = ROOT / "logs/rsl_rl/bsrl_amp/2026-09-14_17-51-40_pd_hip150_knee200_kd2_ankle100kd1_noestimator_mirror01_formal1_8192/params"
    if not path.is_dir():
        pytest.skip("Local frozen baseline not included in this checkout")
    env, agent = load_saved_config(path / "env.yaml"), load_saved_config(path / "agent.yaml")
    assert set(env["scene"]["robot"]["actuators"]) == {"large_motors", "small_motors"}
    assert env["observations"]["policy"].get("heading_error") is None
    assert env["rewards"].get("zero_yaw_initial_heading_lin_vel_xy_exp") is None
    assert agent["algorithm"]["symmetry_cfg"]["use_mirror_loss"]
    assert agent["policy"]["class_name"] == "ActorCritic"
