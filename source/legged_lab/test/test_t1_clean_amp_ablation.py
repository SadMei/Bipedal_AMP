"""CPU checks for the requested clean-AMP comparison and baseline preservation."""
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from t1_heading_ablation import load_saved_config, configuration_differences
from t1_clean_amp_ablation import expected_configuration, validate_baseline, COMMAND_RANGES, UNIFORM_COMMAND

BASELINE = ROOT / "logs/rsl_rl/men_t1_amp/2026-09-17_18-03-20_latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192"


@pytest.fixture
def configs():
    if not BASELINE.is_dir():
        pytest.skip("Local frozen T1 latent baseline unavailable")
    old_env = load_saved_config(BASELINE / "params/env.yaml")
    old_agent = load_saved_config(BASELINE / "params/agent.yaml")
    env, agent = expected_configuration(old_env, old_agent, run_name="clean_test", num_envs=8192,
                                        max_iterations=40000, device="cuda:0")
    return old_env, old_agent, env, agent


def test_fixed_independent_ranges_and_no_curriculum(configs):
    _, _, env, _ = configs
    cmd = env["commands"]["base_velocity"]
    assert cmd["class_type"] == UNIFORM_COMMAND
    for key, value in COMMAND_RANGES.items():
        assert cmd["ranges"][key] == value
    assert cmd["ranges"]["heading"] is None
    assert not cmd["heading_command"] and cmd["rel_heading_envs"] == 0
    assert cmd["straight_command_probability"] == 0
    assert not cmd["single_axis_commands"] and cmd["motion_speed_animation"] is None
    assert env["curriculum"]["velocity_range_and_tracking_std"] is None


def test_heading_removed_from_actor_and_critic(configs):
    old, _, env, agent = configs
    assert old["observations"]["critic"]["heading_error"] is not None
    for group in ("policy", *agent["obs_groups"]["critic"]):
        assert env["observations"][group].get("heading_error") is None
    assert env["rewards"]["zero_yaw_initial_heading_lin_vel_xy_exp"] is None


def test_unconditional_amp_both_sides_and_reference_sampler(configs):
    old, _, env, _ = configs
    assert old["observations"]["disc"]["velocity_commands"] is not None
    assert env["observations"]["disc"]["velocity_commands"] is None
    assert env["observations"]["disc_demo"]["ref_velocity_commands"] is None
    assert env["animation"]["animation"]["command_name"] is None
    assert env["animation"]["animation"]["command_motion_patterns"] is None
    assert not env["animation"]["animation"]["match_command_speed"]
    for group, term in (("disc", "base_lin_vel"), ("disc_demo", "ref_root_lin_vel_b")):
        assert env["observations"][group][term] == old["observations"][group][term]


def test_latent_and_ppo_settings_preserved(configs):
    old_env, old_agent, env, agent = configs
    assert agent["policy"] == old_agent["policy"]
    assert agent["algorithm"] == old_agent["algorithm"]
    assert agent["algorithm"]["symmetry_cfg"] is None
    assert not agent["algorithm"]["state_est_loss_required"]
    assert env["observations"]["policy"] == old_env["observations"]["policy"]
    assert agent["resume"] is False
    assert agent["max_iterations"] == 40000


@pytest.mark.parametrize("key", ["scene", "sim", "events", "actions", "motion_data", "terminations"])
def test_physics_randomization_data_and_reset_unchanged(configs, key):
    old, _, env, _ = configs
    assert configuration_differences(old[key], env[key]) == []


def test_other_rewards_and_privileged_contact_features_preserved(configs):
    old, _, env, _ = configs
    for key, value in old["rewards"].items():
        if key == "track_lin_vel_xy_exp":
            expected = copy.deepcopy(value)
            expected["params"]["std"] = old["curriculum"]["velocity_range_and_tracking_std"]["params"]["tracking_stds"][0]
            assert expected["params"]["std"] == 0.4
            assert env["rewards"][key] == expected
        elif key != "zero_yaw_initial_heading_lin_vel_xy_exp":
            assert env["rewards"][key] == value
    for group in ("reward_privileged", "state_est_critic"):
        assert env["observations"][group] == old["observations"][group]


def test_baseline_not_mutated(configs):
    old_env, old_agent, _, _ = configs
    assert old_env == load_saved_config(BASELINE / "params/env.yaml")
    assert old_agent == load_saved_config(BASELINE / "params/agent.yaml")
    assert old_env["commands"]["base_velocity"]["straight_command_probability"] == 0.5


def test_nonlatent_baseline_rejected(configs):
    env, agent, _, _ = configs
    agent = copy.deepcopy(agent)
    agent["policy"]["class_name"] = "ActorCritic"
    with pytest.raises(ValueError, match="frozen T1 latent"):
        validate_baseline(env, agent)


def test_extra_heading_frame_fields_disabled_if_present(configs):
    env, agent, _, _ = configs
    env = copy.deepcopy(env)
    env["observations"]["critic"]["heading_frame_error"] = {"func": "test"}
    env["observations"]["critic"]["heading_frame_lin_vel_xy"] = {"func": "test"}
    env["rewards"]["heading_frame_lin_vel_xy_exp"] = {"weight": 1}
    new, _ = expected_configuration(env, agent, run_name="test", num_envs=8192,
                                     max_iterations=25000, device="cuda:0")
    assert new["observations"]["critic"]["heading_frame_error"] is None
    assert new["observations"]["critic"]["heading_frame_lin_vel_xy"] is None
    assert new["rewards"]["heading_frame_lin_vel_xy_exp"] is None
