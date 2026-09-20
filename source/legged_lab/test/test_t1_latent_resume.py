"""Continuation must retain the saved experiment and stop at total 40000."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from t1_heading_ablation import load_saved_config, configuration_differences
from t1_latent_resume import (continuation_counts, expected_configuration, restore_training_clock,
                              assert_state_equal, audit_loaded_runner)

BASE = ROOT / "logs/rsl_rl/men_t1_amp/2026-09-17_18-03-20_latent20_h10_recent2_predict59_kl001_nomirror_noactorheading_8192"


def test_total_iterations_off_by_one():
    start, remaining = continuation_counts(24999, 40000)
    assert (start, remaining) == (25000, 15000)
    assert range(start, start + remaining)[-1] == 39999
    assert continuation_counts(24999, 25003) == (25000, 3)


@pytest.mark.parametrize("last,total", [(-1, 40000), (24999, 25000), (24999, 65000), (39999, 40000)])
def test_invalid_cap_rejected(last, total):
    with pytest.raises(ValueError):
        continuation_counts(last, total)


@pytest.fixture
def configs():
    if not BASE.is_dir():
        pytest.skip("Saved local latent configuration unavailable")
    env, agent = load_saved_config(BASE / "params/env.yaml"), load_saved_config(BASE / "params/agent.yaml")
    options = dict(checkpoint=BASE / "model_24999.pt", run_name="resume_test", total_iterations=40000,
                   num_envs=8192, device="cuda:0", log_dir="/tmp/resume_test", completed=25000)
    return env, agent, options


def test_preserve_full_policy_algorithm_and_all_nonruntime_conditions(configs):
    old_env, old_agent, options = configs
    env, agent = expected_configuration(old_env, old_agent, **options, apply_curriculum=False)
    assert agent["policy"] == old_agent["policy"]
    assert agent["algorithm"] == old_agent["algorithm"]
    assert agent["obs_groups"] == old_agent["obs_groups"]
    assert agent["resume"] and agent["max_iterations"] == 40000
    for key in old_env:
        if key != "log_dir":
            assert not configuration_differences(old_env[key], env[key]), key
    assert old_env == load_saved_config(BASE / "params/env.yaml")


def test_resume_curriculum_is_phase_three_not_phase_one(configs):
    old_env, old_agent, options = configs
    env, _ = expected_configuration(old_env, old_agent, **options, apply_curriculum=True)
    assert env["commands"]["base_velocity"]["ranges"]["lin_vel_x"] == [-.7, .8]
    assert env["commands"]["base_velocity"]["ranges"]["lin_vel_y"] == [-.6, .5]
    assert env["commands"]["base_velocity"]["ranges"]["ang_vel_z"] == [-2.05, 1.9]
    assert env["rewards"]["track_lin_vel_xy_exp"]["params"]["std"] == .4
    assert env["observations"]["critic"]["heading_error"] is not None
    assert env["observations"]["policy"]["heading_error"] is None
    assert env["rewards"]["zero_yaw_initial_heading_lin_vel_xy_exp"] is not None
    assert env["commands"]["base_velocity"]["straight_command_probability"] == .5
    assert env["observations"]["disc"]["velocity_commands"] is not None
    assert env["curriculum"] == old_env["curriculum"]


def test_clock_drives_real_curriculum_before_first_command(configs):
    env_cfg, _, _ = configs
    params = copy.deepcopy(env_cfg["curriculum"]["velocity_range_and_tracking_std"]["params"])
    source = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/curriculums.py"
    spec = importlib.util.spec_from_file_location("resume_test_curriculum", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cmd = NS(cfg=NS(ranges=NS()))
    reward = NS(params={"std": .5})
    env = NS(cfg=NS(decimation=4), command_manager=NS(get_term=lambda name: cmd),
             reward_manager=NS(get_term_cfg=lambda name: reward, set_term_cfg=lambda name, value: None))
    restore_training_clock(env, 25000, 24)
    info = module.velocity_range_and_tracking_std(env, [], **params)
    assert env.common_step_counter == 600000 and env._sim_step_counter == 2400000
    assert info["phase"] == 3 and info["iteration"] == 25000
    assert cmd.cfg.ranges.lin_vel_x == [-.7, .8] and reward.params["std"] == .4


def test_reject_other_policy_and_mirror(configs):
    env, agent, options = configs
    agent["algorithm"]["symmetry_cfg"] = {"use_mirror_loss": True}
    with pytest.raises(ValueError, match="no-mirror"):
        expected_configuration(env, agent, **options, apply_curriculum=False)


def test_exact_tensor_and_optimizer_checks():
    state = {"weights": torch.randn(2, 3), "state": {0: {"step": torch.tensor(17.), "exp_avg": torch.randn(3)}}}
    assert_state_equal(copy.deepcopy(state), state)
    wrong = copy.deepcopy(state)
    wrong["state"][0]["exp_avg"][0] += 1
    with pytest.raises(AssertionError):
        assert_state_equal(wrong, state)


def test_learning_rate_scalar_and_next_index_restored():
    p = torch.nn.Linear(3, 2)
    disc = torch.nn.Linear(3, 1)
    disc.disc_obs_normalizer = torch.nn.BatchNorm1d(3)
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-5)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=2e-6)
    alg = NS(policy=p, amp_discriminator=disc, optimizer=optimizer, disc_optimizer=disc_optimizer, learning_rate=1e-4)
    runner = NS(alg=alg, current_learning_iteration=24999)
    ckpt = {"iter": 24999, "model_state_dict": p.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "amp_discriminator_state_dict": disc.state_dict(),
            "amp_discriminator_normalizer_state_dict": disc.disc_obs_normalizer.state_dict(),
            "amp_discriminator_optimizer_state_dict": disc_optimizer.state_dict()}
    audit = audit_loaded_runner(runner, ckpt, 25000)
    assert runner.current_learning_iteration == 25000 and alg.learning_rate == 1e-5
    assert len(audit["exactly_restored_states"]) == 5
    assert not audit["replay_buffers_restored"] and not audit["simulator_or_rng_state_restored"]
