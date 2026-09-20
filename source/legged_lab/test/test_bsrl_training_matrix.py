"""CPU regression checks for the three-run experimental contract."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from bsrl_training_matrix import expected_configuration, command_motion_patterns, VARIANTS, UNIFORM_COMMAND
from t1_heading_ablation import load_saved_config, configuration_differences

BASE = ROOT / "logs/rsl_rl/bsrl_amp/2026-09-17_18-18-30_accad34_reduced_latent20_h10_recent2_predict33_kl001_nomirror_8192"
SPEC = importlib.util.spec_from_file_location("motion_categories", ROOT / "source/legged_lab/legged_lab/managers/motion_command_categories.py")
CATEGORIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CATEGORIES)


@pytest.fixture
def saved():
    return load_saved_config(BASE / "params/env.yaml"), load_saved_config(BASE / "params/agent.yaml")


def configured(saved, variant):
    return expected_configuration(*saved, variant=variant, run_name="test_matrix", num_envs=8192,
                                  max_iterations=40000, device="cuda:0")


@pytest.mark.parametrize("variant", VARIANTS)
def test_training_contract_preserves_network_data_and_ankles(saved, variant):
    old, old_agent = copy.deepcopy(saved)
    env, agent = configured(saved, variant)
    assert agent["max_iterations"] == 40000 and not agent["resume"]
    assert agent["policy"] == old_agent["policy"]
    assert agent["algorithm"] == old_agent["algorithm"]
    assert not agent["algorithm"]["state_est_loss_required"] and agent["algorithm"]["symmetry_cfg"] is None
    for key in ("events", "motion_data", "sim", "curriculum", "terminations"):
        assert configuration_differences(old[key], env[key]) == []
    assert env["scene"]["robot"]["actuators"]["parallel_ankles"] == old["scene"]["robot"]["actuators"]["parallel_ankles"]
    assert env["observations"]["policy"] == old["observations"]["policy"]
    assert saved == (old, old_agent)


def test_scale01_changes_only_action_scale(saved):
    old, _ = saved
    env, _ = configured(saved, "scale01")
    differences = configuration_differences(old, env)
    assert all(d['path'].startswith('actions.joint_pos.scale.') or d['path']=='log_dir' for d in differences)
    assert set(env['actions']['joint_pos']['scale'].values()) == {.1}


@pytest.mark.parametrize("variant", ("matched_bigpd", "clean_bigpd"))
def test_bigpd_hips_and_knee_only(saved, variant):
    env, _ = configured(saved, variant)
    motors = env['scene']['robot']['actuators']
    assert motors['large_motors']['stiffness'] == {'joint_.*_hip_pitch':300, 'joint_.*_hip_roll':300, 'joint_.*_knee_pitch':400}
    assert set(motors['large_motors']['damping'].values()) == {6}
    assert set(motors['small_motors']['stiffness'].values()) == {300}
    assert set(motors['small_motors']['damping'].values()) == {6}
    assert set(env['actions']['joint_pos']['scale'].values()) == {.25}


def test_matched_references_cover_all_34_without_overlap(saved):
    env, _ = configured(saved, 'matched_bigpd')
    a=env['animation']['animation']
    patterns=a['command_motion_patterns']; names=[name for values in patterns.values() for name in values]
    assert len(names)==len(set(names))==34
    assert set(names)==set(env['motion_data']['motion_dataset']['motion_data_weights'])
    assert a['command_name']=='base_velocity' and not a['match_command_speed']
    assert patterns['standing']==('stand_male2_A1',)
    assert all('turn' not in name.lower() for name in patterns['forward'])
    assert env['observations']['disc'] == saved[0]['observations']['disc']
    assert env['rewards']==saved[0]['rewards']


def test_clean_no_heading_or_conditional_amp(saved):
    env, agent = configured(saved, 'clean_bigpd')
    assert env['commands']['base_velocity']['class_type']==UNIFORM_COMMAND
    assert env['animation']['animation']['command_name'] is None
    assert env['rewards']['heading_frame_lin_vel_xy_exp'] is None
    assert env['observations']['critic']['heading_frame_error'] is None
    assert env['observations']['critic']['heading_frame_lin_vel_xy'] is None
    for name,value in saved[0]['rewards'].items():
        if name!='heading_frame_lin_vel_xy_exp': assert env['rewards'][name]==value


def test_seven_way_directions_and_normalized_dominant_axis():
    probes=torch.tensor([[.5,0,0],[-.5,0,0],[0,.4,0],[0,-.4,0],[0,0,.3],[0,0,-.3],[0,0,0],[.8,0,.49]])
    ids,_=CATEGORIES.command_category_ids(probes,(1.2,.8,.5),standing=True)
    assert ids.tolist()==[0,1,2,3,4,5,6,4]


def test_legacy_six_way_results_unchanged():
    generator=torch.Generator().manual_seed(27)
    cmd=torch.randn(500,3,generator=generator); cmd[0]=0
    scales=(.8,.6,2.05)
    scores=torch.stack([cmd[:,axis].mul(sign).clamp_min(0)/scales[axis]
                        for axis in range(3) for sign in (1,-1)],dim=1)
    ids,speed=CATEGORIES.command_category_ids(cmd,scales)
    torch.testing.assert_close(ids,scores.argmax(1))
    torch.testing.assert_close(speed,scores.max(1).values)


def test_reject_unknown_motion_or_excess_iterations(saved):
    with pytest.raises(ValueError,match='Unclassified'):
        command_motion_patterns(['mystery_clip'])
    with pytest.raises(ValueError,match='40000'):
        expected_configuration(*saved,variant='scale01',run_name='bad',num_envs=8192,max_iterations=40001,device='cuda:0')
