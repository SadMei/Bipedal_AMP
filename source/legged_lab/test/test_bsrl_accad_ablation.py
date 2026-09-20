"""CPU contract tests for paired ACCAD analytical-parallel experiments."""
from pathlib import Path
import sys
import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
sys.path.insert(0,str(ROOT/'scripts/tools/retarget'))
from bsrl_accad_ablation import expected_configuration, validate_baseline, check_configuration, load_dataset, runtime_check
from t1_heading_ablation import load_saved_config, configuration_differences
from export_accad_amp import training_payload
from bsrl_conventions import BSRL_AMP_JOINT_NAMES, BSRL_AMP_KEY_BODY_NAMES


def baseline():
    p=ROOT/'logs/rsl_rl/bsrl_amp/2026-09-16_10-51-46_reduced_parallelankle_headingframevel_criticpriv_mirror01_noactorheading_formal1_8192/params'
    if not p.is_dir(): pytest.skip('Local saved baseline required')
    return load_saved_config(p/'env.yaml'),load_saved_config(p/'agent.yaml')


def test_pair_diff_is_only_mirror_and_run_name():
    e,a=baseline();validate_baseline(e,a)
    off,ao=expected_configuration(e,a,'off','/data',{'clip':1.},'off')
    on,an=expected_configuration(e,a,'on','/data',{'clip':1.},'on')
    assert configuration_differences(off,on)==[]
    assert {d['path'] for d in configuration_differences(ao,an)}=={'run_name','algorithm.symmetry_cfg'}
    assert off['observations']==e['observations'] and off['rewards']==e['rewards']
    assert off['scene']==e['scene'] and off['sim']==e['sim']
    assert ao['algorithm']['symmetry_cfg'] is None
    assert an['algorithm']['symmetry_cfg']['mirror_loss_coeff']==.1


def test_unrequested_change_rejected():
    e,a=baseline();expected,agent=expected_configuration(e,a,'new','/data',{},'off')
    changed=copy.deepcopy(expected);changed['sim']['dt']=.001
    with pytest.raises(ValueError,match='Unrequested'):
        check_configuration(expected,agent,changed,agent)


def test_full_export_inventory():
    p=ROOT/'source/legged_lab/legged_lab/data/MotionData/bsrl/amp/accad_g1full_34_20260916/manifest.json'
    if not p.is_file(): pytest.skip('Local dataset required')
    m,w=load_dataset(p)
    assert len(w)==34 and set(w.values())=={1.}
    assert m['frames']==10896 and m['duration_s']==pytest.approx(181.6)


def test_export_reorders_without_motion_change():
    joints=list(reversed(BSRL_AMP_JOINT_NAMES))
    source=dict(joint_coordinate_convention='bsrl_xyz_positive_v1',
        grounding_convention='source_mesh_clearance_transfer_with_flight_v1',
        dof_names=joints,key_body_names=list(reversed(BSRL_AMP_KEY_BODY_NAMES)),
        dof_pos=np.zeros((3,12)),root_pos=np.array([[0,0,.8],[1,0,.9],[2,1,.8]]),
        root_rot=np.array([[0,0,0,1],[0,0,0,-1],[0,0,0,1]]),fps=60,
        key_body_pos=np.arange(18).reshape(3,2,3),source_metadata={},source_times=np.arange(3)/60)
    output=training_payload(source)
    np.testing.assert_allclose(output['root_pos'],source['root_pos'])
    np.testing.assert_allclose(output['root_rot'],[[1,0,0,0]]*3)
    np.testing.assert_allclose(output['key_body_pos'],source['key_body_pos'][:,::-1])
    assert output['loop_mode']==0 and not output['training_quality_validated']


def test_runtime_accepts_unflattened_amp_history():
    e,a=baseline()
    dims={'policy':(240,),'critic':(305,),'state_est_critic':(5,),
          'reward_privileged':(60,),'disc':(4,39),'disc_demo':(4,39)}
    env=SimpleNamespace(cfg=SimpleNamespace(to_dict=lambda:e),
        observation_manager=SimpleNamespace(group_obs_dim=dims),physics_dt=.005,step_dt=.02,
        scene={'robot':SimpleNamespace(num_joints=12,actuators={k:None for k in ('large_motors','small_motors','parallel_ankles')})})
    manifest=ROOT/'source/legged_lab/legged_lab/data/MotionData/bsrl/amp/accad_g1full_34_20260916/manifest.json'
    base=ROOT/'logs/rsl_rl/bsrl_amp/2026-09-16_10-51-46_reduced_parallelankle_headingframevel_criticpriv_mirror01_noactorheading_formal1_8192'
    agent=SimpleNamespace(run_name='new',obs_groups=a['obs_groups'],to_dict=lambda:a)
    with patch('bsrl_accad_ablation.check_configuration'):
        result=runtime_check(env,agent,base,manifest,'off')
        assert result['disc_dim']==156
        dims['disc']=(4,38)
        with pytest.raises(RuntimeError,match='Observation contract'):
            runtime_check(env,agent,base,manifest,'off')
