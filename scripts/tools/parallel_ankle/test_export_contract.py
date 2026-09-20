"""CPU-only frozen policy and explicit-ankle deployment configuration checks."""
from pathlib import Path
import copy

import numpy as np
import pytest

from ._paths import REPO_ROOT
from .export_checkpoint import critic_widths
from .replay_policy import load_nominal, JOINTS
from scripts.rsl_rl.t1_heading_ablation import load_saved_config


def frozen(prefix):
    paths=list((REPO_ROOT/'logs/rsl_rl/bsrl_amp').glob(prefix+'*'))
    paths=[p for p in paths if (p/'params/env.yaml').is_file()]
    if len(paths)!=1: pytest.skip('Local frozen run unavailable')
    return paths[0],load_saved_config(paths[0]/'params/env.yaml')


def test_old_critic_350_and_new_privileged_370():
    _,old=frozen('2026-09-15_15-14-55_')
    _,new=frozen('2026-09-16_10-51-46_')
    assert critic_widths(old)==dict(policy=240,critic=285,state_est_critic=5,reward_privileged=60)
    assert critic_widths(new)==dict(policy=240,critic=305,state_est_critic=5,reward_privileged=60)
    for corrupt in ('unexpected_term','heading_missing','wrong_history','wrong_func'):
        bad=copy.deepcopy(new);critic=bad['observations']['critic']
        if corrupt=='unexpected_term':critic['extra']={'func':'unknown'}
        if corrupt=='heading_missing':del critic['heading_frame_error']
        if corrupt=='wrong_history':critic['history_length']=4
        if corrupt=='wrong_func':critic['heading_frame_error']['func']='unknown'
        with pytest.raises(ValueError):critic_widths(bad)


def test_explicit_ankle_uses_motor_limits_not_passive_zero_or_guard():
    run,_=frozen('2026-09-15_16-30-14_')
    nominal,_=load_nominal(run)
    ankle=[i for i,n in enumerate(JOINTS) if 'ankle' in n]
    np.testing.assert_allclose(nominal['effort_limit'][ankle],112.)
    np.testing.assert_allclose(nominal['velocity_limit'][ankle],26.17993877991494)
    np.testing.assert_allclose(nominal['peak_power'][ankle],1250.)
    np.testing.assert_allclose(nominal['stiffness'][ankle],100.)
    np.testing.assert_allclose(nominal['damping'][ankle],1.)
    np.testing.assert_allclose(nominal['armature'][ankle],.01)
    assert nominal['policy_dt']==.02 and nominal['training_physics_dt']==.001
