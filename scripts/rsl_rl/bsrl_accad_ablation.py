"""Frozen analytical-parallel BSRL + full ACCAD, differing only by mirror loss."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

from frozen_run_config import restore_run_config
from t1_heading_ablation import configuration_differences, load_saved_config, resolved_scene_config

ACCAD_GROUNDING = 'bsrl_accad_source_clearance_v1'
HEADING_MODULE = 'legged_lab.tasks.locomotion.amp.mdp.heading_frame'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_dataset(manifest_path):
    import joblib
    import numpy as np
    retarget = Path(__file__).resolve().parents[1]/'tools/retarget'
    sys.path.insert(0,str(retarget))
    from bsrl_conventions import (BSRL_AMP_JOINT_NAMES, BSRL_AMP_KEY_BODY_NAMES,
                                  BSRL_JOINT_COORDINATE_CONVENTION, validate_bsrl_joint_positions)
    path=Path(manifest_path).resolve(strict=True)
    manifest=json.loads(path.read_text())
    if manifest['clip_count']!=34 or manifest['reference_names']!=30 or manifest['grounding_convention']!=ACCAD_GROUNDING:
        raise ValueError('Expected full ACCAD 34-clip training-format export')
    if len(manifest['files'])!=34 or len({f['stem'] for f in manifest['files']})!=34:
        raise ValueError('Invalid dataset inventory')
    for record in manifest['files']:
        p=Path(record['path']).resolve(strict=True)
        if p.parent!=path.parent or p.name!=record['stem']+'.pkl' or file_hash(p)!=record['sha256']:
            raise ValueError('Motion path/hash mismatch')
        motion=joblib.load(p)
        if (motion.get('grounding_convention')!=ACCAD_GROUNDING or
                motion.get('joint_coordinate_convention')!=BSRL_JOINT_COORDINATE_CONVENTION or
                tuple(motion['joint_names'])!=BSRL_AMP_JOINT_NAMES or
                tuple(motion['key_body_names'])!=BSRL_AMP_KEY_BODY_NAMES or motion['loop_mode']!=0):
            raise ValueError('Motion convention mismatch')
        n=record['frames']
        for name,shape in [('root_pos',(n,3)),('root_rot',(n,4)),('dof_pos',(n,12)),('key_body_pos',(n,2,3))]:
            x=np.asarray(motion[name])
            if x.shape!=shape or not np.isfinite(x).all():
                raise ValueError('Invalid motion array '+name)
        if motion['fps']!=60 or not np.allclose(np.linalg.norm(motion['root_rot'],axis=1),1,atol=1e-5):
            raise ValueError('Wrong fps/quaternion')
        validate_bsrl_joint_positions(motion['dof_pos'],motion['joint_names'],str(p))
    return manifest, {r['stem']:float(r['weight']) for r in manifest['files']}


def validate_baseline(env,agent):
    robot=env['scene']['robot']
    if set(robot['actuators'])!={'large_motors','small_motors','parallel_ankles'}:
        raise ValueError('Expected analytical parallel ankles, not direct serial or explicit closed loops')
    if not robot['actuators']['parallel_ankles']['class_type'].endswith(':ReducedAnkleActuator'):
        raise ValueError('Wrong parallel actuator')
    if agent['policy']['class_name']!='ActorCritic' or agent['algorithm'].get('state_est_loss_required'):
        raise ValueError('Estimator must remain disabled')
    if env['observations'].get('state_est_label') is not None:
        raise ValueError('Estimator labels must remain disabled')
    if env['observations']['policy'].get('heading_error') or env['observations']['policy'].get('heading_frame_error'):
        raise ValueError('Actor heading must remain absent')
    if env['commands']['base_velocity']['class_type']!=HEADING_MODULE+':HeadingReferenceVelocityCommand':
        raise ValueError('Expected latest heading-reference reduced baseline')
    if env['rewards']['heading_frame_lin_vel_xy_exp']['weight']!=1.:
        raise ValueError('Expected existing heading-frame reward')
    symmetry=agent['algorithm']['symmetry_cfg']
    if not symmetry['use_mirror_loss'] or symmetry['use_data_augmentation'] or symmetry['mirror_loss_coeff']!=.1:
        raise ValueError('Expected baseline mirror-only loss 0.1')
    if agent['resume'] or agent['policy']['actor_obs_normalization'] or agent['policy']['critic_obs_normalization']:
        raise ValueError('Fresh unnormalized ActorCritic required')


def expected_configuration(saved_env,saved_agent,run_name,dataset_dir,weights,mirror):
    if mirror not in ('off','on'):
        raise ValueError('mirror must be off/on')
    env,agent=copy.deepcopy(saved_env),copy.deepcopy(saved_agent)
    dataset=env['motion_data']['motion_dataset']
    dataset['motion_data_dir']=str(dataset_dir)
    dataset['motion_data_weights']=dict(weights)
    agent['run_name']=run_name
    if mirror=='off':
        agent['algorithm']['symmetry_cfg']=None
    return env,agent


def check_configuration(expected_env,expected_agent,actual_env,actual_agent):
    expected_env=copy.deepcopy(expected_env)
    expected_env['log_dir']=actual_env.get('log_dir')
    diff=configuration_differences(expected_env,resolved_scene_config(actual_env),'env')
    diff+=configuration_differences(expected_agent,actual_agent,'agent')
    if diff:
        raise ValueError(f'Unrequested ACCAD ablation changes: {diff}')


def apply_accad_ablation(env_cfg,agent_cfg,baseline_run,manifest_path,mirror):
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg
    from legged_lab.assets.bsrl import BSRL_CFG
    from legged_lab.assets.bsrl_reduced_ankle import with_reduced_ankles
    from legged_lab.tasks.locomotion.amp.mdp.heading_frame import (
        HeadingReferenceVelocityCommand,heading_frame_error,heading_frame_lin_vel_xy,heading_frame_lin_vel_xy_exp)
    baseline=Path(baseline_run).resolve(strict=True)
    saved_env=load_saved_config(baseline/'params/env.yaml')
    saved_agent=load_saved_config(baseline/'params/agent.yaml')
    validate_baseline(saved_env,saved_agent)
    manifest_path=Path(manifest_path).resolve(strict=True)
    manifest,weights=load_dataset(manifest_path)
    run_name=agent_cfg.run_name
    if not run_name or run_name==saved_agent['run_name'] or agent_cfg.resume:
        raise ValueError('Distinct fresh run required')
    for name,value,expected in [('num_envs',env_cfg.scene.num_envs,saved_env['scene']['num_envs']),
                                ('seed',agent_cfg.seed,saved_agent['seed']),
                                ('max_iterations',agent_cfg.max_iterations,saved_agent['max_iterations'])]:
        if value!=expected:
            raise ValueError(f'Keep baseline {name}: {value} != {expected}')
    # Install the typed templates for fields absent in current task defaults.
    env_cfg.scene.robot=with_reduced_ankles(BSRL_CFG,physics_dt=saved_env['sim']['dt'])
    env_cfg.commands.base_velocity.class_type=HeadingReferenceVelocityCommand
    env_cfg.rewards.heading_frame_lin_vel_xy_exp=RewardTermCfg(
        func=heading_frame_lin_vel_xy_exp,weight=1.,params={'command_name':'base_velocity','std':.4})
    env_cfg.observations.critic.heading_frame_error=ObservationTermCfg(
        func=heading_frame_error,params={'command_name':'base_velocity'})
    env_cfg.observations.critic.heading_frame_lin_vel_xy=ObservationTermCfg(
        func=heading_frame_lin_vel_xy,params={'command_name':'base_velocity'})
    audit=restore_run_config(env_cfg,agent_cfg,baseline)
    env_cfg.motion_data.motion_dataset.motion_data_dir=str(manifest_path.parent)
    env_cfg.motion_data.motion_dataset.motion_data_weights=weights
    agent_cfg.run_name=run_name
    if mirror=='off':
        agent_cfg.algorithm.symmetry_cfg=None
    expected_env,expected_agent=expected_configuration(saved_env,saved_agent,run_name,manifest_path.parent,weights,mirror)
    check_configuration(expected_env,expected_agent,env_cfg.to_dict(),agent_cfg.to_dict())
    audit.update(fresh_training=True,mirror=mirror,mirror_loss_coeff=.1 if mirror=='on' else 0.,
        estimator=False,actor_heading=False,heading_reward_and_critic_preserved=True,
        topology='serial_skeleton_analytical_parallel_ankles_no_explicit_loops',
        dataset_manifest=str(manifest_path),dataset_manifest_sha256=file_hash(manifest_path),
        dataset_clips=len(weights),dataset_frames=manifest['frames'],dataset_duration_s=manifest['duration_s'],
        sampling=manifest['sampling'],known_data_quality_warnings=manifest['warnings'],
        only_training_changes=['reference motion dataset: full ACCAD 34 clips',
                               'mirror loss disabled' if mirror=='off' else 'mirror 0.1 unchanged'])
    return audit


def runtime_check(env,agent_cfg,baseline_run,manifest_path,mirror):
    baseline=Path(baseline_run)
    manifest_path=Path(manifest_path).resolve(strict=True)
    _,weights=load_dataset(manifest_path)
    ee,ea=expected_configuration(load_saved_config(baseline/'params/env.yaml'),
        load_saved_config(baseline/'params/agent.yaml'),agent_cfg.run_name,manifest_path.parent,weights,mirror)
    check_configuration(ee,ea,env.cfg.to_dict(),agent_cfg.to_dict())
    robot=env.scene['robot'];dims=env.observation_manager.group_obs_dim
    total_critic=sum(dims[name][0] for name in agent_cfg.obs_groups['critic'])
    if robot.num_joints!=12 or set(robot.actuators)!={'large_motors','small_motors','parallel_ankles'}:
        raise RuntimeError('Wrong simulated plant')
    if (tuple(dims['policy'])!=(240,) or total_critic!=370
            or tuple(dims['disc'])!=(4,39) or tuple(dims['disc_demo'])!=(4,39)):
        raise RuntimeError(f'Observation contract changed: {dims}')
    return dict(num_joints=12,actuators=list(robot.actuators),actor_dim=240,critic_dim=370,disc_dim=156,
        physics_dt=env.physics_dt,policy_dt=env.step_dt,estimator=False,mirror=mirror,
        dataset_manifest_sha256=file_hash(manifest_path),dataset_clips=len(weights),
        unrequested_configuration_differences=[])
