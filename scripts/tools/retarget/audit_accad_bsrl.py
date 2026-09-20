"""Audit ACCAD translation, official SMPL-X FK, BSRL export, and full video decode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import imageio.v2 as imageio
import numpy as np

from accad_smplx import (SMPL_TO_ROBOT_BASIS, load_motion, motion_fk, skeleton_fk,
                        source_mesh_foot_heights, flight_clearance_target)
from cmu_babel import align_initial_heading, save_json
from prepare_accad_bsrl import sha
from retarget_accad_bsrl import PREVIEW_FLOOR_CLEARANCE, STAND_IK_RESIDUAL_TOLERANCE


def official_fk_error(motion, model_path, rest, parents):
    import smplx
    import torch
    idx = np.unique(np.linspace(0, len(motion['trans'])-1, 5, dtype=int))
    n = len(idx)
    model = smplx.SMPLX(model_path=model_path, gender=motion['gender'],
                        num_betas=len(motion['betas']), use_pca=False, flat_hand_mean=True, ext='npz')
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        output = model(betas=tensor(motion['betas'])[None].expand(n, -1),
            global_orient=tensor(motion['root_orient'][idx]), body_pose=tensor(motion['pose_body'][idx]),
            transl=tensor(motion['trans'][idx]), expression=torch.zeros(n, 10),
            left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
            jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3), reye_pose=torch.zeros(n, 3))
    angles = np.concatenate([motion['root_orient'][:, None], motion['pose_body'].reshape(-1, 21, 3)], axis=1)
    reference, _ = skeleton_fk(rest, parents, angles[idx], motion['trans'][idx])
    error = float(np.abs(output.joints[:, :22].numpy()-reference).max())
    if error > 2e-5:
        raise ValueError(f'Official SMPL-X FK mismatch: {error}')
    return dict(official_max_joint_error_m=error, official_sample_indices=idx.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--video-dir', type=Path, help='If supplied, require and fully decode every video')
    parser.add_argument('--gmr-dir', type=Path, default=Path('/root/mjw/GMR'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.gmr_dir))
    import mujoco
    from general_motion_retargeting.grounding import RobotGrounder
    from retarget_cmu_babel_bsrl import foot_clearances
    audit = json.loads((args.run/'audit.json').read_text())
    manifest = json.loads(Path(audit['source_manifest']).read_text())
    for record in [*manifest['source_files'].values(), *manifest['body_models'].values()]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Changed source or body model')
    grounder = RobotGrounder('bsrl')
    model, data = grounder.model, grounder.data
    foot_ids = [[g for g in grounder.geom_ids if model.body(model.geom_bodyid[g]).name == f'link_{s}_ankle_roll']
                for s in ('left', 'right')]
    outputs = []
    for entry in audit['clips']:
        clip = entry['clip']; stem, category = clip['stem'], clip['category']
        raw = load_motion(clip['source_path'])
        model_path = manifest['body_models'][raw['gender']]['path']
        p, r, rest, parents, idx = motion_fk(raw, model_path, entry['fps'])
        official = official_fk_error(raw, model_path, rest, parents)
        trans_error = float(np.abs((p[:, 0]-p[0, 0])-(raw['trans'][idx]-raw['trans'][0])).max())
        if trans_error > 1e-9:
            raise ValueError('Source translation mismatch')
        expected, _, _ = align_initial_heading(p, r @ SMPL_TO_ROBOT_BASIS)
        expected *= entry['scale']; expected[:, :, 2] -= entry['source_display_floor_m']
        with np.load(args.run/'source_fk'/category/(stem+'.npz'), allow_pickle=False) as source:
            np.testing.assert_allclose(source['positions'], expected, atol=1e-9, rtol=0)
            np.testing.assert_array_equal(source['source_indices'], idx)
            saved_clearances = source['sole_clearance']
            saved_target = source['target_min_clearance'] if 'target_min_clearance' in source else None
            saved_mesh_heights = source['source_mesh_foot_min_z'] if 'source_mesh_foot_min_z' in source else None
        path = args.run/'gmr'/category/(stem+'.pkl')
        with path.open('rb') as stream:
            motion = pickle.load(stream)
        count = len(motion['root_pos'])
        if count != len(idx) or not motion['kinematic_preview_only']:
            raise ValueError('Frame count or preview flag mismatch')
        np.testing.assert_allclose(motion['source_times'], idx/raw['fps'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(motion['root_rot'], axis=1), 1, atol=1e-9, rtol=0)
        qpos = np.tile(model.qpos0, (count, 1))
        qpos[:, :3], qpos[:, 3:7] = motion['root_pos'], motion['root_rot'][:, [3, 0, 1, 2]]
        for col, name in enumerate(motion['dof_names']):
            joint = model.joint(name)
            values = motion['dof_pos'][:, col]
            if values.min() < joint.range[0]-1e-5 or values.max() > joint.range[1]+1e-5:
                raise ValueError('Joint limit violation')
            qpos[:, joint.qposadr[0]] = values
        if not np.isfinite(qpos).all():
            raise ValueError('Nonfinite exported pose')
        errors, clearance = [], []
        for i, pose in enumerate(qpos):
            clearance.append(foot_clearances(grounder, pose, foot_ids))
            points = np.array([data.body(n).xpos.copy() for n in motion['key_body_names']])
            errors.append(float(np.abs(points-motion['key_body_pos'][i]).max()))
        if max(errors) > 1e-5:
            raise ValueError('Viewer/export FK mismatch')
        clearance = np.array(clearance)
        np.testing.assert_allclose(clearance, saved_clearances, atol=1e-9, rtol=0)
        expected_target = np.full(count, PREVIEW_FLOOR_CLEARANCE)
        if category in ('run', 'run_transition'):
            mesh_heights = source_mesh_foot_heights(raw, model_path, idx)
            np.testing.assert_allclose(mesh_heights, saved_mesh_heights, atol=1e-7, rtol=0)
            expected_target, _ = flight_clearance_target(mesh_heights, entry['scale'])
        if saved_target is not None:
            np.testing.assert_allclose(saved_target, expected_target, atol=1e-9, rtol=0)
        np.testing.assert_allclose(clearance.min(axis=1), expected_target, atol=1e-8, rtol=0)
        # Each pre-projection foot can be +/- the IK residual tolerance. Moving
        # the lower foot to the floor gives the upper foot a 2*tolerance bound.
        stand_bound = PREVIEW_FLOOR_CLEARANCE + 2*STAND_IK_RESIDUAL_TOLERANCE
        if category == 'stand' and clearance.max() > stand_bound+1e-8:
            raise ValueError('Standing sole clearance exceeds the bilateral IK tolerance bound')
        delta = p[:, 0, :2]-p[0, 0, :2]
        net = float(np.linalg.norm(delta[-1]))
        cross = np.abs(delta[:, 0]*delta[-1, 1]-delta[:, 1]*delta[-1, 0])/max(net, 1e-12)
        robot_net = float(np.linalg.norm(qpos[-1, :2]-qpos[0, :2]))
        expected_xy_delta = expected[:,0,:2]-expected[0,0,:2]
        actual_xy_delta = qpos[:,:2]-qpos[0,:2]
        displacement_error = float(np.linalg.norm(actual_xy_delta-expected_xy_delta,axis=1).max())
        np.testing.assert_allclose(displacement_error, entry['root_xy_displacement_error_max_m'], atol=1e-9, rtol=0)
        # A centimetre-level IK/root-reference difference is a quality warning,
        # not evidence of discarded trans. Still fail on substantial travel loss.
        source_travel = float(np.linalg.norm(np.diff(expected_xy_delta,axis=0),axis=1).sum())
        robot_travel = float(np.linalg.norm(np.diff(actual_xy_delta,axis=0),axis=1).sum())
        if source_travel > .4 and robot_travel < .8*source_travel:
            raise ValueError('Substantial loss of horizontal path travel')
        review_flags = []
        if displacement_error > .04:
            review_flags.append('root_reference_or_IK_displacement_error_over_4cm')
        if entry['joint_speed_max_rad_s'] > 20:
            review_flags.append('joint_speed_over_20rad_s_review_not_motor_limit')
        if category in ('run','run_transition'):
            review_flags.append('source_mesh_floor_and_flight_are_geometric_estimates')
        if entry['geometry_pass'] is not True:
            review_flags.append('category_purity_failed_or_unassessed')
        video_check = {}
        if args.video_dir:
            video = args.video_dir/category/(stem+'.mp4')
            metadata = json.loads(video.with_suffix('.json').read_text())
            reader = imageio.get_reader(str(video))
            decoded = 0
            try:
                info = reader.get_meta_data()
                for frame in reader:
                    if frame.shape != (720, 1440, 3):
                        raise ValueError('Invalid video resolution')
                    decoded += 1
            finally:
                reader.close()
            if decoded != metadata['rendered_frames'] or abs(info['fps']-metadata['video_fps']) > .01:
                raise ValueError('Incomplete or incorrectly timed video')
            video_check = dict(video_frames=decoded, video_duration_s=decoded/info['fps'],
                               video=str(video), video_sha256=sha(video))
        outputs.append(dict(stem=stem, category=category, **official,
            source_translation_max_error_m=trans_error, source_net_xy_m=net,
            source_path_xy_length_m=float(np.linalg.norm(np.diff(delta, axis=0), axis=1).sum()),
            source_max_cross_track_to_endpoint_line_m=float(cross.max()),
            robot_net_xy_m=robot_net, max_viewer_export_fk_error_m=max(errors),
            root_xy_displacement_error_max_m=displacement_error,
            quality_review_flags=review_flags,
            left_right_sole_clearance_max_m=clearance.max(axis=0).tolist(),
            standing_bilateral_clearance_bound_m=stand_bound if category == 'stand' else None,
            source_frames=count, source_duration_s=count/motion['fps'], **video_check,
            source_geometric_air_frames=int(np.sum(expected_target > .01)),
            pkl_sha256=sha(path), category_screen_pass=entry['geometry_pass']))
        print('[VERIFIED]', stem, flush=True)
    if len(outputs) != len(manifest['clips']) or {o['stem'] for o in outputs} != {c['stem'] for c in manifest['clips']}:
        raise ValueError('Expected every selected source clip, without silent omissions')
    report_name = 'verification.json' if args.video_dir else 'verification_data.json'
    save_json(args.run/report_name, dict(success=True, success_scope='data_integrity_not_training_quality',
        videos_checked=bool(args.video_dir), checks=outputs,
        limitations=['Category names are source candidates, not curated pure segments.',
                     'Vertical clearance transfer and ankle-width adaptation are explicit morphology adjustments.',
                     'Running air clearance comes from source mesh geometry, not authoritative contact labels.',
                     'No stance-foot locking, contact-force, balance, motor-speed feasibility, or AMP readiness validation.',
                     'Raw MuJoCo mesh-distance zero readings are not proof of inter-foot contact or clearance.']))
    print(f'All {len(outputs)} exports verified; videos checked: {bool(args.video_dir)}.', flush=True)


if __name__ == '__main__':
    main()
