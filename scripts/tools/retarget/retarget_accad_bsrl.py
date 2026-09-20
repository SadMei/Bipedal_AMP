"""Translation-preserving ACCAD SMPL-X -> BSRL kinematic preview, no training edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from accad_smplx import (BODY_NAMES, SMPL_TO_ROBOT_BASIS, limb_reference_frame,
                        load_motion, motion_fk, source_mesh_foot_heights, flight_clearance_target)
from cmu_babel import align_initial_heading, category_passes, save_json, trajectory_metrics
from prepare_accad_bsrl import sha
from retarget_cmu_babel_bsrl import (enforce_ankle_width, foot_clearances,
                                    sole_metrics, sole_vertices)

PREVIEW_FLOOR_CLEARANCE = .0005
STAND_IK_RESIDUAL_TOLERANCE = .001


def screen_category(category, stats):
    if category in ('run', 'run_transition'):
        return None  # Walking pelvis-yaw gate is not a running-motion validator.
    if category == 'backward':
        return bool(stats['median_forward_m_s'] < -.1 and stats['lateral_fraction'] < .4
                    and stats['net_xy_m'] > .25)
    return bool(category_passes(category, stats))


def config(output):
    table = {'base_link': ['Pelvis', 100, 30, [0., 0., 0.], [1., 0., 0., 0.]]}
    for side in ('left', 'right'):
        for body, source, pw, rw in [('thigh', 'Hip', 0, 15), ('shank', 'Knee', 5, 20),
                                     ('ankle_roll', 'Ankle', 150, 40)]:
            table[f'link_{side}_{body}'] = [side.title()+source, pw, rw, [0., 0., 0.], [1., 0., 0., 0.]]
    value = dict(robot_root_name='base_link', human_root_name='Pelvis', ground_height=0.,
        human_height_assumption=1., use_ik_match_table1=True, use_ik_match_table2=False,
        human_scale_table={entry[0]: 1. for entry in table.values()}, ik_match_table1=table, ik_match_table2={})
    path = output/'accad_smplx_to_bsrl.json'
    save_json(path, value)
    return path


def build_targets(positions, rotations, rest, model):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    human_leg = np.mean([np.linalg.norm(rest[k]-rest[h])+np.linalg.norm(rest[a]-rest[k])
                         for h, k, a in [(1, 4, 7), (2, 5, 8)]])
    robot_leg = np.mean([sum(np.linalg.norm(data.body(f'link_{side}_{b}').xpos-data.body(f'link_{side}_{a}').xpos)
                             for a, b in [('thigh', 'shank'), ('shank', 'ankle_roll')])
                         for side in ('left', 'right')])
    scale = float(robot_leg/human_leg)
    points = positions*scale
    source_floor = float(np.quantile(points[:, [7, 8, 10, 11], 2].min(axis=1), .02))
    points[:, :, 2] -= source_floor
    pelvis_rot = rotations[:, 0] @ SMPL_TO_ROBOT_BASIS
    pelvis_offset = data.body('base_link').xpos-.5*(data.body('link_left_thigh').xpos+data.body('link_right_thigh').xpos)
    target_pos = {'Pelvis': .5*(points[:, 1]+points[:, 2]) + pelvis_rot @ pelvis_offset}
    target_rot = {'Pelvis': pelvis_rot}
    for side, h, k, a in [('Left', 1, 4, 7), ('Right', 2, 5, 8)]:
        for label, index, basis in [('Hip', h, limb_reference_frame(rest[k]-rest[h])),
                                    ('Knee', k, limb_reference_frame(rest[a]-rest[k])),
                                    ('Ankle', a, SMPL_TO_ROBOT_BASIS)]:
            target_pos[side+label] = points[:, index].copy()
            target_rot[side+label] = rotations[:, index] @ basis
    target_pos['LeftAnkle'], target_pos['RightAnkle'], width = enforce_ankle_width(
        target_pos['LeftAnkle'], target_pos['RightAnkle'], target_pos['Pelvis'], target_rot['Pelvis'])
    quats = {name: Rotation.from_matrix(r).as_quat(scalar_first=True) for name, r in target_rot.items()}
    frames = [{name: [p[i].copy(), quats[name][i].copy()] for name, p in target_pos.items()}
              for i in range(len(points))]
    return frames, points, scale, width, source_floor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gmr-dir', type=Path, default=Path('/root/mjw/GMR'))
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--include-unsegmented-preview', action='store_true',
                        help='Keep whole source clips even if the category purity screen fails; never AMP-ready')
    args = parser.parse_args()
    if not 1 <= args.fps <= 120:
        parser.error('FPS must be in [1,120]')
    sys.path.insert(0, str(args.gmr_dir))
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.grounding import RobotGrounder
    from general_motion_retargeting.params import IK_CONFIG_DICT, BSRL_KEY_BODY_NAMES, BSRL_JOINT_COORDINATE_CONVENTION
    from general_motion_retargeting.utils.mujoco_model import body_positions, local_body_positions, joint_names_in_qpos_order
    manifest_path = args.source/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for record in [*manifest['source_files'].values(), *manifest['body_models'].values()]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Source/model hash mismatch: '+record['path'])
    IK_CONFIG_DICT['accad_smplx'] = {'bsrl': config(args.output)}
    audits = []
    for clip in manifest['clips']:
        stem, category = clip['stem'], clip['category']
        if args.clips and stem not in args.clips:
            continue
        outpath = args.output/'gmr'/category/(stem+'.pkl')
        if outpath.exists():
            raise FileExistsError(outpath)
        motion = load_motion(clip['source_path'])
        model_path = manifest['body_models'][motion['gender']]['path']
        p, r, rest, parents, indices = motion_fk(motion, model_path, args.fps)
        times = indices/motion['fps']
        # Root heading is defined by SMPL semantic forward, not its local +X axis.
        semantic = r @ SMPL_TO_ROBOT_BASIS
        points, _, yaw_removed = align_initial_heading(p, semantic)
        alignment = Rotation.from_euler('z', -yaw_removed).as_matrix()
        rotations = alignment @ r
        source_stats = trajectory_metrics(points[:, 0], rotations[:, 0] @ SMPL_TO_ROBOT_BASIS, args.fps)
        length_error = float(np.abs(np.linalg.norm(p[:, 1:]-p[:, parents[1:]], axis=-1)-
                                   np.linalg.norm(rest[1:]-rest[parents[1:]], axis=-1)).max())
        if length_error > 1e-9:
            raise ValueError('Invalid source FK bone lengths')
        audit = dict(clip=clip, source_trajectory=source_stats,
            source_fk_bone_length_error_m=length_error,
            geometry_pass=screen_category(category, source_stats),
            category_screen='CMU preview heuristic: pelvis yaw and body-frame velocity, not an authoritative ACCAD label',
            unsegmented_full_source=True)
        if not audit['geometry_pass'] and not args.include_unsegmented_preview:
            audit['status'] = 'rejected_source_trajectory'
            save_json(args.output/'metrics'/category/(stem+'.json'), audit)
            audits.append(audit)
            print('[REJECTED]', stem, source_stats, flush=True)
            continue
        if not audit['geometry_pass']:
            print('[UNSEGMENTED PREVIEW: category screen failed]', stem, source_stats, flush=True)
        retargeter = GeneralMotionRetargeting('accad_smplx', 'bsrl', verbose=False)
        grounder = RobotGrounder('bsrl')
        if joint_names_in_qpos_order(retargeter.model) != joint_names_in_qpos_order(grounder.model):
            raise ValueError('Grounder/model joint order mismatch')
        frames, scaled, scale, width, source_floor = build_targets(points, rotations, rest, retargeter.model)
        foot_ids = [[gid for gid in grounder.geom_ids if grounder.model.body(grounder.model.geom_bodyid[gid]).name == f'link_{side}_ankle_roll']
                    for side in ('left', 'right')]
        if category == 'stand':
            vertices = sole_vertices(grounder, foot_ids)
            for frame in frames:
                shifts = []
                for side, verts in zip(('Left', 'Right'), vertices):
                    pos, quat = frame[side+'Ankle']
                    target_z = PREVIEW_FLOOR_CLEARANCE-Rotation.from_quat(quat, scalar_first=True).apply(verts)[:, 2].min()
                    shifts.append(target_z-pos[2])
                    pos[2] = target_z
                frame['Pelvis'][0][2] += np.mean(shifts)
        qpos = []
        for i, frame in enumerate(frames):
            if i == 0:
                for _ in range(25):
                    retargeter.retarget({n: [p.copy(), q.copy()] for n, (p, q) in frame.items()})
            pose = retargeter.retarget(frame)
            if category == 'stand':
                for _ in range(4):
                    residual = PREVIEW_FLOOR_CLEARANCE-foot_clearances(grounder, pose, foot_ids)
                    if np.abs(residual).max() < STAND_IK_RESIDUAL_TOLERANCE:
                        break
                    frame['Pelvis'][0][2] += .5*float(residual.mean())
                    for j, side in enumerate(('Left', 'Right')):
                        frame[side+'Ankle'][0][2] += residual[j]
                    pose = retargeter.retarget(frame)
            qpos.append(pose)
        qpos = np.array(qpos)
        if not np.isfinite(qpos).all():
            raise ValueError('Nonfinite IK')
        mesh_heights = None
        grounding_info = None
        target_clearance = np.full(len(qpos), PREVIEW_FLOOR_CLEARANCE)
        if category in ('run', 'run_transition'):
            mesh_heights = source_mesh_foot_heights(motion, model_path, indices)
            target_clearance, grounding_info = flight_clearance_target(mesh_heights, scale)
        shift = target_clearance-np.array([grounder.qpos_min_z(q) for q in qpos])
        qpos[:, 2] += shift  # Never modifies XY; running keeps source geometric air clearance.
        foot_stats, clearance = sole_metrics(grounder, qpos, args.fps)
        world_pos, names = body_positions(retargeter.model, qpos)
        local_pos, local_names = local_body_positions(retargeter.model, qpos)
        if names != local_names:
            raise ValueError('FK name mismatch')
        ids = np.flatnonzero(retargeter.model.jnt_type != 0)
        joints = qpos[:, retargeter.model.jnt_qposadr[ids]]
        limits = retargeter.model.jnt_range[ids]
        violation = float(max(0., np.max(limits[:, 0]-joints), np.max(joints-limits[:, 1])))
        if violation > 1e-5:
            raise ValueError('Joint-limit violation')
        robot_stats = trajectory_metrics(qpos[:, :3], Rotation.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix(), args.fps)
        root_xy_error = float(np.linalg.norm((qpos[:, :2]-qpos[0, :2])-
                                             (scaled[:, 0, :2]-scaled[0, 0, :2]), axis=-1).max())
        if category != 'stand' and robot_stats['net_xy_m'] < .5*source_stats['net_xy_m']*scale:
            raise ValueError('Unexpected loss of root displacement')
        audit.update(status='kinematic_preview', fps=args.fps, frames=len(qpos), duration_s=len(qpos)/args.fps,
            last_sample_s=float(times[-1]), scale=scale, removed_initial_yaw_rad=yaw_removed,
            source_display_floor_m=source_floor, robot_trajectory=robot_stats,
            root_xy_displacement_error_max_m=root_xy_error,
            joint_limit_violation_rad=violation, joint_speed_max_rad_s=float(np.abs(np.diff(joints, axis=0)*args.fps).max()),
            grounding=('source_mesh_clearance_transfer_with_flight_v1' if grounding_info else
                       'bilateral_standing_sole_targets_and_nonflight_lowest_sole_projection_0.5mm'),
            grounding_details=grounding_info,
            minimum_target_ankle_width_m=.22, per_foot_lateral_adaptation_p50_max_m=np.quantile(width, [.5, 1]).tolist(),
            root_z_projection_min_p50_max_m=np.quantile(shift, [0, .5, 1]).tolist(),
            warnings=['Kinematic preview only, not a trained policy or dynamic feasibility check.',
                      'Preserves horizontal travel and turning, after constant initial alignment and leg-length scaling.',
                      ('Running: source foot-mesh rise retained; geometric floor estimate, not force/contact labels.'
                       if grounding_info else 'Lowest sole projected onto floor; no locomotion stance labels.'),
                      'No horizontal foot locking.',
                      'Joint velocities/contact stability are not validated for AMP training.',
                      'Whole source sequence retained, including start/end reorientation; folder categories are source-name candidates.']
                     + ([] if audit['geometry_pass'] else ['Category purity screen failed or unavailable: inspect/segment before training.']), **foot_stats)
        payload = dict(fps=args.fps, root_pos=qpos[:, :3], root_rot=qpos[:, [4, 5, 6, 3]], dof_pos=qpos[:, 7:],
            dof_names=joint_names_in_qpos_order(retargeter.model), local_body_pos=local_pos, link_body_list=names,
            key_body_pos=world_pos[:, [names.index(n) for n in BSRL_KEY_BODY_NAMES]], key_body_names=list(BSRL_KEY_BODY_NAMES),
            source_metadata=clip, source_times=times, dataset_label='AMASS ACCAD / SMPL-X',
            source_panel_title='SMPL-X source (scaled)', joint_coordinate_convention=BSRL_JOINT_COORDINATE_CONVENTION,
            category_screen_pass=audit['geometry_pass'],
            feet_panel_title=('Feet: source-mesh flight clearance' if grounding_info else 'Feet: non-flight support projection'),
            grounding_convention=audit['grounding'], kinematic_preview_only=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open('xb') as stream:
            pickle.dump(payload, stream, protocol=4)
        source_path = args.output/'source_fk'/category/(stem+'.npz')
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open('xb') as stream:
            arrays = dict(positions=scaled, bones=np.array(BODY_NAMES), parents=parents,
                times=times, source_indices=indices, sole_clearance=clearance, target_min_clearance=target_clearance)
            if mesh_heights is not None:
                arrays['source_mesh_foot_min_z'] = mesh_heights
            np.savez_compressed(stream, **arrays)
        save_json(args.output/'metrics'/category/(stem+'.json'), audit)
        audits.append(audit)
        print(f'[DONE] {stem}: {len(qpos)} frames; source travel {source_stats["net_xy_m"]:.3f}m, '
              f'robot {robot_stats["net_xy_m"]:.3f}m, scale {scale:.3f}; yaw {robot_stats["net_yaw_deg"]:.1f}', flush=True)
    save_json(args.output/'audit.json', dict(clips=audits, source_manifest=str(manifest_path)))


if __name__ == '__main__':
    main()
