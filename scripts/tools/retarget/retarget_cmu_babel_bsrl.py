"""Retarget audited CMU/BABEL candidates to BSRL for isolated kinematic previews.

This four-class preview assumes non-flight locomotion. Per-frame sole projection
keeps at least one robot foot on the floor; it is NOT contact-force simulation,
does not pin horizontal stance position, and must NOT be applied to running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.ndimage import binary_closing
from scipy.spatial.transform import Rotation

from cmu_babel import (AXES, align_initial_heading, category_passes, forward_kinematics,
                       parse_amc, parse_asf, sample_interval, save_json, trajectory_metrics)


def leg_reference_frame(direction):
    z = -(AXES @ np.asarray(direction))
    z /= np.linalg.norm(z)
    x = np.array([1., 0., 0.])
    x -= np.dot(x, z)*z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def make_config(output):
    table = {'base_link': ['Pelvis', 100, 30, [0., 0., 0.], [1., 0., 0., 0.]]}
    for side in ('left', 'right'):
        for body, source, pw, rw in [('thigh', 'Hip', 0, 15), ('shank', 'Knee', 5, 20),
                                     ('ankle_roll', 'Ankle', 150, 40)]:
            table[f'link_{side}_{body}'] = [side.title()+source, pw, rw, [0., 0., 0.], [1., 0., 0., 0.]]
    config = dict(robot_root_name='base_link', human_root_name='Pelvis', ground_height=0.,
        human_height_assumption=1., use_ik_match_table1=True, use_ik_match_table2=False,
        human_scale_table={entry[0]: 1. for entry in table.values()},
        ik_match_table1=table, ik_match_table2={})
    path = output / 'cmu_asf_to_bsrl.json'
    save_json(path, config)
    return path


MINIMUM_ANKLE_WIDTH = .22


def enforce_ankle_width(left, right, pelvis, rotation, minimum=MINIMUM_ANKLE_WIDTH):
    """BSRL feet are wider than mocap feet. Symmetrically separate close targets.

    This is an explicit robot morphology adaptation, not a source-data edit.
    The returned displacement is recorded so distortion is inspectable.
    """
    local_l = np.einsum('nji,nj->ni', rotation, left-pelvis)
    local_r = np.einsum('nji,nj->ni', rotation, right-pelvis)
    correction = np.maximum(0., minimum-(local_l[:, 1]-local_r[:, 1]))*.5
    local_l[:, 1] += correction
    local_r[:, 1] -= correction
    return pelvis+np.einsum('nij,nj->ni', rotation, local_l), pelvis+np.einsum('nij,nj->ni', rotation, local_r), correction


def targets_for_robot(skeleton, positions, rotations, model):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = skeleton['names']
    human_leg = np.mean([sum(skeleton['bones'][prefix+b]['length'] for b in ('femur', 'tibia'))
                         for prefix in ('l', 'r')])
    robot_leg = np.mean([sum(np.linalg.norm(data.body(f'link_{side}_{b}').xpos-data.body(f'link_{side}_{a}').xpos)
                             for a, b in [('thigh', 'shank'), ('shank', 'ankle_roll')])
                         for side in ('left', 'right')])
    scale = robot_leg / human_leg
    points = positions * scale
    floor_ids = [names.index(n) for n in ('lfoot', 'ltoes', 'rfoot', 'rtoes')]
    source_floor = float(np.quantile(points[:, floor_ids, 2].min(axis=1), .02))
    points[:, :, 2] -= source_floor
    pelvis_offset = data.body('base_link').xpos - .5*(data.body('link_left_thigh').xpos + data.body('link_right_thigh').xpos)
    hip_mid = .5*(points[:, names.index('lhipjoint')] + points[:, names.index('rhipjoint')])
    target_pos = {'Pelvis': hip_mid + (rotations[:, 0] @ pelvis_offset)}
    target_rot = {'Pelvis': rotations[:, 0]}
    for side, prefix in [('Left', 'l'), ('Right', 'r')]:
        for target, point_bone, rot_bone in [('Hip', 'hipjoint', 'femur'),
                                            ('Knee', 'femur', 'tibia'), ('Ankle', 'tibia', 'foot')]:
            target_pos[side+target] = points[:, names.index(prefix+point_bone)]
            frame = np.eye(3) if target == 'Ankle' else leg_reference_frame(skeleton['bones'][prefix+rot_bone]['direction'])
            target_rot[side+target] = rotations[:, names.index(prefix+rot_bone)] @ frame
    target_pos['LeftAnkle'], target_pos['RightAnkle'], width_correction = enforce_ankle_width(
        target_pos['LeftAnkle'], target_pos['RightAnkle'], target_pos['Pelvis'], target_rot['Pelvis'])
    quaternions = {name: Rotation.from_matrix(r).as_quat(scalar_first=True) for name, r in target_rot.items()}
    frames = [{name: [p[i].copy(), quaternions[name][i].copy()] for name, p in target_pos.items()}
              for i in range(len(points))]
    return frames, points, scale, width_correction, source_floor


def source_support(points, names, fps, category):
    """Conservative geometric support estimate, NOT force-plate labels."""
    if category == 'stand':
        return np.ones((len(points), 2), bool), 0
    heights, mask = [], []
    for prefix in ('l', 'r'):
        foot = points[:, [names.index(prefix+'foot'), names.index(prefix+'toes')]]
        height = foot[:, :, 2].min(axis=1)
        height -= np.quantile(height, .05)
        speed = np.linalg.norm(np.gradient(foot[:, :, :2], 1/fps, axis=0), axis=-1).min(axis=1)
        contact = (height < .03) & (speed < .35)
        contact = binary_closing(contact, structure=np.ones(max(1, round(.08*fps))), border_value=1)
        heights.append(height); mask.append(contact)
    mask = np.column_stack(mask)
    empty = np.flatnonzero(~mask.any(axis=1))
    # Four classes are deliberately non-flight. Where the marker heuristic is
    # uncertain, choose the lower foot and expose how often this was necessary.
    if len(empty):
        lower = np.argmin(np.column_stack(heights)[empty], axis=1)
        mask[empty, lower] = True
    return mask, len(empty)


def foot_clearances(grounder, qpos, ids):
    import mujoco
    grounder.data.qpos[:] = qpos
    mujoco.mj_forward(grounder.model, grounder.data)
    return np.array([min(grounder._geom_min_z(gid) for gid in group) for group in ids])


def sole_vertices(grounder, ids):
    """Foot collision mesh vertices expressed in the ankle body frame."""
    import mujoco
    grounder.data.qpos[:] = grounder.model.qpos0
    mujoco.mj_forward(grounder.model, grounder.data)
    output = []
    for group in ids:
        vertices = []
        for gid in group:
            if grounder.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = grounder.model.geom_dataid[gid]
            adr, count = grounder.model.mesh_vertadr[mid], grounder.model.mesh_vertnum[mid]
            local = grounder.model.mesh_vert[adr:adr+count]*grounder.model.mesh_scale[mid]
            world = local @ grounder.data.geom_xmat[gid].reshape(3, 3).T + grounder.data.geom_xpos[gid]
            bid = grounder.model.geom_bodyid[gid]
            vertices.append((world-grounder.data.xpos[bid]) @ grounder.data.xmat[bid].reshape(3, 3))
        if not vertices:
            raise ValueError('Missing sole collision mesh')
        output.append(np.concatenate(vertices))
    return output


def sole_metrics(grounder, qpos, fps):
    import mujoco
    left, right, ankle, foot_distances = [], [], [], []
    by_side = {side: [gid for gid in grounder.geom_ids if grounder.model.body(grounder.model.geom_bodyid[gid]).name == f'link_{side}_ankle_roll']
               for side in ('left', 'right')}
    for q in qpos:
        grounder.data.qpos[:] = q
        mujoco.mj_forward(grounder.model, grounder.data)
        for side, rows in [('left', left), ('right', right)]:
            rows.append(min(grounder._geom_min_z(gid) for gid in by_side[side]))
        ankle.append([grounder.data.body(f'link_{side}_ankle_roll').xpos.copy() for side in ('left', 'right')])
        foot_distances.append(min(mujoco.mj_geomDistance(grounder.model, grounder.data, a, b, 1., None)
                                  for a in by_side['left'] for b in by_side['right']))
    clearance = np.column_stack([left, right])
    ankle = np.array(ankle)
    slip = np.linalg.norm(np.diff(ankle[:, :, :2], axis=0), axis=-1)*fps
    support = (clearance[:-1] < .005) & (clearance[1:] < .005)
    samples = slip[support]
    return dict(sole_min_p50_p95_max_m=np.quantile(clearance.min(axis=1), [0, .5, .95, 1]).tolist(),
                left_support_fraction=float(np.mean(clearance[:, 0] < .005)),
                right_support_fraction=float(np.mean(clearance[:, 1] < .005)),
                inter_foot_collision_distance_min_m=float(min(foot_distances)),
                support_ankle_xy_speed_p50_p95_m_s=(np.quantile(samples, [.5, .95]).tolist() if len(samples) else []),
                support_measurement='Geometric sole height <5mm; ankle XY speed includes foot rocking, not force-based contact/slip.'), clearance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest-name', default='manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gmr-dir', type=Path, default=Path('/root/mjw/GMR'))
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--clips', nargs='+')
    args = parser.parse_args()
    if not 1 <= args.fps <= 120:
        parser.error('--fps must be in [1,120]')
    sys.path.insert(0, str(args.gmr_dir))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.grounding import RobotGrounder
    from general_motion_retargeting.params import IK_CONFIG_DICT, BSRL_KEY_BODY_NAMES, BSRL_JOINT_COORDINATE_CONVENTION
    from general_motion_retargeting.utils.mujoco_model import body_positions, local_body_positions, joint_names_in_qpos_order
    if Path(args.manifest_name).name != args.manifest_name:
        parser.error('--manifest-name must be a simple filename')
    manifest = json.loads((args.source / args.manifest_name).read_text())
    # Detect accidentally replaced downloads before using the registration.
    for name, record in manifest['files'].items():
        if name.endswith(('.asf', '.amc')) and hashlib.sha256((args.source / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
    IK_CONFIG_DICT['cmu_asf'] = {'bsrl': make_config(args.output)}
    all_audits, cached = [], {}
    for clip in manifest['clips']:
        stem, category = clip['stem'], clip['category']
        if args.clips and stem not in args.clips:
            continue
        outpath = args.output / 'gmr' / category / (stem+'.pkl')
        if outpath.exists():
            raise FileExistsError(outpath)
        if clip['motion'] not in cached:
            skeleton = parse_asf((args.source / (clip['subject']+'.asf')).read_text())
            raw = parse_amc((args.source / (clip['motion']+'.amc')).read_text(), skeleton)
            p, r = forward_kinematics(skeleton, raw)
            cached[clip['motion']] = skeleton, p, r
        skeleton, full_p, full_r = cached[clip['motion']]
        points, matrices, times = sample_interval(full_p, full_r, clip['source_fps'], clip['start_t'], clip['end_t'], args.fps)
        points, matrices, yaw_removed = align_initial_heading(points, matrices)
        source_stats = trajectory_metrics(points[:, 0], matrices[:, 0], args.fps)
        audit = dict(clip=clip, source_trajectory=source_stats, geometry_pass=bool(category_passes(category, source_stats)))
        if not audit['geometry_pass']:
            audit['status'] = 'rejected_source_trajectory'
            save_json(args.output / 'metrics' / category / (stem+'.json'), audit)
            all_audits.append(audit)
            print(f'[REJECTED] {stem}: {source_stats}', flush=True)
            continue
        retargeter = GeneralMotionRetargeting('cmu_asf', 'bsrl', verbose=False)
        grounder = RobotGrounder('bsrl')
        if joint_names_in_qpos_order(retargeter.model) != joint_names_in_qpos_order(grounder.model):
            raise ValueError('Grounder/model joint order mismatch')
        frames, scaled_points, scale, width_correction, source_floor = targets_for_robot(skeleton, points, matrices, retargeter.model)
        support, uncertain_support = source_support(scaled_points, skeleton['names'], args.fps, category)
        foot_ids = [[gid for gid in grounder.geom_ids if grounder.model.body(grounder.model.geom_bodyid[gid]).name == f'link_{side}_ankle_roll']
                    for side in ('left', 'right')]
        if category == 'stand':
            vertices = sole_vertices(grounder, foot_ids)
            for frame in frames:
                shifts = []
                for side, verts in zip(('Left', 'Right'), vertices):
                    pos, quat = frame[side+'Ankle']
                    z_target = .0005-Rotation.from_quat(quat, scalar_first=True).apply(verts)[:, 2].min()
                    shifts.append(z_target-pos[2])
                    pos[2] = z_target
                frame['Pelvis'][0][2] += np.mean(shifts)
        qpos = []
        for i, frame in enumerate(frames):
            if i == 0:
                for _ in range(25):
                    retargeter.retarget({n: [p.copy(), q.copy()] for n, (p, q) in frame.items()})
            pose = retargeter.retarget(frame)
            # Refine supporting ankle Z targets using actual BSRL sole geometry.
            # This fixes bilateral standing support without shifting swing feet
            # onto the floor or pretending to simulate contact forces.
            for iteration in range(4 if category == 'stand' else 0):
                height = foot_clearances(grounder, pose, foot_ids)
                residual = .0005-height
                if np.abs(residual[support[i]]).max() < .001:
                    break
                common = float(np.mean(residual[support[i]]))
                frame['Pelvis'][0][2] += .5*common
                for side_index, side in enumerate(('Left', 'Right')):
                    if support[i, side_index]:
                        frame[side+'Ankle'][0][2] += residual[side_index]
                pose = retargeter.retarget(frame)
            qpos.append(pose)
        qpos = np.array(qpos)
        if not np.isfinite(qpos).all():
            raise ValueError(f'Nonfinite IK: {stem}')
        raw_minimum = np.array([grounder.qpos_min_z(q) for q in qpos])
        # Scope is explicitly non-flight. A 0.5mm clearance avoids z fighting.
        shift = .0005-raw_minimum
        qpos[:, 2] += shift
        feet_stats, clearance = sole_metrics(grounder, qpos, args.fps)
        world_pos, body_names = body_positions(retargeter.model, qpos)
        local_pos, local_names = local_body_positions(retargeter.model, qpos)
        if local_names != body_names:
            raise ValueError('Inconsistent FK names')
        joint_ids = np.flatnonzero(retargeter.model.jnt_type != 0)
        values = qpos[:, retargeter.model.jnt_qposadr[joint_ids]]
        limits = retargeter.model.jnt_range[joint_ids]
        violation = float(max(0., np.max(limits[:, 0]-values), np.max(values-limits[:, 1])))
        if violation > 1e-5:
            raise ValueError(f'Joint-limit violation {stem}: {violation}')
        robot_stats = trajectory_metrics(qpos[:, :3], Rotation.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix(), args.fps)
        audit.update(status='kinematic_preview', fps=args.fps, frames=len(qpos), duration_s=len(qpos)/args.fps,
            scale=scale, removed_initial_yaw_rad=yaw_removed, source_display_floor_m=source_floor,
            robot_trajectory=robot_stats, joint_limit_violation_rad=violation,
            joint_speed_max_rad_s=float(np.abs(np.diff(qpos[:, 7:], axis=0)*args.fps).max()),
            grounding='bilateral_standing_sole_targets_and_nonflight_lowest_sole_projection_0.5mm',
            minimum_target_ankle_width_m=MINIMUM_ANKLE_WIDTH,
            source_support_uncertain_fallback_frames=uncertain_support,
            source_support_use='Bilateral standing enforced; walking support heuristic is diagnostic only.',
            inferred_support_clearance_p50_p95_max_m=np.quantile(clearance[support], [.5, .95, 1]).tolist(),
            per_foot_lateral_adaptation_p50_max_m=np.quantile(width_correction, [.5, 1]).tolist(),
            root_z_projection_min_p50_max_m=np.quantile(shift, [0, .5, 1]).tolist(),
            warnings=['Kinematic IK replay, not a trained policy or dynamic simulation.',
                      'Non-flight support projection only; no horizontal foot pinning or contact-force validation.',
                      'Source labels and split retained; preview files are NOT installed in AMP training.'], **feet_stats)
        payload = dict(fps=args.fps, root_pos=qpos[:, :3], root_rot=qpos[:, [4, 5, 6, 3]], dof_pos=qpos[:, 7:],
            dof_names=joint_names_in_qpos_order(retargeter.model), local_body_pos=local_pos, link_body_list=body_names,
            key_body_pos=world_pos[:, [body_names.index(n) for n in BSRL_KEY_BODY_NAMES]],
            key_body_names=list(BSRL_KEY_BODY_NAMES), source_metadata=clip, source_times=times,
            joint_coordinate_convention=BSRL_JOINT_COORDINATE_CONVENTION,
            grounding_convention='cmu_nonflight_standing_support_preview_v3', kinematic_preview_only=True)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open('xb') as stream:
            pickle.dump(payload, stream, protocol=4)
        source_path = args.output / 'source_fk' / category / (stem+'.npz')
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open('xb') as stream:
            np.savez_compressed(stream, positions=scaled_points, bones=np.array(skeleton['names']),
                                parents=skeleton['parents'], times=times, sole_clearance=clearance, inferred_support=support)
        save_json(args.output / 'metrics' / category / (stem+'.json'), audit)
        all_audits.append(audit)
        print(f'[DONE] {stem}: {len(qpos)} frames, yaw {robot_stats["net_yaw_deg"]:.1f} deg, '
              f'clearance {feet_stats["sole_min_p50_p95_max_m"]}', flush=True)
    save_json(args.output / 'audit.json', dict(clips=all_audits, source_manifest=str(args.source/args.manifest_name)))


if __name__ == '__main__':
    main()
