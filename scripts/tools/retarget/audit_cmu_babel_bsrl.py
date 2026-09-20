"""Independently check CMU FK lengths, BSRL export FK, floor and video decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import imageio.v2 as imageio
import numpy as np

from cmu_babel import forward_kinematics, parse_amc, parse_asf, save_json


def check_source_fk(skeleton, positions, rotations):
    bones = [skeleton['bones'][n] for n in skeleton['names'][1:]]
    lengths = np.array([bone['length'] for bone in bones])
    direction_norms = np.linalg.norm([bone['direction'] for bone in bones], axis=-1)
    # Public ASF directions are rounded decimal vectors, not exactly unit length.
    # Check rigid FK against the actual input offsets without normalizing the data
    # or relaxing the 1e-9 m FK tolerance. Also bound the input rounding separately.
    measured = np.linalg.norm(positions[:, 1:] - positions[:, skeleton['parents'][1:]], axis=-1)
    error = float(np.abs(measured - lengths * direction_norms).max())
    orthogonal = float(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3)).max())
    direction_error = float(np.abs(direction_norms - 1).max())
    if (not np.isfinite([error, orthogonal, direction_error]).all()
            or error > 1e-9 or orthogonal > 1e-9 or direction_error > 1e-5):
        raise ValueError(f'Invalid source FK: offset_error={error}, '
                         f'orthogonality_error={orthogonal}, direction_norm_error={direction_error}')
    return dict(max_bone_length_error_m=error,
                max_nominal_length_rounding_error_m=float(np.abs(measured - lengths).max()),
                max_source_direction_norm_error=direction_error,
                max_rotation_orthogonality_error=orthogonal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--gmr-dir', type=Path, default=Path('/root/mjw/GMR'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.gmr_dir))
    import mujoco
    from general_motion_retargeting.params import ROBOT_VIEWER_XML_DICT
    audit = json.loads((args.run / 'audit.json').read_text())
    source_manifest = Path(audit['source_manifest'])
    manifest = json.loads(source_manifest.read_text())
    source = source_manifest.parent
    source_checks, checked = [], set()
    for c in manifest['clips']:
        if c['motion'] in checked:
            continue
        checked.add(c['motion'])
        skeleton = parse_asf((source/(c['subject']+'.asf')).read_text())
        motion = parse_amc((source/(c['motion']+'.amc')).read_text(), skeleton)
        positions, rotations = forward_kinematics(skeleton, motion)
        source_checks.append(dict(motion=c['motion'], **check_source_fk(skeleton, positions, rotations)))
    model = mujoco.MjModel.from_xml_path(str(ROBOT_VIEWER_XML_DICT['bsrl']))
    data = mujoco.MjData(model)
    output = []
    for p in sorted((args.run/'gmr').glob('*/*.pkl')):
        category, stem = p.parent.name, p.stem
        with p.open('rb') as stream:
            motion = pickle.load(stream)
        count = len(motion['root_pos'])
        qpos = np.tile(model.qpos0, (count, 1))
        qpos[:, :3], qpos[:, 3:7] = motion['root_pos'], motion['root_rot'][:, [3, 0, 1, 2]]
        for column, name in enumerate(motion['dof_names']):
            qpos[:, model.joint(name).qposadr[0]] = motion['dof_pos'][:, column]
        if not np.isfinite(qpos).all():
            raise ValueError(f'Nonfinite output {stem}')
        errors = []
        for i, frame in enumerate(qpos):
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            viewer = np.array([data.body(name).xpos.copy() for name in motion['key_body_names']])
            errors.append(np.max(np.abs(viewer-motion['key_body_pos'][i])))
        if max(errors) > 1e-5:
            raise ValueError(f'Viewer/export FK mismatch: {stem}')
        metrics = json.loads((args.run/'metrics'/category/(stem+'.json')).read_text())
        if metrics['joint_limit_violation_rad'] > 1e-5:
            raise ValueError('Joint-limit violation')
        if metrics['inter_foot_collision_distance_min_m'] < -.001:
            raise ValueError(f'Inter-foot penetration: {stem}')
        if max(abs(v-.0005) for v in metrics['sole_min_p50_p95_max_m']) > 1e-5:
            raise ValueError(f'Unexpected floor clearance: {stem}')
        video = args.video_dir/category/(stem+'.mp4')
        metadata = json.loads(video.with_suffix('.json').read_text())
        reader = imageio.get_reader(str(video))
        decoded = 0
        try:
            info = reader.get_meta_data()
            for frame in reader:
                if frame.shape != (720, 1440, 3):
                    raise ValueError(f'Unexpected video frame shape {frame.shape}')
                decoded += 1
        finally:
            reader.close()
        if decoded != metadata['rendered_frames']:
            raise ValueError(f'Incomplete video: {stem}: {decoded}')
        output.append(dict(stem=stem, category=category, frames=count, video_frames=decoded,
                           duration_s=decoded/info['fps'], video=str(video),
                           video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                           max_viewer_export_fk_error_m=float(max(errors))))
    if set(o['category'] for o in output) != {'straight', 'turn', 'sideways', 'stand'}:
        raise ValueError('Missing one of the four categories')
    report = dict(source_checks=source_checks, videos=output, success=True,
                  limitations=['Frame-count agreement and visual spot checks do not prove frame-exact AMASS fitting alignment.',
                               'No contact-force, dynamic balance, or AMP training validation.'])
    save_json(args.run/'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
