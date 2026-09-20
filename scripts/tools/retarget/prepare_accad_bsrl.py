"""Extract an explicit ACCAD SMPL-X preview selection from local user archives.

No downloads, unsafe archive extraction, overwrites, or AMP dataset installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from accad_smplx import load_motion
from cmu_babel import load_babel, save_json
from accad_reference import reference_weights, resolve_reference


SELECTION = [
    ('straight_female1_B3', 'straight', 'Female1Walking_c3d/B3_-_walk1_stageii.npz', 'Forward walk', False),
    ('turn_left_male2_B10', 'turn', 'Male2Walking_c3d/B10_-__Walk_turn_left_45_stageii.npz', 'Walk turn left 45', True),
    ('turn_right_male2_B14', 'turn', 'Male2Walking_c3d/B14_-__Walk_turn_right_45_t2_stageii.npz', 'Walk turn right 45', True),
    ('side_left_male1_B22', 'sideways', 'Male1Walking_c3d/Walk_B22_-_Side_step_left_stageii.npz', 'Side step left', True),
    ('side_right_male1_B23', 'sideways', 'Male1Walking_c3d/Walk_B23_-_Side_step_right_stageii.npz', 'Side step right', True),
    ('side_left_male2_B22', 'sideways', 'Male2Walking_c3d/B22_-__side_step_left_stageii.npz', 'Side step left', True),
    ('side_right_male2_B23', 'sideways', 'Male2Walking_c3d/B23_-__side_step_right_stageii.npz', 'Side step right', True),
    ('stand_male2_A1', 'stand', 'Male2General_c3d/A1-_Stand_stageii.npz', 'Stand', False),
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_verified(path, payload):
    path = Path(path)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if sha(path) != digest:
            raise FileExistsError(f'Differing output exists: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
    return dict(path=str(path), bytes=len(payload), sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('/root/mjw/ACCAD.tar.bz2'))
    parser.add_argument('--models-zip', type=Path, default=Path('/root/mjw/models_smplx_v1_1.zip'))
    parser.add_argument('--babel', type=Path, default=Path('/root/mjw/babel_v1-0_release.zip'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', choices=['preview8', 'g1-full'], default='preview8')
    parser.add_argument('--g1-config', type=Path, default=Path(__file__).resolve().parents[3]/
        'source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py')
    parser.add_argument('--archive-index', type=Path,
                        default=Path('/root/mjw/GMR/motion_data/ACCAD_smplx_source/manifest.json'))
    args = parser.parse_args()
    reference = None
    if args.selection == 'g1-full':
        index_manifest = json.loads(args.archive_index.read_text())
        if sha(args.archive) != index_manifest['archive_sha256']:
            raise ValueError('Archive does not match the existing full archive index')
        weights = reference_weights(args.g1_config)
        selection, mappings = resolve_reference(weights,
            [m['path'] for m in index_manifest['archive_members']], SELECTION)
        reference = dict(config_path=str(args.g1_config), config_sha256=sha(args.g1_config),
                         reference_name_count=len(weights), mappings=mappings,
                         unresolved_subject_names=[m['reference_name'] for m in mappings if len(m['source_candidates'])>1])
    else:
        selection = [dict(stem=s, category=c, relative=r, label=l, in_local_g1_amp_name_list=g)
                     for s,c,r,l,g in SELECTION]
    requested = {'ACCAD/'+s['relative'] for s in selection}
    files, members = {}, []
    with tarfile.open(args.archive, 'r|bz2') as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive path')
            if not item.isfile():
                continue
            members.append(dict(path=item.name, bytes=item.size))
            if item.name in requested:
                if item.size > 128*1024**2:
                    raise ValueError('Oversized motion file')
                files[item.name] = write_verified(args.output/item.name, archive.extractfile(item).read())
                print('[EXTRACTED]', item.name, flush=True)
    if requested-set(files):
        raise ValueError(f'Missing selected sources: {requested-set(files)}')
    models = {}
    with zipfile.ZipFile(args.models_zip) as archive:
        for gender in ('male', 'female'):
            name = f'models/smplx/SMPLX_{gender.upper()}.npz'
            models[gender] = write_verified(args.output/name, archive.read(name))
            print('[MODEL]', name, flush=True)
    babel = load_babel(args.babel)
    index = {}
    for sid, record in babel.items():
        p = PurePosixPath(record['feat_p'])
        if not str(p).startswith('ACCAD/ACCAD/'):
            continue
        identity = '/'.join(p.parts[2:-1])+'/'+p.stem.removesuffix('_poses').replace(' ', '_')+'_stageii.npz'
        index.setdefault(identity, []).append(dict(sid=sid, split=record['split'], feat_p=record['feat_p'],
            duration_s=record['dur'], frame_ann=record.get('frame_ann'), seq_ann=record.get('seq_ann')))
    clips = []
    for selected in selection:
        stem, category, relative, label, g1 = [selected[k] for k in
            ('stem', 'category', 'relative', 'label', 'in_local_g1_amp_name_list')]
        path = args.output/'ACCAD'/relative
        m = load_motion(path)
        clips.append(dict(stem=stem, category=category, motion='ACCAD/'+relative,
            label=label, source_path=str(path), gender=m['gender'], source_fps=m['fps'],
            source_frames=len(m['trans']), source_duration_s=(len(m['trans'])-1)/m['fps'],
            in_local_g1_amp_name_list=g1, babel_matches=index.get(relative, []),
            label_basis='ACCAD original motion name; full sequence, trajectory screening required'))
        clips[-1].update({k: v for k,v in selected.items() if k not in ('relative',)})
    manifest = dict(dataset='AMASS ACCAD SMPL-X', archive=str(args.archive), archive_sha256=sha(args.archive),
        models_archive=str(args.models_zip), models_archive_sha256=sha(args.models_zip),
        source_files=files, body_models=models, archive_members=members, clips=clips,
        kinematic_preview_only=True, downloaded_bytes=0,
        notes=([f'All {reference["reference_name_count"]} local G1 reference names matched; all ambiguous subject candidates retained.']
               if reference else ['Six selected names also occur in the third-party Legged Lab G1 AMP config.']) + [
               'Straight walking and standing are additional ACCAD candidates for the requested four categories.',
               'Not a claim about Unitree proprietary training; no synthetic mirroring or heading removal per frame.',
               'BABEL train/val annotations are provenance only, not a new training split or crop.'],
        licenses=['https://amass.is.tue.mpg.de/license.html', 'https://smpl-x.is.tue.mpg.de/modellicense.html',
                  'https://babel.is.tue.mpg.de/license.html'])
    if reference is not None:
        manifest.update(selection=args.selection, reference=reference)
    save_json(args.output/'manifest.json', manifest)
    print('[DONE]', len(clips), 'clips; source and model archives preserved', flush=True)


if __name__ == '__main__':
    main()
