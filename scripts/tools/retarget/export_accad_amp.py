"""Explicit ACCAD training-format export; preserves full data and its quality warnings.

Does not mislabel source-clearance running motions as the old forced-grounding
dataset. Only the opt-in ACCAD training launcher accepts this convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import joblib
import numpy as np

from bsrl_conventions import (BSRL_AMP_JOINT_NAMES, BSRL_AMP_KEY_BODY_NAMES,
                              BSRL_JOINT_COORDINATE_CONVENTION, validate_bsrl_joint_positions)
from cmu_babel import save_json
from prepare_accad_bsrl import sha

ACCAD_GROUNDING = 'bsrl_accad_source_clearance_v1'
SOURCE_GROUNDINGS = {'source_mesh_clearance_transfer_with_flight_v1',
                    'bilateral_standing_sole_targets_and_nonflight_lowest_sole_projection_0.5mm'}


def training_payload(source):
    if source['joint_coordinate_convention'] != BSRL_JOINT_COORDINATE_CONVENTION:
        raise ValueError('Wrong BSRL joint convention')
    if source['grounding_convention'] not in SOURCE_GROUNDINGS:
        raise ValueError('Unknown ACCAD clearance convention')
    ji = [source['dof_names'].index(n) for n in BSRL_AMP_JOINT_NAMES]
    bi = [source['key_body_names'].index(n) for n in BSRL_AMP_KEY_BODY_NAMES]
    root_rot = np.asarray(source['root_rot'][:, [3,0,1,2]], dtype=np.float32).copy()
    # q and -q describe the same rotation; keep adjacent quaternion signs
    # continuous without changing any pose or yaw trajectory.
    flips = np.sum(root_rot[1:]*root_rot[:-1],axis=1) < 0
    root_rot[1:] *= np.cumprod(np.where(flips,-1.,1.))[:,None]
    output = dict(fps=float(source['fps']), root_pos=np.asarray(source['root_pos'],dtype=np.float32),
        root_rot=root_rot, dof_pos=np.asarray(source['dof_pos'][:,ji],dtype=np.float32),
        key_body_pos=np.asarray(source['key_body_pos'][:,bi],dtype=np.float32), loop_mode=0,
        joint_names=list(BSRL_AMP_JOINT_NAMES), key_body_names=list(BSRL_AMP_KEY_BODY_NAMES),
        joint_coordinate_convention=BSRL_JOINT_COORDINATE_CONVENTION,
        grounding_convention=ACCAD_GROUNDING, source_grounding_convention=source['grounding_convention'],
        source_metadata=source['source_metadata'], source_times=source['source_times'],
        training_quality_validated=False, source_kinematic_only=True,
        processing='format_only_xyzw_to_wxyz_joint_order_float32_clamp_no_crop_no_smoothing')
    validate_bsrl_joint_positions(output['dof_pos'], output['joint_names'], 'ACCAD export')
    count = len(output['root_pos'])
    for key,shape in [('root_pos',(count,3)),('root_rot',(count,4)),('dof_pos',(count,12)),
                      ('key_body_pos',(count,2,3))]:
        if output[key].shape != shape or not np.isfinite(output[key]).all():
            raise ValueError('Invalid output '+key)
    if count<2 or not np.allclose(np.linalg.norm(root_rot,axis=1),1.,atol=1e-5):
        raise ValueError('Invalid quaternion/frame count')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.run=args.run.resolve(strict=True); args.output=args.output.resolve()
    verification=json.loads((args.run/'verification.json').read_text())
    catalog=json.loads((args.run/'dataset_catalog.json').read_text())
    if not verification['success'] or catalog['reference_name_count']!=30 or catalog['converted_source_count']!=34:
        raise ValueError('Expected verified full 34-clip ACCAD conversion')
    records=[]
    for check in verification['checks']:
        stem,category=check['stem'],check['category']
        src=args.run/'gmr'/category/(stem+'.pkl')
        if sha(src)!=check['pkl_sha256']:
            raise ValueError('Source changed '+stem)
        with src.open('rb') as stream:
            raw=pickle.load(stream)
        payload=training_payload(raw)
        target=args.output/(stem+'.pkl')
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            joblib.dump(payload,stream)
        loaded=joblib.load(target)
        for key in ('root_pos','root_rot','dof_pos','key_body_pos'):
            np.testing.assert_array_equal(loaded[key],payload[key])
        np.testing.assert_allclose(loaded['root_pos'][-1]-loaded['root_pos'][0],
                                   raw['root_pos'][-1]-raw['root_pos'][0],atol=2e-6,rtol=0)
        records.append(dict(stem=stem,category=category,path=str(target),sha256=sha(target),
            source=str(src),source_sha256=check['pkl_sha256'],frames=len(payload['root_pos']),
            fps=payload['fps'],weight=1.,quality_review_flags=check['quality_review_flags']))
    if len(records)!=34:
        raise ValueError('Missing ACCAD clips')
    save_json(args.output/'manifest.json',dict(dataset='ACCAD G1 full source candidates -> BSRL AMP',
        source_run=str(args.run),source_catalog_sha256=sha(args.run/'dataset_catalog.json'),
        source_verification_sha256=sha(args.run/'verification.json'),
        grounding_convention=ACCAD_GROUNDING,files=records,clip_count=34,reference_names=30,
        frames=sum(r['frames'] for r in records),duration_s=sum(r['frames']/r['fps'] for r in records),
        sampling='equal_weight_per_converted_clip_both_arms_identical',
        quality_warnings_preserved=True,training_quality_validated=False,
        processing='no_cropping_no_smoothing_no_remirroring_no_regrounding',
        warnings=['Eight clips exceed the 20rad/s manual review threshold; retained for this requested experiment.',
                  'Running sole clearance is geometric, not force-contact validated.',
                  'Two reference names have both subject variants; extra B3/A1 retained.']))
    print(f'[EXPORTED] {len(records)} motions / {sum(r["frames"] for r in records)} frames -> {args.output}')


if __name__=='__main__':
    main()
