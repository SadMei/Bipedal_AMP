"""Strict, translation-preserving SMPL-X skeleton FK for ACCAD previews.

Uses the licensed, user-provided SMPL-X model's shaped joint locations, not an
approximate stick figure. Only the first 22 body joints are needed by BSRL.
No surface mesh LBS is needed for these joints; the result is checked against
the official smplx package by the preview audit.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# Columns are robot forward/left/up expressed in SMPL-X's reference frame.
SMPL_TO_ROBOT_BASIS = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
BODY_NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
              'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
              'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
              'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')


def load_motion(path):
    with np.load(path, allow_pickle=False) as data:
        required = ('surface_model_type', 'gender', 'mocap_frame_rate', 'trans',
                    'root_orient', 'pose_body', 'betas')
        if any(k not in data for k in required):
            raise ValueError(f'Not an AMASS stage-II SMPL-X motion: {path}')
        motion = {k: data[k].copy() for k in required}
    if str(motion.pop('surface_model_type').item()) != 'smplx':
        raise ValueError('SMPL-X required, not SMPL+H')
    motion['gender'] = str(motion['gender'].item())
    motion['fps'] = float(motion.pop('mocap_frame_rate').item())
    if motion['gender'] not in ('male', 'female', 'neutral') or not 1 <= motion['fps'] <= 1000:
        raise ValueError('Invalid gender or FPS')
    n = len(motion['trans'])
    for key, shape in [('trans', (n, 3)), ('root_orient', (n, 3)), ('pose_body', (n, 63))]:
        if motion[key].shape != shape or not np.isfinite(motion[key]).all():
            raise ValueError(f'Invalid {key}')
    if n < 2 or motion['betas'].ndim != 1 or not np.isfinite(motion['betas']).all():
        raise ValueError('Invalid frame count or shape coefficients')
    return motion


def sample_indices(n, source_fps, output_fps):
    """Exact physical-time decimation; reject rates requiring interpolation."""
    if n < 2 or not np.isfinite([source_fps, output_fps]).all() or min(source_fps, output_fps) <= 0:
        raise ValueError('Invalid sample count or frame rate')
    ratio = source_fps / output_fps
    if ratio < 1 or abs(ratio-round(ratio)) > 1e-9:
        raise ValueError('This adapter requires an integer downsampling ratio')
    return np.arange(0, n, round(ratio), dtype=int)


def shaped_skeleton(model_path, betas):
    with np.load(model_path, allow_pickle=False) as model:
        template = np.asarray(model['v_template'], dtype=np.float64)
        directions = np.asarray(model['shapedirs'][:, :, :len(betas)], dtype=np.float64)
        if directions.shape[-1] != len(betas):
            raise ValueError('Model has fewer shape coefficients than the motion')
        shaped = template + np.einsum('vci,i->vc', directions, betas)
        rest = np.asarray(model['J_regressor'], dtype=np.float64) @ shaped
        parents = model['kintree_table'][0].astype(np.int64)
        parents[0] = -1
    if not np.isfinite(rest).all() or len(rest) < 22:
        raise ValueError('Invalid shaped skeleton')
    return rest[:22], parents[:22]


def skeleton_fk(rest, parents, local_rotvec, trans):
    """Match SMPL-X rigid joint transforms, including its root rest offset."""
    n, joints = local_rotvec.shape[:2]
    if local_rotvec.shape != (n, len(rest), 3) or trans.shape != (n, 3):
        raise ValueError('Invalid pose dimensions')
    positions = np.empty((n, joints, 3), dtype=np.float64)
    rotations = np.empty((n, joints, 3, 3), dtype=np.float64)
    local = Rotation.from_rotvec(local_rotvec.reshape(-1, 3)).as_matrix().reshape(n, joints, 3, 3)
    positions[:, 0] = rest[0] + trans
    rotations[:, 0] = local[:, 0]
    for i in range(1, joints):
        p = int(parents[i])
        if not 0 <= p < i:
            raise ValueError('Skeleton must be in parent-first order')
        rotations[:, i] = rotations[:, p] @ local[:, i]
        positions[:, i] = positions[:, p] + rotations[:, p] @ (rest[i]-rest[p])
    return positions, rotations


def motion_fk(motion, model_path, output_fps=60.):
    indices = sample_indices(len(motion['trans']), motion['fps'], output_fps)
    rest, parents = shaped_skeleton(model_path, motion['betas'])
    angles = np.concatenate([motion['root_orient'][:, None], motion['pose_body'].reshape(-1, 21, 3)], axis=1)
    positions, rotations = skeleton_fk(rest, parents, angles[indices], motion['trans'][indices])
    error = np.max(np.abs((positions[:, 0]-positions[0, 0]) -
                          (motion['trans'][indices]-motion['trans'][indices[0]])))
    if error > 1e-9:
        raise ValueError('Root translation was lost')
    return positions, rotations, rest, parents, indices


def limb_reference_frame(down):
    """Build forward/left/up axes for a reference thigh or shank."""
    z = -np.asarray(down, dtype=float)
    z /= np.linalg.norm(z)
    x = np.array([0., 0., 1.])
    x -= x.dot(z)*z
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('Degenerate limb reference frame')
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def source_mesh_foot_heights(motion, model_path, indices, batch_size=32):
    """Actual fitted SMPL-X foot-surface minima, not ankle-joint heights.

    Body poses/shape/translation are unchanged. Zero hands and face cannot alter
    foot LBS vertices selected by ankle+foot skinning weight > 0.5.
    """
    import smplx
    import torch
    model = smplx.SMPLX(model_path=str(model_path), gender=motion['gender'],
                        num_betas=len(motion['betas']), use_pca=False, flat_hand_mean=True, ext='npz')
    masks = [model.lbs_weights[:, [a, f]].sum(dim=1) > .5 for a,f in [(7,10), (8,11)]]
    if any(int(mask.sum()) < 50 for mask in masks):
        raise ValueError('Insufficient SMPL-X foot vertices')
    values = []
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start+batch_size]; n = len(idx)
            output = model(betas=tensor(motion['betas'])[None].expand(n,-1),
                global_orient=tensor(motion['root_orient'][idx]), body_pose=tensor(motion['pose_body'][idx]),
                transl=tensor(motion['trans'][idx]), expression=torch.zeros(n,10),
                left_hand_pose=torch.zeros(n,45), right_hand_pose=torch.zeros(n,45),
                jaw_pose=torch.zeros(n,3), leye_pose=torch.zeros(n,3), reye_pose=torch.zeros(n,3))
            values.append(np.column_stack([output.vertices[:, mask, 2].min(dim=1).values.numpy() for mask in masks]))
    return np.concatenate(values).astype(np.float64)


def flight_clearance_target(foot_heights, scale, clearance=.0005, noise_allowance=.005):
    """Transfer source geometric sole rise, including flight, with a robust floor.

    This is a geometric heuristic, not a force/contact label. Only vertical
    correction is returned; walking/running phase transitions stay unsegmented.
    """
    heights = np.asarray(foot_heights, dtype=float)
    if heights.ndim != 2 or heights.shape[1] != 2 or not np.isfinite(heights).all() or scale <= 0:
        raise ValueError('Invalid source foot heights or scale')
    lowest = heights.min(axis=1)*scale
    floor = float(np.quantile(lowest, .05))
    target = clearance + np.maximum(0., lowest-floor-noise_allowance)
    return target, dict(method='source_smplx_foot_mesh_minimum_rise_v1',
        source_floor_quantile=.05, scaled_source_floor_m=floor,
        geometric_noise_allowance_m=noise_allowance,
        target_clearance_min_p50_max_m=np.quantile(target,[0,.5,1]).tolist(),
        geometric_air_fraction_over_1cm=float(np.mean(target > .01)),
        force_contact_labels=False)
