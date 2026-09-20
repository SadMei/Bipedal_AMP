"""Small, strict CMU ASF/AMC reader and BABEL helpers (no SMPL body model).

ASF coordinates are X-left/Y-up/Z-forward. CMU's documented length factor is
0.0254 / ASF.units.length. All public outputs use metres and X-forward/Y-left/Z-up.
Only the fully specified, degree-valued CMU dialect is supported; fail on others.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

AXES = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def parse_asf(text):
    sections = {}
    current = None
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith(':'):
            current = line.split()[0][1:]
            sections[current] = []
        elif current:
            sections[current].append(line)
    units = dict(line.split(maxsplit=1) for line in sections['units'])
    root = dict(line.split(maxsplit=1) for line in sections['root'])
    if units.get('angle') != 'deg' or root['axis'] != 'XYZ':
        raise ValueError('Only degree-valued XYZ CMU ASF files are supported')
    if root['order'].lower().split() != ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']:
        raise ValueError('Unsupported root channel order')
    if any(float(v) != 0 for k in ('position', 'orientation') for v in root[k].split()):
        raise ValueError('Nonzero ASF root reference pose requires a different adapter')
    scale = .0254 / float(units['length'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid ASF length units')
    bones = {'root': dict(direction=np.zeros(3), length=0., axis=np.zeros(3),
                          dof=['tx', 'ty', 'tz', 'rx', 'ry', 'rz'])}
    block = None
    for line in sections['bonedata']:
        if line == 'begin':
            block = {}
        elif line == 'end':
            name = block['name']
            if name in bones:
                raise ValueError(f'Duplicate bone {name}')
            axis = block['axis'].split()
            if axis[-1] != 'XYZ':
                raise ValueError('Unsupported ASF axis order')
            dof = block.get('dof', '').split()
            if any(d not in ('rx', 'ry', 'rz') for d in dof) or len(set(dof)) != len(dof):
                raise ValueError('Invalid bone channels')
            bones[name] = dict(direction=np.array(block['direction'].split(), float),
                               length=float(block['length']) * scale,
                               axis=np.array(axis[:3], float), dof=dof)
            block = None
        elif block is not None and not line.startswith('('):
            key, value = line.split(maxsplit=1)
            block[key] = value
    parent = {}
    for line in sections['hierarchy']:
        if line in ('begin', 'end'):
            continue
        p, *children = line.split()
        for child in children:
            if child in parent or p not in bones or child not in bones:
                raise ValueError('Invalid ASF hierarchy')
            parent[child] = p
    names = ['root']
    while len(names) < len(bones):
        ready = [n for n in bones if n not in names and parent.get(n) in names]
        if not ready:
            raise ValueError('Disconnected or cyclic ASF hierarchy')
        names.extend(ready)
    return dict(bones=bones, names=names, parents=np.array([-1] + [names.index(parent[n]) for n in names[1:]]),
                metres_per_unit=scale)


def parse_amc(text, skeleton):
    if ':DEGREES' not in text or ':FULLY-SPECIFIED' not in text:
        raise ValueError('AMC must be fully specified and expressed in degrees')
    expected = {n: len(b['dof']) for n, b in skeleton['bones'].items() if b['dof']}
    frames, frame = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', ':')):
            continue
        if line.isdigit():
            if int(line) != len(frames) + 1:
                raise ValueError('AMC frame IDs must be contiguous, one-based')
            frame = {}
            frames.append(frame)
            continue
        name, *values = line.split()
        if frame is None or name not in expected or name in frame or len(values) != expected[name]:
            raise ValueError(f'Invalid AMC channels at {name}')
        frame[name] = np.array(values, float)
    if not frames or any(set(f) != set(expected) for f in frames):
        raise ValueError('Empty or incomplete AMC frames')
    result = {n: np.stack([f[n] for f in frames]) for n in expected}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite AMC samples')
    return result


def forward_kinematics(skeleton, motion):
    names, parents = skeleton['names'], skeleton['parents']
    count = len(motion['root'])
    positions = np.zeros((count, len(names), 3))
    matrices = np.zeros((count, len(names), 3, 3))
    positions[:, 0] = motion['root'][:, :3] * skeleton['metres_per_unit']
    matrices[:, 0] = Rotation.from_euler('xyz', motion['root'][:, 3:], degrees=True).as_matrix()
    for i, name in enumerate(names[1:], 1):
        bone = skeleton['bones'][name]
        angles = np.zeros((count, 3))
        for column, channel in enumerate(bone['dof']):
            angles[:, 'xyz'.index(channel[1])] = motion[name][:, column]
        c = Rotation.from_euler('xyz', bone['axis'], degrees=True).as_matrix()
        local = c @ Rotation.from_euler('xyz', angles, degrees=True).as_matrix() @ c.T
        matrices[:, i] = matrices[:, parents[i]] @ local
        positions[:, i] = positions[:, parents[i]] + (matrices[:, i] @ (bone['direction'] * bone['length']))
    return positions @ AXES.T, AXES @ matrices @ AXES.T


def load_babel(archive):
    """Use dense train/val labels only; preserve original splits in all outputs."""
    result = {}
    with ZipFile(archive) as z:
        for split in ('train', 'val'):
            data = json.loads(z.read(f'babel_v1.0_release/{split}.json'))
            result.update({str(sid): dict(record, split=split) for sid, record in data.items()})
    return result


def cmu_identity(feat_p):
    m = re.fullmatch(r'CMU/CMU/(\d+)/(\d+)_(\d+)_poses\.npz', feat_p)
    if m is None or int(m[1]) != int(m[2]):
        raise ValueError(f'Not a simple CMU source identity: {feat_p}')
    subject = str(int(m[1])).zfill(2)
    return subject, f'{subject}_{int(m[3]):02d}'


def sample_interval(positions, matrices, fps, start, end, output_fps=60.):
    """Keep physical time; never stretch a source to match BABEL's duration."""
    if not 0 <= start < end or fps <= 0 or output_fps <= 0:
        raise ValueError('Invalid sample interval/rate')
    last = (len(positions)-1)/fps
    if start > last or end > len(positions)/fps + 1e-6:
        raise ValueError('Interval exceeds the source')
    times = start + np.arange(int(np.ceil((end-start)*output_fps-1e-9))) / output_fps
    times = times[times <= last+1e-9]
    if len(times) < 2:
        raise ValueError('Not enough samples')
    index = np.minimum(times * fps, len(positions)-1)
    lo, hi = np.floor(index).astype(int), np.ceil(index).astype(int)
    alpha = (index-lo)[:, None, None]
    points = positions[lo]*(1-alpha) + positions[hi]*alpha
    source_t = np.arange(len(positions))/fps
    rotations = np.stack([Slerp(source_t, Rotation.from_matrix(matrices[:, j]))(times).as_matrix()
                          for j in range(matrices.shape[1])], axis=1)
    return points, rotations, times


def align_initial_heading(points, rotations):
    yaw = np.arctan2(rotations[0, 0, 1, 0], rotations[0, 0, 0, 0])
    alignment = Rotation.from_euler('z', -yaw).as_matrix()
    origin = points[0, 0].copy()
    origin[2] = 0.
    return (points-origin) @ alignment.T, alignment @ rotations, float(yaw)


def trajectory_metrics(points, root_matrices, fps):
    root = np.asarray(points)
    smooth = max(1, min(len(root), round(.2*fps)))
    yaw = np.unwrap(np.arctan2(root_matrices[:, 1, 0], root_matrices[:, 0, 0]))
    yaw = uniform_filter1d(yaw, smooth, mode='nearest')
    velocity = np.gradient(uniform_filter1d(root, smooth, axis=0, mode='nearest'), 1/fps, axis=0)
    forward = np.cos(yaw)*velocity[:, 0] + np.sin(yaw)*velocity[:, 1]
    lateral = -np.sin(yaw)*velocity[:, 0] + np.cos(yaw)*velocity[:, 1]
    total = np.mean(np.abs(forward)) + np.mean(np.abs(lateral)) + 1e-12
    return dict(net_yaw_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
                yaw_range_deg=float(np.rad2deg(np.ptp(yaw))),
                root_xy_extent_m=float(np.linalg.norm(np.ptp(root[:, :2], axis=0))),
                net_xy_m=float(np.linalg.norm(root[-1, :2]-root[0, :2])),
                lateral_fraction=float(np.mean(np.abs(lateral))/total),
                median_forward_m_s=float(np.median(forward)), median_lateral_m_s=float(np.median(lateral)),
                median_xy_speed_m_s=float(np.median(np.linalg.norm(velocity[:, :2], axis=1))))


def category_passes(category, metrics):
    if category == 'straight':
        return (metrics['yaw_range_deg'] <= 15 and metrics['lateral_fraction'] < .25
                and metrics['net_xy_m'] >= .4 and metrics['median_forward_m_s'] > .1)
    if category == 'turn':
        return abs(metrics['net_yaw_deg']) >= 20
    if category == 'sideways':
        return (metrics['yaw_range_deg'] <= 20 and metrics['lateral_fraction'] > .7
                and metrics['net_xy_m'] >= .25)
    if category == 'stand':
        return metrics['root_xy_extent_m'] < .10 and metrics['median_xy_speed_m_s'] < .08
    raise ValueError(category)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Refusing to replace differing output: {path}')
    else:
        with path.open('x') as stream:
            stream.write(content)
