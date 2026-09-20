"""Four-category CMU/BABEL previews: source, BSRL, and enlarged feet.

Camera azimuth stays fixed in the initial-heading frame, so turns are visible.
No montage, policy rollout, or implicit installation into the AMP dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cmu_babel import save_json


def capsule(scene, start, end, color, radius=.009):
    if np.linalg.norm(end-start) < 1e-8:
        return
    if scene.ngeom >= scene.maxgeom:
        raise ValueError('Scene geom capacity exceeded')
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                       np.eye(3).ravel(), np.array(color, np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    scene.ngeom += 1


def path_inset(draw, positions, frame, font):
    x0, y0, size = 12, 108, 172
    draw.rectangle((x0, y0, x0+size, y0+size), fill=(24, 29, 36))
    draw.text((x0+8, y0+7), 'Root XY (fixed frame)', fill='white', font=font)
    xy = positions[:, :2]
    extent = max(.3, float(np.ptp(xy, axis=0).max()))
    centre = .5*(xy.min(axis=0)+xy.max(axis=0))
    pixels = (xy-centre)/extent*(size-42)
    pixels = np.column_stack([x0+size/2-pixels[:, 1], y0+size/2+15-pixels[:, 0]])
    draw.line([tuple(p) for p in pixels], fill=(130, 145, 160), width=2)
    x, y = pixels[frame]
    draw.ellipse((x-4, y-4, x+4, y+4), fill=(240, 180, 40))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--gmr-dir', type=Path, default=Path('/root/mjw/GMR'))
    parser.add_argument('--slowdown', type=float, default=2.)
    parser.add_argument('--still-only', action='store_true')
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--fixed-source-camera', action='store_true',
                        help='Show the source travel from one fixed world camera')
    args = parser.parse_args()
    if not 1 <= args.slowdown <= 4:
        parser.error('--slowdown must be in [1,4]')
    sys.path.insert(0, str(args.gmr_dir))
    from general_motion_retargeting.params import ROBOT_VIEWER_XML_DICT
    model = mujoco.MjModel.from_xml_path(str(ROBOT_VIEWER_XML_DICT['bsrl']))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 480, 720
    renderer = mujoco.Renderer(model, height=720, width=480)
    robot_option, source_option = mujoco.MjvOption(), mujoco.MjvOption()
    robot_option.geomgroup[3] = 0
    source_option.geomgroup[1] = source_option.geomgroup[3] = 0
    colors = {'left': (.22, .65, .95, 1.), 'right': (1., .55, .15, 1.)}
    for gid in range(model.ngeom):
        if model.geom_group[gid] == 1:
            for side, color in colors.items():
                if model.body(model.geom_bodyid[gid]).name == f'link_{side}_ankle_roll':
                    model.geom_rgba[gid] = color
    font = ImageFont.truetype('DejaVuSans.ttf', 19)
    small = ImageFont.truetype('DejaVuSans.ttf', 15)
    camera = mujoco.MjvCamera()
    try:
        for path in sorted((args.run / 'gmr').glob('*/*.pkl')):
            if args.clips and path.stem not in args.clips:
                continue
            with path.open('rb') as stream:
                motion = pickle.load(stream)
            category, stem = path.parent.name, path.stem
            with np.load(args.run / 'source_fk' / category / (stem+'.npz'), allow_pickle=False) as source:
                positions, names, parents = source['positions'], source['bones'].tolist(), source['parents']
                clearance = source['sole_clearance']
            audit = json.loads((args.run / 'metrics' / category / (stem+'.json')).read_text())
            qpos = np.zeros((len(motion['root_pos']), model.nq))
            qpos[:, :3], qpos[:, 3:7] = motion['root_pos'], motion['root_rot'][:, [3, 0, 1, 2]]
            for col, name in enumerate(motion['dof_names']):
                qpos[:, model.joint(name).qposadr[0]] = motion['dof_pos'][:, col]
            stride = max(1, round(motion['fps'] / (30*args.slowdown)))
            indices = list(range(0, len(qpos), stride))
            still_index = min(len(indices)-1, len(indices)//2)
            if args.still_only:
                indices = [indices[still_index]]
                still_index = 0
            folder = args.video_dir / category
            folder.mkdir(parents=True, exist_ok=True)
            video, still = folder / (stem+'.mp4'), folder / (stem+('_still.png' if args.still_only else '_preview.png'))
            if still.exists() or (not args.still_only and video.exists()):
                raise FileExistsError(still if still.exists() else video)
            writer = None if args.still_only else imageio.get_writer(str(video),
                fps=motion['fps']/stride/args.slowdown, codec='libx264', quality=8,
                macro_block_size=1, ffmpeg_params=['-threads', '2', '-pix_fmt', 'yuv420p'])
            try:
                for sequence_index, i in enumerate(indices):
                    data.qpos[:] = qpos[i]
                    mujoco.mj_forward(model, data)
                    panels = []
                    camera.azimuth, camera.elevation, camera.distance = 135, -14, 2.8
                    camera.lookat[:] = positions[i, 0] + [0, 0, .05]
                    if args.fixed_source_camera:
                        trajectory = positions[:, 0]
                        camera.lookat[:] = .5*(trajectory.min(axis=0)+trajectory.max(axis=0))
                        camera.lookat[2] = .7
                        camera.distance = max(3.0, 2.1*float(np.linalg.norm(np.ptp(trajectory[:, :2], axis=0)))+1.5)
                        camera.elevation = -24
                    renderer.update_scene(data, camera=camera, scene_option=source_option)
                    if args.fixed_source_camera:
                        for j in range(1, i+1, 3):
                            start, end = positions[max(0, j-3), 0].copy(), positions[j, 0].copy()
                            start[2] = end[2] = .004
                            capsule(renderer.scene, start, end, (.9, .7, .2, 1.), radius=.012)
                    for child, parent in enumerate(parents):
                        if parent < 0:
                            continue
                        name = names[child]
                        color = colors['left'] if name.startswith('l') and name not in ('lowerback', 'lowerneck') else colors['right'] if name.startswith('r') else (.65, .7, .77, 1.)
                        capsule(renderer.scene, positions[i, parent], positions[i, child], color)
                    panels.append(renderer.render().copy())
                    camera.lookat[:] = data.body('base_link').xpos + [0, 0, -.25]
                    camera.distance = 2.05
                    renderer.update_scene(data, camera=camera, scene_option=robot_option)
                    panels.append(renderer.render().copy())
                    camera.lookat[:] = .5*(data.body('link_left_ankle_roll').xpos + data.body('link_right_ankle_roll').xpos) + [0, 0, .035]
                    separation = np.linalg.norm(data.body('link_left_ankle_roll').xpos[:2] - data.body('link_right_ankle_roll').xpos[:2])
                    camera.distance, camera.elevation = max(.95, 2.1*(separation+.30)), -22
                    renderer.update_scene(data, camera=camera, scene_option=robot_option)
                    panels.append(renderer.render().copy())
                    frame = Image.fromarray(np.concatenate(panels, axis=1))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1440, 95), fill=(17, 21, 29))
                    dataset = motion.get('dataset_label', 'CMU + BABEL')
                    heading = f'{dataset} | {stem} | {1/args.slowdown:g}x | KINEMATIC IK - NOT a trained policy'
                    heading_font = font if draw.textlength(heading, font=font) < 1415 else small
                    draw.text((12, 7), heading, fill='white', font=heading_font)
                    draw.text((12, 35), f'Source time {motion["source_times"][i]:.3f}s | {motion["source_metadata"]["label"]} | blue LEFT / orange RIGHT | fixed camera azimuth', fill=(220, 225, 230), font=small)
                    if motion.get('category_screen_pass') is False:
                        draw.rectangle((485, 108, 950, 138), fill=(70, 39, 15))
                        draw.text((492, 114), 'Unsegmented source; category needs review', fill=(255, 210, 125), font=small)
                    source_title = motion.get('source_panel_title', 'ASF source skeleton (scaled)')
                    if args.fixed_source_camera:
                        source_title += ' | world-fixed view'
                    for col, title in enumerate([source_title, 'BSRL 12-DoF IK replay',
                                                 motion.get('feet_panel_title', 'Feet: non-flight support projection')]):
                        draw.text((col*480+12, 66), title, fill='white', font=small)
                    draw.rectangle((960, 678, 1440, 720), fill=(17, 21, 29))
                    draw.text((972, 686), f'Sole clearance L {clearance[i,0]*1000:.1f} / R {clearance[i,1]*1000:.1f} mm', fill='white', font=small)
                    path_inset(draw, qpos[:, :3], i, small)
                    if args.fixed_source_camera:
                        draw.rectangle((480, 678, 960, 720), fill=(17, 21, 29))
                        travel = np.linalg.norm(qpos[i, :2]-qpos[0, :2])
                        draw.text((492, 686), f'Robot XY distance from start: {travel:.2f} m', fill='white', font=small)
                    if sequence_index == still_index:
                        frame.save(still)
                    if writer is not None:
                        writer.append_data(np.asarray(frame))
                if writer is not None:
                    save_json(video.with_suffix('.json'), dict(clip=motion['source_metadata'],
                        playback_speed=1/args.slowdown, source_start_s=float(motion['source_times'][0]),
                        source_last_sample_s=float(motion['source_times'][-1]), rendered_frames=len(indices),
                        video_fps=motion['fps']/stride/args.slowdown, kinematic_only=True,
                        camera=('World-fixed source camera; root-following robot detail; XY trajectory inset'
                                if args.fixed_source_camera else 'World-fixed azimuth, root-following position, XY trajectory inset'),
                        grounding=audit['grounding']))
                print(f'[DONE] {still if args.still_only else video}', flush=True)
            finally:
                if writer is not None:
                    writer.close()
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
