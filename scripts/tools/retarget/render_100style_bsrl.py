"""Render source skeleton, BSRL IK replay, and enlarged feet side by side."""

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

from retarget_100style_bsrl import LABELS, MOVEMENTS


def add_capsule(scene, start, end, rgba, radius=.01):
    if np.linalg.norm(end-start) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                       np.zeros(3), np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--gmr-dir", type=Path, default=Path("/root/mjw/GMR"))
    parser.add_argument("--movements", choices=MOVEMENTS, nargs="+", default=list(MOVEMENTS))
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--start", type=float, default=5.)
    parser.add_argument("--slowdown", type=float, default=2.)
    parser.add_argument("--video-dir", type=Path, help="Override the output video directory")
    parser.add_argument("--still-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.slowdown <= 0 or args.start < 0:
        parser.error("Invalid time arguments")
    sys.path.insert(0, str(args.gmr_dir))
    from general_motion_retargeting.params import ROBOT_VIEWER_XML_DICT

    model = mujoco.MjModel.from_xml_path(str(ROBOT_VIEWER_XML_DICT["bsrl"]))
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 480, 720
    renderer = mujoco.Renderer(model, height=720, width=480)
    robot_option, source_option = mujoco.MjvOption(), mujoco.MjvOption()
    robot_option.geomgroup[3] = 0
    source_option.geomgroup[1] = source_option.geomgroup[3] = 0
    colors = {"left": (.22, .65, .95, 1.), "right": (1., .55, .15, 1.)}
    for gid in range(model.ngeom):
        body = model.body(model.geom_bodyid[gid]).name
        if model.geom_group[gid] == 1:
            for side in colors:
                if body == f"link_{side}_ankle_roll":
                    model.geom_rgba[gid] = colors[side]
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    output_dir = args.video_dir if args.video_dir is not None else args.run / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.elevation = -16
    try:
        for movement in args.movements:
            stem = f"Neutral_{movement}"
            with (args.run / "gmr" / (stem + ".pkl")).open("rb") as stream:
                motion = pickle.load(stream)
            with np.load(args.run / "source_fk" / (stem + ".npz")) as source:
                positions, names, parents = source["positions"], source["bones"].tolist(), source["parents"]
            qpos = np.zeros((len(motion["root_pos"]), model.nq))
            qpos[:, :3] = motion["root_pos"]
            qpos[:, 3:7] = motion["root_rot"][:, [3, 0, 1, 2]]
            for column, name in enumerate(motion["dof_names"]):
                qpos[:, model.joint(name).qposadr[0]] = motion["dof_pos"][:, column]
            fps = motion["fps"]
            start = min(round(args.start*fps), max(0, len(qpos)-round(args.seconds*fps)))
            stop = min(len(qpos), start + round(args.seconds*fps))
            stride = max(1, round(fps / (30*args.slowdown)))
            indices = list(range(start, stop, stride))
            if args.still_only:
                indices = [indices[min(len(indices)-1, round(fps/stride))]]
            path = output_dir / (stem + ".mp4")
            still_path = output_dir / (stem + ("_still" if args.still_only else "_preview") + ".png")
            if (not args.still_only and path.exists()) or still_path.exists():
                raise FileExistsError(path if path.exists() else still_path)
            writer = None if args.still_only else imageio.get_writer(
                str(path), fps=fps/stride/args.slowdown, codec="libx264", quality=8,
                macro_block_size=1, ffmpeg_params=["-threads", "2", "-pix_fmt", "yuv420p"])
            try:
                for sequence_idx, frame_idx in enumerate(indices):
                    data.qpos[:] = qpos[frame_idx]
                    mujoco.mj_forward(model, data)
                    matrix = data.body("base_link").xmat.reshape(3, 3)
                    yaw = np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))
                    panels = []
                    camera.azimuth, camera.elevation = yaw+135, -14
                    camera.lookat[:] = positions[frame_idx, names.index("Hips")] + [0, 0, .08]
                    camera.distance = 2.75
                    renderer.update_scene(data, camera=camera, scene_option=source_option)
                    for child, parent in enumerate(parents):
                        if parent < 0:
                            continue
                        name = names[child]
                        rgba = colors["left"] if name.startswith("Left") else colors["right"] if name.startswith("Right") else (.65, .7, .77, 1.)
                        add_capsule(renderer.scene, positions[frame_idx, parent], positions[frame_idx, child], rgba)
                    panels.append(renderer.render().copy())
                    camera.lookat[:] = data.body("base_link").xpos + [0, 0, -.25]
                    camera.distance = 2.05
                    renderer.update_scene(data, camera=camera, scene_option=robot_option)
                    panels.append(renderer.render().copy())
                    camera.lookat[:] = .5*(data.body("link_left_ankle_roll").xpos + data.body("link_right_ankle_roll").xpos) + [0, 0, .03]
                    camera.distance, camera.elevation = .8, -22
                    renderer.update_scene(data, camera=camera, scene_option=robot_option)
                    panels.append(renderer.render().copy())
                    frame = Image.fromarray(np.concatenate(panels, axis=1))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1440, 89), fill=(17, 21, 29))
                    draw.text((14, 9), f"100STYLE {stem} | {LABELS[movement]} | {1/args.slowdown:g}x playback | KINEMATIC REPLAY - not a trained policy", font=font, fill="white")
                    source_time = (motion["source_metadata"]["start_frame"] + frame_idx) / fps
                    draw.text((14, 39), f"Source time {source_time:.2f}s | blue = LEFT, orange = RIGHT | constant floor offset; original turning retained", font=small_font, fill=(220, 225, 230))
                    for col, label in enumerate(("Source skeleton (scaled)", "BSRL 12-DoF GMR", "Feet close-up")):
                        draw.text((col*480+14, 66), label, font=small_font, fill="white")
                    for boundary in (480, 960):
                        draw.line((boundary, 90, boundary, 720), fill=(80, 85, 95), width=2)
                    if args.still_only or sequence_idx == min(len(indices)-1, round(2*fps/stride)):
                        frame.save(still_path)
                    if writer is not None:
                        writer.append_data(np.asarray(frame))
                    if (sequence_idx+1) % 180 == 0:
                        print(f"[{stem}] rendered {sequence_idx+1}/{len(indices)}", flush=True)
            finally:
                if writer is not None:
                    writer.close()
            if not args.still_only:
                metadata = {"video": path.name, "source_start_frame": motion["source_metadata"]["start_frame"] + start,
                            "source_stop_frame_exclusive": motion["source_metadata"]["start_frame"] + stop,
                            "clip_start_seconds": start/fps, "clip_duration_seconds": (stop-start)/fps,
                            "playback_speed": 1/args.slowdown, "kinematic_only": True,
                            "camera": "Three panels follow instantaneous root heading; original turns are preserved in motion data."}
                path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
            print(f"[DONE] {path if not args.still_only else still_path}", flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
