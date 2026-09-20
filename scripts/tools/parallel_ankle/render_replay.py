"""Render recorded closed/serial trajectories without re-running the policy."""
import argparse
# Entry point relocated to Bipedal_AMP/scripts/tools/parallel_ankle.
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ankle_camera(camera, data, side=None):
    rotation = data.body("base_link").xmat.reshape(3, 3)
    yaw = np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
    if side is None:
        camera.lookat[:] = (data.body("link_right_ankle_roll").xpos + data.body("link_left_ankle_roll").xpos)/2 + [0,0,.12]
        camera.distance, camera.azimuth = .95, yaw + 130
    else:
        camera.lookat[:] = data.body(f"link_{side}_ankle_roll").xpos + [0, 0, .105]
        camera.distance = .67
        camera.azimuth = yaw + (120 if side == "left" else 240)
    camera.elevation = -12


def video_label(manifest):
    if manifest.get("checkpoint_label"):
        return manifest["checkpoint_label"]
    if manifest.get("checkpoint"):
        return Path(manifest["checkpoint"]).stem
    return "checkpoint unspecified"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record",type=Path)
    parser.add_argument("video",type=Path)
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--ankle-closeup", action="store_true", help="Track both feet to inspect linkage connections")
    views.add_argument("--split-ankles", action="store_true", help="Independent enlarged left/right ankle panels")
    views.add_argument('--body-and-feet',action='store_true',help='Fixed-azimuth body view beside enlarged feet')
    parser.add_argument("--slowdown", type=float, default=1., help="Playback duration multiplier; 2 means half speed")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", type=Path, help="Save a still at about 5 simulation seconds")
    args=parser.parse_args()
    if args.slowdown <= 0 or args.width <= 0 or args.height <= 0 or args.width % 2:
        parser.error("Positive slowdown/dimensions and an even width are required")
    if args.video.exists(): raise FileExistsError(args.video)
    if args.preview and args.preview.exists(): raise FileExistsError(args.preview)
    manifest=json.loads((args.record/"manifest.json").read_text())
    summary=json.loads((args.record/"summary.json").read_text())
    model=mujoco.MjModel.from_xml_path(str(args.record/"model.xml"))
    data=mujoco.MjData(model)
    panel_width=args.width//2 if (args.split_ankles or args.body_and_feet) else args.width
    model.vis.global_.offwidth=panel_width; model.vis.global_.offheight=args.height
    renderer=mujoco.Renderer(model,height=args.height,width=panel_width)
    camera=mujoco.MjvCamera(); camera.distance=3.; camera.azimuth=125; camera.elevation=-18
    option=mujoco.MjvOption(); option.geomgroup[3]=0
    args.video.parent.mkdir(parents=True,exist_ok=True)
    fps=1/manifest.get("policy_dt", .02)/args.slowdown
    writer=imageio.get_writer(str(args.video),fps=fps,codec="libx264",quality=8,
                             macro_block_size=1,ffmpeg_params=["-threads","2","-pix_fmt","yuv420p"])
    try:
        font=ImageFont.truetype("DejaVuSans.ttf", max(16, round(args.height/45)))
    except OSError:
        font=ImageFont.load_default()
    count=0
    try:
        with np.load(args.record/"policy.npz") as record:
            quat=record['qpos'][:,3:7]
            w,x,y,z=quat.T
            yaw=np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
            yaw-=yaw[0]
            for i,t in enumerate(record["time_s"]):
                data.qpos[:]=record["qpos"][i]; data.qvel[:]=record["qvel"][i]
                data.time=float(t); mujoco.mj_forward(model,data)
                if args.body_and_feet:
                    camera.lookat[:]=data.body('base_link').xpos + [0,0,-.15]
                    camera.distance,camera.azimuth,camera.elevation=2.7,125,-18
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    body=renderer.render().copy()
                    ankle_camera(camera,data)
                    separation=np.linalg.norm(data.body('link_left_ankle_roll').xpos-data.body('link_right_ankle_roll').xpos)
                    camera.distance=max(1.05,1.7*(separation+.30))
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    frame=Image.fromarray(np.concatenate([body,renderer.render().copy()],axis=1))
                elif args.split_ankles:
                    panels=[]
                    for side in ("left", "right"):
                        ankle_camera(camera, data, side)
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        panels.append(renderer.render().copy())
                    frame=Image.fromarray(np.concatenate(panels, axis=1))
                else:
                    if args.ankle_closeup:
                        ankle_camera(camera, data)
                    else:
                        camera.lookat[:]=data.body("base_link").xpos
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    frame=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(frame)
                header=75 if args.height>=1000 else 60
                draw.rectangle((0,0,args.width,header),fill=(15,15,15))
                experiment=manifest.get('experiment_label','BSRL')
                label=f"{experiment} {video_label(manifest)} | {manifest['topology']} | cmd={manifest['command']} | sim t={t:.2f}s | {1/args.slowdown:g}x"
                draw.text((15,8),label,fill="white",font=font)
                limit_label="independent ankle joint caps=112 Nm" if manifest['topology']=="serial" else f"ankle motor limit={manifest['ankle_limit_nm']} Nm"
                draw.text((15,header//2+2),f"{limit_label} | body vx={record['base_velocity_b'][i,3]:.2f} m/s | no reset | blue=long, orange=short",fill="white",font=font)
                if args.body_and_feet:
                    draw.line((panel_width,header,panel_width,args.height),fill='white',width=2)
                    draw.rectangle((0,args.height-42,args.width,args.height),fill=(15,15,15))
                    draw.text((12,args.height-33),f'World-fixed azimuth | cumulative yaw {yaw[i]:+.1f} deg | X/Y {data.qpos[0]:+.2f}/{data.qpos[1]:+.2f} m',fill='white',font=font)
                    draw.text((panel_width+12,args.height-33),'Feet close-up | true recorded joint states',fill='white',font=font)
                    if (summary.get('fall') or summary.get('failure')) and i>=len(record['time_s'])-25:
                        draw.rectangle((10,header+10,args.width-10,header+45),fill=(105,25,20))
                        draw.text((18,header+17),f"Stopped at {summary['seconds']:.3f}s: {summary.get('fall') or summary.get('failure')}",fill='white',font=font)
                if args.split_ankles:
                    draw.line((panel_width,header,panel_width,args.height),fill="white",width=2)
                    for side,offset in (("left",0),("right",panel_width)):
                        angles=[np.rad2deg(data.joint(f"joint_{side}_ankle_{axis}").qpos[0]) for axis in ("pitch","roll")]
                        gaps=[np.linalg.norm(data.site(f"rod_end_{side}_{branch}").xpos-data.site(f"foot_end_{side}_{branch}").xpos)*1000 for branch in (1,2)]
                        draw.rectangle((offset+8,header+8,offset+panel_width-8,header+74),fill=(15,15,15))
                        draw.text((offset+18,header+12),f"{side.upper()} ankle | rear / outer view",fill="white",font=font)
                        draw.text((offset+18,header+42),f"P/R {angles[0]:+.1f}/{angles[1]:+.1f} deg | gap {max(gaps):.3f} mm",fill="white",font=font)
                if args.preview and i == min(len(record["time_s"])-1, round(5/manifest.get("policy_dt",.02))):
                    args.preview.parent.mkdir(parents=True,exist_ok=True)
                    frame.save(args.preview)
                writer.append_data(np.asarray(frame)); count+=1
                if count % 250 == 0:
                    print(f"Rendered {count}/{len(record['time_s'])} frames",flush=True)
    finally:
        writer.close(); renderer.close()
    print(json.dumps(dict(video=str(args.video.resolve()),frames=count,fps=fps,
                          seconds=count/fps,slowdown=args.slowdown,trajectory_stop=summary["fall"])),flush=True)


if __name__=="__main__": main()
