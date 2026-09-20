#!/usr/bin/env python3
"""Delivered door ONNX in the user's T1 sim2sim plant, with vendor PD gains.

Only the vendor policy's observation/action contract is adapted. The real user's
BSRLMujoco/T1 runtime supplies contact settings, implicit-fast integration,
position-PD implementation, effort/velocity limits and its 1ms physics loop.
No training, source XML or deployment defaults are modified on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

import record_door_amp_audit as vendor


def record(args):
    import imageio.v2 as imageio
    import mujoco
    import onnxruntime as ort
    from PIL import Image, ImageDraw, ImageFont

    simtool = Path("/root/mjw/mjw_simtool")
    sys.path.insert(0, str(simtool / "scripts"))
    import sim2simT1AMP.run_sim2sim_standalone as t1
    core = t1.core
    xml = simtool / "scripts/sim2simT1/assets/men_t1_path_25dofs_collision_mc_amt.xml"
    folder = Path("/root/mjw/door/document") / ("mujoco_sim_0824" if args.variant == "0827" else "mujoco_sim_0916")
    source = folder / "zy_25dof_mujoco_sim.py"
    policy = folder / "policy" / ("0827-27-91000.onnx" if args.variant == "0827" else "2026-09-16_14-26-34.onnx")
    original = vendor.original_namespace(source)
    if tuple(original["JOINT_NAMES"]) != core.JOINT_NAMES:
        raise ValueError("Vendor and T1 joint-name order differ")
    # In-process overrides only. Do NOT substitute the user's .25 action scale
    # or posture offsets for the vendor ONNX's learned input/output convention.
    core.KP = original["stiffness_array"].copy()
    core.KD = original["damping_array"].copy()
    core.DEFAULT_JOINT_POS = original["default_joint_pos"].copy()
    core.ACTION_SCALE = original["action_scale"].copy()
    core.START_HEIGHT = original["base_init_height"]
    core.TARGET_LOWER = core.TARGET_UPPER = None
    command, outside_gamepad_range = vendor.checked_command(
        args.command, original["cmd_limit"], original["stand_com_threshold"], args.allow_outside_gamepad_range)
    if outside_gamepad_range:
        print(f"[COMMAND] direct injection {command.tolist()} exceeds delivered gamepad limit {original['cmd_limit'].tolist()}; NOT clipped", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1,840] and session.get_outputs()[0].shape == [1,25]
    engine = SimpleNamespace(policy=session, input_name=session.get_inputs()[0].name,
                             output_name=session.get_outputs()[0].name)
    sim = core.BSRLMujoco(xml, headless=True, command=command, delay_steps=0,
                         actor_include_key_body=False, actor_include_heading_error=False)
    ns = vendor.original_namespace(source, sim.model)
    ns["d"] = sim.data
    ns["reset_robot"]()
    sim.set_action(np.zeros(25))
    model, data = sim.model, sim.data
    np.testing.assert_array_equal(sim.kp, original["stiffness_array"])
    np.testing.assert_array_equal(sim.kd, original["damping_array"])
    mujoco.mj_saveLastXML(str(output / "runtime_model.xml"), model)
    sources = (source, policy, xml, Path(t1.__file__), Path(core.__file__), Path(vendor.__file__), Path(__file__))
    hashes = {str(p): vendor.digest(p) for p in sources}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for p in sources:
        if p.suffix != ".onnx":
            shutil.copy2(p, snapshot / (p.parent.name + "_" + p.name))
    root = slice(0,6)
    manifest = dict(variant=args.variant, environment="user_t1_sim2sim_vendor_gains", policy=str(policy),
        model_xml=str(xml), source_hashes=hashes, runtime_xml_sha256=vendor.digest(output / "runtime_model.xml"),
        command=command.tolist(), seconds=args.seconds, physics_dt=core.PHYSICS_DT, policy_dt=core.POLICY_DT,
        delivered_gamepad_limit=original["cmd_limit"].tolist(),
        outside_delivered_gamepad_range=outside_gamepad_range, command_clipped=False,
        pd_implementation="user runner: implicit affine position PD, 1000Hz; vendor KP/KD unchanged",
        kp=sim.kp.tolist(), kd=sim.kd.tolist(), default_joint_pos=core.DEFAULT_JOINT_POS.tolist(),
        action_scale=core.ACTION_SCALE.tolist(), base_init_height=core.START_HEIGHT,
        contact_friction=core.CONTACT_FRICTION.tolist(), contact_condim=core.CONTACT_CONDIM,
        contact_solref=core.CONTACT_SOLREF.tolist(), contact_solimp=core.CONTACT_SOLIMP.tolist(),
        effort_limits=sim.effort.tolist(), velocity_limits=sim.velocity_limit.tolist(),
        velocity_limit_source=sim.velocity_limit_source, armature=model.dof_armature[sim._qvel_adr].tolist(),
        free_base={k:getattr(model,k)[root].tolist() for k in ("dof_damping","dof_frictionloss","dof_armature")},
        observation="verbatim vendor 84/frame, 10 oldest-first frames, embedded ONNX normalization",
        target_clipping=None, delay_steps=0, yaw_command_compensation=False, automatic_resets=0,
        fall_criteria="root height <0.65m or tilt >60deg; no reset or early termination",
        caveats=["Changes several plant/runtime factors together, not an XML-only causal ablation.",
                 "PD gains are vendor values, but runtime is user's 1ms implicit position PD, not vendor 5ms explicit torque PD.",
                 "Vendor observation offsets, action scaling/defaults, reset pose/height preserved for policy compatibility.",
                 "Ordinary-foot XML chosen, as in the user's September 14/15 T1 recordings, not old four-point default."],
        versions=dict(mujoco=mujoco.__version__,onnxruntime=ort.__version__),videos=[])
    physics = {k:[] for k in ("time","qpos","qvel","ctrl","actuator_force")}
    inference = {k:[] for k in ("time","obs","action","target","command")}
    frames = {k:[] for k in ("time","qpos","qvel","body_velocity","yaw_drift_deg","height","tilt_deg")}
    previous_yaw = unwrapped_yaw = 0.
    first_fall = None
    steps = round(args.seconds / core.PHYSICS_DT)
    assert steps % core.DECIMATION == 0
    print(f"[CONTRACT] {args.variant} user XML; vendor KP/KD; dt={core.PHYSICS_DT}; free_base={manifest['free_base']}",flush=True)
    for step in range(steps):
        if step % core.DECIMATION == 0:
            obs, action = vendor.infer(ns, engine, command)
            sim.set_action(action)
            np.testing.assert_allclose(sim.target, ns["target_dof_pos"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(obs[-84:-81], command*original["command_scale"], atol=1e-7, rtol=0)
            for key,value in zip(inference,(data.time,obs,action,sim.target.copy(),command.copy())):
                inference[key].append(value)
        before = data.time
        sim.step()
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                and math.isclose(data.time-before,core.PHYSICS_DT,abs_tol=1e-9)):
            raise RuntimeError("Nonfinite state or unexpected MuJoCo reset")
        q = data.qpos[3:7]
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        unwrapped_yaw += math.atan2(math.sin(yaw-previous_yaw),math.cos(yaw-previous_yaw))
        previous_yaw=yaw
        height=float(data.qpos[2])
        tilt=math.degrees(math.acos(float(np.clip(-ns["compute_single_obs"]()[-1],-1,1))))
        if first_fall is None and (height<.65 or tilt>60):
            first_fall=float(data.time)
        for key,value in zip(physics,(data.time,data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),data.actuator_force.copy())):
            physics[key].append(value)
        if (step+1)%core.DECIMATION==0:
            body_vel=ns["quat_rotate_inverse_wxyz"](q,data.qvel[:3])
            for key,value in zip(frames,(data.time,data.qpos.copy(),data.qvel.copy(),body_vel,math.degrees(unwrapped_yaw),height,tilt)):
                frames[key].append(value)
        if (step+1)%5000==0:
            print(f"[SIM] {args.variant} T1 t={data.time:.2f}s yaw={math.degrees(unwrapped_yaw):+.2f} fall={first_fall}",flush=True)
    physics,inference,frames=[{k:np.asarray(v) for k,v in group.items()} for group in (physics,inference,frames)]
    for name,group in (("physics",physics),("policy",inference),("frames",frames)):
        assert all(np.isfinite(v).all() for v in group.values())
        np.savez_compressed(output / f"{name}.npz",**group)
    manifest.update(simulation_seconds=float(data.time),fps=50,frames_per_view=len(frames["time"]),
        first_fall_time=first_fall,final_yaw_drift_deg=float(frames["yaw_drift_deg"][-1]),
        displacement_xy=data.qpos[:2].tolist(),mean_body_vx=float(frames["body_velocity"][:,0].mean()),
        mean_body_vx_after_5s=(float(frames["body_velocity"][frames["time"]>=5,0].mean())
                               if np.any(frames["time"]>=5) else None))
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(f"[RESULT] {args.variant} T1 vx={manifest['mean_body_vx']:.4f} yaw={manifest['final_yaw_drift_deg']:.2f} fall={first_fall}",flush=True)
    if args.no_video:
        return
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,1280)
    model.vis.global_.offheight=max(model.vis.global_.offheight,720)
    renderer=mujoco.Renderer(model,height=720,width=1280)
    camera=mujoco.MjvCamera()
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
    try:
        for view in ("whole","feet"):
            path=output / f"door_{args.variant}_user_t1_{vendor.command_tag(command)}_{view}.mp4"
            camera.azimuth=130
            camera.elevation=-15 if view=="whole" else -10
            camera.distance=3.7 if view=="whole" else 1.55
            with imageio.get_writer(str(path),fps=50,codec="libx264",quality=8,macro_block_size=1,
                    ffmpeg_log_level="error",output_params=["-threads","2","-movflags","+faststart"]) as writer:
                for i,t in enumerate(frames["time"]):
                    data.qpos[:]=frames["qpos"][i];data.qvel[:]=frames["qvel"][i];data.time=t
                    mujoco.mj_forward(model,data)
                    camera.lookat[:]=[data.qpos[0],data.qpos[1],.85 if view=="whole" else .25]
                    renderer.update_scene(data,camera=camera)
                    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                    draw.rectangle((8,8,1272,98),fill=(18,22,28))
                    lines=[f"DOOR {args.variant} ONNX | USER T1 XML / runtime | VENDOR KP/KD | {view}",
                           f"cmd=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}) | NO heading compensation / NO resets | t={t:.2f}s",
                           f"body vx={frames['body_velocity'][i,0]:+.3f} m/s | yaw={frames['yaw_drift_deg'][i]:+.1f} deg | fall={first_fall if first_fall is not None and first_fall<=t else None}"]
                    for row,line in enumerate(lines):
                        draw.text((18,12+row*28),line,font=font,fill="white")
                    writer.append_data(np.asarray(frame))
                    if i in (0,249,749,len(frames["time"])-1):
                        frame.save(output / f"preview_{view}_{i:04d}.png")
                    if (i+1)%500==0:
                        print(f"[VIDEO] {args.variant} T1 {view} {i+1}/{len(frames['time'])}",flush=True)
            manifest["videos"].append(str(path))
    finally:
        renderer.close();sim.close()
    for path,old in hashes.items():
        assert vendor.digest(path)==old, f"Source changed during replay: {path}"
    (output / "manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(f"[COMPLETE] {output}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",choices=("0827","0916"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=30.)
    parser.add_argument("--command",type=float,nargs=3,default=[.6,0.,0.])
    parser.add_argument("--allow-outside-gamepad-range",action="store_true",
                        help="Explicitly inject commands beyond the original joystick limit; no clipping")
    parser.add_argument("--no-video",action="store_true")
    args=parser.parse_args()
    if args.output.exists() or not np.isfinite(args.seconds) or args.seconds<=0:
        parser.error("Require new output directory and positive finite duration")
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID","0")
    record(args)
