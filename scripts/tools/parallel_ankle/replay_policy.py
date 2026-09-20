"""Replay the unchanged BSRL ONNX in serial / real-geometry closed-chain physics.

Dedicated diagnostic: does not change mjw_simtool defaults, AMP assets or runs.
The actor still commands 12 joint positions. Only four ankle drives are replaced.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

if __package__:
    from .build_models import SERIAL_XML, full_robot_xml
    from .real_ankle import CALIBRATION, CORE, HERE, WORKSPACE, Geometry, RobotCore, closure_gaps
else:
    from build_models import SERIAL_XML, full_robot_xml
    from real_ankle import CALIBRATION, CORE, HERE, WORKSPACE, Geometry, RobotCore, closure_gaps

spec = importlib.util.spec_from_file_location(
    "bsrl_observation_contract", WORKSPACE / "mjw_simtool/scripts/sim2simBSRL/run_sim2sim_standalone.py")
obs_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obs_contract)
JOINTS = obs_contract.JOINT_NAMES
BASELINE_RUN = WORKSPACE / "Bipedal_AMP/logs/rsl_rl/bsrl_amp/2026-09-07_14-49-15_noestimator_mirror01_friction0515_formal1_8192"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_nominal(run):
    with (run / "params/env.yaml").open() as stream:
        cfg = yaml.load(stream, Loader=yaml.BaseLoader)
    with (run / "params/agent.yaml").open() as stream:
        agent = yaml.load(stream, Loader=yaml.BaseLoader)
    if cfg["observations"]["state_est_label"] != "null":
        raise ValueError("This diagnostic currently requires the no-estimator baseline")
    obs_names = tuple(cfg["observations"]["policy"]["joint_pos"]["params"]["asset_cfg"]["joint_names"])
    if obs_names != JOINTS:
        raise ValueError("Policy joint order differs from the verified BSRL layout")
    fields = ("stiffness", "damping", "armature", "effort_limit", "velocity_limit", "peak_power")
    values = {key: np.empty(12) for key in fields}
    robot = cfg["scene"]["robot"]
    for i, name in enumerate(JOINTS):
        matches = [a for a in robot["actuators"].values() if any(re.fullmatch(p, name) for p in a["joint_names_expr"])]
        if len(matches) != 1: raise ValueError(f"Ambiguous actuator: {name}")
        actuator = matches[0]
        reduced = actuator["class_type"].endswith(":ReducedAnkleActuator")
        closed = actuator['class_type'].endswith(':ClosedAnkleActuator')
        mapped_motor = reduced or closed
        if mapped_motor and "ankle" not in name:
            raise ValueError("Reduced transmission must only control ankles")
        if not mapped_motor and not actuator["class_type"].endswith((":DelayedPowerLimitedPDActuator",
                                                                    ':SubsteppedPowerLimitedPDActuator')):
            raise ValueError(f"Unsupported actuator contract: {actuator['class_type']}")
        for field in fields:
            # ReducedAnkleActuator's effort_limit=1e9 / velocity_limit=1000
            # are numerical joint guards, NOT the physical motor envelope.
            source = {"effort_limit": "motor_effort", "velocity_limit": "motor_speed",
                      "peak_power": "motor_power"}.get(field, field) if mapped_motor else field
            value = actuator[source]
            if isinstance(value, dict):
                found = [v for p,v in value.items() if re.fullmatch(p,name)]
                if len(found)!=1: raise ValueError(f"Ambiguous {field}: {name}")
                value = found[0]
            values[field][i] = float(value)
    values["default_pos"] = np.array([float(next(v for p,v in robot["init_state"]["joint_pos"].items() if re.fullmatch(p,n))) for n in JOINTS])
    values["action_scale"] = np.array([float(next(v for p,v in cfg["actions"]["joint_pos"]["scale"].items() if re.fullmatch(p,n))) for n in JOINTS])
    values["start_height"] = float(robot["init_state"]["pos"][2])
    values["policy_dt"] = float(cfg["sim"]["dt"]) * int(cfg["decimation"])
    values['training_physics_dt'] = float(cfg['sim']['dt'])
    values['training_actuator_types'] = [a['class_type'] for a in robot['actuators'].values()]
    assert cfg["actions"]["joint_pos"]["use_default_offset"] == "true"
    clip = cfg["actions"]["joint_pos"]["clip"]
    values["target_lower"] = values["target_upper"] = None
    if clip != "null":
        bounds = []
        for name in JOINTS:
            found = [v for pattern, v in clip.items() if re.fullmatch(pattern, name)]
            if len(found) != 1:
                raise ValueError(f"Ambiguous action clipping: {name}")
            bounds.append([float(v) for v in found[0]])
        bounds = np.asarray(bounds)
        if bounds.shape != (12, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Invalid action clipping bounds")
        values["target_lower"], values["target_upper"] = bounds.T
    return values, agent


def action_target(nominal, action):
    target = nominal["default_pos"] + action * nominal["action_scale"]
    if nominal["target_lower"] is not None:
        target = np.clip(target, nominal["target_lower"], nominal["target_upper"])
    return target


def clip_motor(torque, velocity, effort, speed, power):
    magnitude = np.minimum(effort, power / np.maximum(np.abs(velocity), 1e-6))
    lower = np.where(velocity <= -speed, 0., -magnitude)
    upper = np.where(velocity >= speed, 0., magnitude)
    return np.clip(torque, lower, upper)


def recovered_model_xml(path, auxiliary_mass, equality_timeconst):
    """Load an explicitly selected, integrity-checked CAD recovery model."""
    path = Path(path).resolve()
    provenance = json.loads((path.parent / "parallel_ankle_recovery.json").read_text())
    if provenance["geometry_revision"] != Geometry.revision:
        raise ValueError("Recovered model geometry revision differs")
    if sha(path) != provenance["xml_sha256"].get(path.name):
        raise ValueError("Recovered XML checksum differs from its recovery manifest")
    for relative, source in provenance["restored_assets"].items():
        if sha(path.parent.parent / relative) != source["sha256"]:
            raise ValueError(f"Recovered mesh checksum differs: {relative}")
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir", "")).resolve()))
    constraints = root.findall("./equality/connect")
    if len(constraints) != 4:
        raise ValueError("Expected four explicit ankle closure constraints")
    for constraint in constraints:
        np.testing.assert_allclose(float(constraint.get("solref").split()[0]), equality_timeconst, rtol=0, atol=1e-15)
    for body in root.findall(".//body"):
        if body.get("name", "").startswith(("crank_", "rod_")):
            np.testing.assert_allclose(float(body.find("geom").get("mass")), auxiliary_mass, rtol=0, atol=1e-15)
    return ET.tostring(root, encoding="unicode")


def policy_provenance(run, policy, metadata, checkpoint=None):
    """Do not associate an explicitly selected ONNX with an unrelated checkpoint."""
    checkpoint = Path(checkpoint or metadata.get("checkpoint", run / "model_29999.pt")).resolve()
    if checkpoint.parent != run.resolve():
        raise ValueError("Checkpoint and nominal training run differ")
    checkpoint_hash = sha(checkpoint)
    if metadata.get("checkpoint_sha256") and checkpoint_hash != metadata["checkpoint_sha256"]:
        raise ValueError("ONNX metadata and checkpoint checksum differ")
    if metadata.get("run") and Path(metadata["run"]).name != run.name:
        raise ValueError("ONNX metadata and nominal training run differ")
    return dict(checkpoint=str(checkpoint), checkpoint_label=checkpoint.stem,
                checkpoint_sha256=checkpoint_hash, policy=str(policy.resolve()),
                policy_sha256=sha(policy), onnx_metadata=metadata)


class Replay:
    # Reuse the already-tested term-major actor convention, not a second
    # observation implementation.  Legacy actors use 5 frames (240D), while
    # latent actors use the same 48D frame with 10 frames (480D).
    _state = obs_contract.BSRLMujoco._state
    _observation_frame = obs_contract.BSRLMujoco._observation_frame
    observation = obs_contract.BSRLMujoco.observation

    def __init__(self, run, topology, core, command, auxiliary_mass=.001,
                 equality_timeconst=.002, ankle_limit=None, *, model_xml=None,
                 history_length=5):
        self.nominal, self.agent_cfg = load_nominal(run)
        self.core, self.topology = core, topology
        if model_xml is not None:
            if topology != "closed" or core is not None:
                raise ValueError("Explicit recovered XML requires closed topology and CAD-only geometry")
            self.geometry = {side: Geometry.from_cad(side) for side in ("right", "left")}
            self.xml = recovered_model_xml(model_xml, auxiliary_mass, equality_timeconst)
        else:
            self.geometry = {side: Geometry(side,core) for side in ("right","left")}
        if model_xml is None and topology == "closed":
            self.xml = full_robot_xml(core, auxiliary_mass=auxiliary_mass, equality_timeconst=equality_timeconst)
        elif model_xml is None:
            self.xml = obs_contract._make_runtime_xml(SERIAL_XML)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        m = self.model
        m.opt.timestep=.001
        m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.opt.iterations=100; m.opt.tolerance=1e-10
        self.data=mujoco.MjData(m)
        self._qpos_adr=np.array([m.joint(n).qposadr[0] for n in JOINTS])
        self._qvel_adr=np.array([m.joint(n).dofadr[0] for n in JOINTS])
        self._body_ids=np.array([m.body(n).id for n in obs_contract.KEY_BODY_NAMES])
        self.actor_include_key_body=False; self.actor_include_heading_error=False
        self.action=np.zeros(12); self.command=np.asarray(command, dtype=float)
        self.target=self.nominal["default_pos"].copy()
        if history_length not in (5, 10):
            raise ValueError("BSRL replay history must be 5 or 10 frames")
        self.history_length=history_length
        self._history=deque(maxlen=history_length)
        self.ankle_indices={s: np.array([JOINTS.index(f"joint_{s}_ankle_{a}") for a in ("pitch","roll")]) for s in self.geometry}
        self.direct_ids={}
        for i,name in enumerate(JOINTS):
            jid=m.joint(name).id
            m.jnt_solref[jid]=obs_contract.JOINT_LIMIT_SOLREF
            m.jnt_solimp[jid]=obs_contract.JOINT_LIMIT_SOLIMP
            m.jnt_margin[jid]=0
            m.dof_armature[self._qvel_adr[i]]=self.nominal["armature"][i]
            m.dof_damping[self._qvel_adr[i]]=0
            m.jnt_actfrclimited[jid]=False
            matches=np.flatnonzero((m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)&(m.actuator_trnid[:,0]==jid))
            if topology=="closed" and "ankle" in name:
                assert len(matches)==0, "A passive ankle must not also have a direct actuator"
            else:
                assert len(matches)==1
                self.direct_ids[i]=int(matches[0])
        # Keep exactly the existing sim2sim contact and joint-limit settings in BOTH models.
        geoms={m.geom("floor").id}
        for body_name in obs_contract.CONTACT_BODY_NAMES:
            bid=m.body(body_name).id
            geoms.update(g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and (m.geom_contype[g] or m.geom_conaffinity[g]))
        for gid in geoms:
            m.geom_solref[gid]=obs_contract.CONTACT_SOLREF
            m.geom_solimp[gid]=obs_contract.CONTACT_SOLIMP
            m.geom_friction[gid]=obs_contract.CONTACT_FRICTION
            m.geom_condim[gid]=obs_contract.CONTACT_CONDIM
        for aid in range(m.nu):
            m.actuator_gaintype[aid]=mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[aid]=mujoco.mjtBias.mjBIAS_NONE
            m.actuator_gainprm[aid]=0; m.actuator_gainprm[aid,0]=1
            m.actuator_biasprm[aid]=0
            m.actuator_ctrllimited[aid]=False
            m.actuator_forcelimited[aid]=False  # Explicit motor envelope below.
        self.motor_ids={s: np.array([m.actuator(f"drive_{s}_{i}").id for i in (1,2)]) for s in self.geometry} if topology=="closed" else {}
        self.ankle_limit=ankle_limit
        mujoco.mj_setConst(m,self.data)
        self.reset()

    def reset(self):
        d,m=self.data,self.model
        mujoco.mj_resetData(m,d)
        d.qpos[:3]=[0,0,self.nominal["start_height"]]
        d.qpos[3:7]=[1,0,0,0]
        d.qpos[self._qpos_adr]=self.nominal["default_pos"]
        if self.topology=="closed":
            for side,g in self.geometry.items():
                g.initialize(m,d,self.nominal["default_pos"][self.ankle_indices[side]],f"{side}_")
        mujoco.mj_forward(m,d)
        self.initial_gaps=closure_gaps(m,d,["right_","left_"]) if self.topology=="closed" else np.zeros(4)
        assert self.initial_gaps.max()<1e-10
        self.action[:]=0; self.target[:]=self.nominal["default_pos"]
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(self._observation_frame().copy())

    def apply(self):
        q,dq,_=self._state(); n=self.nominal
        requested=n["stiffness"]*(self.target-q)-n["damping"]*dq
        direct=clip_motor(requested,dq,n["effort_limit"],n["velocity_limit"],n["peak_power"])
        for i,aid in self.direct_ids.items(): self.data.ctrl[aid]=direct[i]
        motor_requested=[]; motor_applied=[]; motor_speeds=[]; jacs=[]; angles=[]
        for side,ix in self.ankle_indices.items():
            angle,jac=self.geometry[side].forward(q[ix])
            req=np.linalg.solve(jac.T,requested[ix])
            if self.topology=="closed":
                speed=np.array([self.data.joint(f"motor_{side}_{i}").qvel[0] for i in (1,2)])
            else:
                speed=jac@dq[ix]
            limit=n["effort_limit"][ix] if self.ankle_limit is None else self.ankle_limit
            applied=clip_motor(req,speed,limit,n["velocity_limit"][ix],n["peak_power"][ix])
            if self.topology=="closed":
                self.data.ctrl[self.motor_ids[side]]=applied
            elif self.topology=="mapped":
                converted=jac.T@applied
                for index,value in zip(ix,converted): self.data.ctrl[self.direct_ids[index]]=value
            else:
                # Serial baseline has independent joint limits, not joint-to-motor saturation.
                applied=np.linalg.solve(jac.T,direct[ix])
            motor_requested.extend(req); motor_applied.extend(applied); motor_speeds.extend(speed)
            jacs.append(jac); angles.extend(angle)
        # No direct qfrc_applied and no post-integration velocity clamping.
        assert not np.any(self.data.qfrc_applied)
        return requested,np.array(motor_requested),np.array(motor_applied),np.array(motor_speeds),np.array(jacs),np.array(angles)


def policy_history_length(input_shape, output_shape):
    """Validate the deployable BSRL contract and return its history length."""
    if list(output_shape) != [1, 12]:
        raise ValueError(f"Expected BSRL policy output [1, 12], got {output_shape}")
    if len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] not in (240, 480):
        raise ValueError(f"Expected BSRL policy input [1, 240] or [1, 480], got {input_shape}")
    return int(input_shape[1]) // 48


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--onnx", type=Path, help="Use this exact ONNX instead of the run's default export")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for ONNX provenance verification")
    parser.add_argument("--model-xml", type=Path,
                        help="Explicit CAD-recovered closed XML; verifies its manifest, not missing hardware C++")
    parser.add_argument("--topology", choices=("serial","mapped","closed"), default="closed")
    parser.add_argument("--command", nargs=3,type=float,default=(.6,0,0))
    parser.add_argument("--seconds",type=float,default=30)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--video",type=Path)
    parser.add_argument('--label',type=str,help='Experiment label embedded in saved replay and videos')
    parser.add_argument("--auxiliary-mass",type=float,default=.001)
    parser.add_argument("--equality-timeconst",type=float,default=.002)
    parser.add_argument("--ankle-limit",type=float)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.video and args.video.exists(): raise FileExistsError(args.video)
    if args.seconds<=0 or args.auxiliary_mass<=0 or args.equality_timeconst<=0:
        raise ValueError("Positive duration/mass/constraint time constant required")
    if args.ankle_limit is not None and args.ankle_limit<=0:
        raise ValueError("Ankle limit must be positive")
    if args.model_xml is not None and args.topology != "closed":
        parser.error("--model-xml requires --topology closed")
    policy=(args.onnx or args.run/"exported/policy.onnx").resolve()
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(policy),opts,providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected one ONNX input and one ONNX output")
    history_length=policy_history_length(session.get_inputs()[0].shape, session.get_outputs()[0].shape)
    metadata=session.get_modelmeta().custom_metadata_map
    if args.onnx and not metadata.get("checkpoint_sha256"):
        raise ValueError("Explicit ONNX must include its checkpoint SHA256 metadata")
    provenance=policy_provenance(args.run,policy,metadata,args.checkpoint)
    core=None if args.model_xml else RobotCore()
    sim=Replay(args.run,args.topology,core,args.command,args.auxiliary_mass,args.equality_timeconst,args.ankle_limit,
               model_xml=args.model_xml,history_length=history_length)
    args.output.mkdir(parents=True)
    decimation=round(sim.nominal["policy_dt"]/sim.model.opt.timestep)
    renderer=writer=None
    rows={key:[] for key in ("time_s","qpos","qvel","obs","action","target","command","base_velocity_b")}
    physics={key:[] for key in ("time_s","joint_pos","joint_vel","requested_joint_torque","motor_requested","motor_applied","motor_velocity","closure_gap","motor_angle_error","control","jacobian")}
    start=time.monotonic(); stop=None; failure=None; video_frames=0
    manifest=dict(run=str(args.run.resolve()),**provenance,
                  experiment_label=args.label or 'BSRL',
                  calibration_sha256=sha(CALIBRATION) if core is not None else None,
                  cpp_sha256=sha(CORE/"AnkleKinematics.cpp") if core is not None else None,
                  hardware_cpp_calibration_verified=core is not None,
                  recovered_model_xml=str(args.model_xml.resolve()) if args.model_xml else None,
                  recovered_model_sha256=sha(args.model_xml) if args.model_xml else None,
                  serial_xml_sha256=sha(SERIAL_XML), automatic_resets=0,
                  topology=args.topology,command=args.command,physics_dt=.001,pd_dt=.001,policy_dt=sim.nominal["policy_dt"],
                  actor_input_dim=48*history_length,actor_history_length=history_length,
                  evaluation_conditions='nominal deployment: no noise, DR, delay, push, or automatic reset',
                  training_physics_dt=sim.nominal['training_physics_dt'],
                  auxiliary_mass_per_body_kg=args.auxiliary_mass,unmeasured_auxiliary_inertia=True,
                  motor_armature_added=0,equality_timeconst=args.equality_timeconst,
                  ankle_limit_nm=args.ankle_limit or 112,total_mass_kg=float(sim.model.body_mass.sum()),
                  motor_order=["right_long", "right_short", "left_long", "left_short"],
                  geometry_source="user-confirmed original CAD zero-pose connection points",
                  geometry={side:dict(mounts_m=g.mounts.tolist(),pivots_m=g.pivots.tolist(),
                                     crank_lengths_m=g.l3.tolist(),rod_lengths_m=g.l4.tolist())
                            for side,g in sim.geometry.items()},
                  solver="generalized rigid-rod closure and analytic constraint Jacobian in URDF frame",
                  mounting_revision=Geometry.revision,
                  hardware_solver_modified=False,
                  nominal={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sim.nominal.items()},
                  init_closure_gap_m=sim.initial_gaps.tolist(),encoder_recalibration=False,
                  feedback="simulated pitch/roll joints; FK consistency separately checked",
                  passive_ankle_direct_actuators=0 if args.topology=="closed" else 4)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output/"model.xml").write_text(sim.xml+"\n")
    print(json.dumps(dict(starting=provenance["checkpoint_label"], policy=str(policy),
                          recovered_xml=manifest["recovered_model_xml"], seconds=args.seconds)), flush=True)
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True,exist_ok=True)
            sim.model.vis.global_.offwidth=1280; sim.model.vis.global_.offheight=720
            renderer=mujoco.Renderer(sim.model,height=720,width=1280)
            writer=imageio.get_writer(str(args.video),fps=50,codec="libx264",quality=8)
            camera=mujoco.MjvCamera(); camera.distance=3.; camera.azimuth=125; camera.elevation=-18
        for step in range(round(args.seconds/.001)):
            if step and step % 5000 == 0:
                print(f"Simulated {sim.data.time:.1f}/{args.seconds:.1f} s", flush=True)
            mujoco.mj_forward(sim.model,sim.data)
            if step%decimation==0:
                obs=sim.observation()
                action=session.run(None,{session.get_inputs()[0].name:obs})[0].reshape(12)
                assert np.isfinite(obs).all() and np.isfinite(action).all()
                sim.action[:]=action
                sim.target[:]=action_target(sim.nominal, action)
                base_v=np.zeros(6)
                mujoco.mj_objectVelocity(sim.model,sim.data,mujoco.mjtObj.mjOBJ_BODY,sim.model.body("base_link").id,base_v,1)
                values=(sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),obs[0],action.copy(),sim.target.copy(),sim.command.copy(),base_v)
                for key,value in zip(rows,values): rows[key].append(value)
                if renderer:
                    camera.lookat[:]=sim.data.xpos[sim.model.body("base_link").id]
                    renderer.update_scene(sim.data,camera=camera)
                    writer.append_data(renderer.render()); video_frames+=1
            requested,mreq,mapplied,mvel,jac,mangles=sim.apply()
            q,dq,_=sim._state()
            gap=np.zeros(4); angle_error=np.zeros(4)
            if args.topology=="closed":
                gap=closure_gaps(sim.model,sim.data,["right_","left_"])
                actual=np.array([sim.data.joint(f"motor_{s}_{i}").qpos[0] for s in ("right","left") for i in (1,2)])
                angle_error=actual-mangles
                if gap.max()>.002: raise RuntimeError(f"Closure gap exceeds 2 mm: {gap}")
            values=(sim.data.time,q,dq,requested,mreq,mapplied,mvel,gap,angle_error,sim.data.ctrl.copy(),jac)
            for key,value in zip(physics,values): physics[key].append(value)
            mujoco.mj_step(sim.model,sim.data)
            if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                raise RuntimeError("Non-finite MuJoCo state")
            rotation=obs_contract._quat_to_mat_wxyz(sim.data.qpos[3:7])
            if sim.data.qpos[2]<.5 or rotation[2,2]<.5:
                stop="base_height" if sim.data.qpos[2]<.5 else "bad_orientation"
                break
    except Exception as error:
        failure=f"{type(error).__name__}: {error}"
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    rows={k:np.asarray(v) for k,v in rows.items()}; physics={k:np.asarray(v) for k,v in physics.items()}
    np.savez_compressed(args.output/"policy.npz",**rows)
    np.savez_compressed(args.output/"physics.npz",**physics)
    summary=dict(topology=args.topology,seconds=float(sim.data.time),fall=stop,failure=failure,
                 video_frames=video_frames,wall_seconds=time.monotonic()-start,
                 final_base_pos=sim.data.qpos[:3].tolist(),
                 warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(sim.data.warning) if w.number})
    if len(physics["time_s"]):
        summary.update(closure_max_m=physics["closure_gap"].max(axis=0).tolist(),
                       motor_angle_error_max_rad=np.abs(physics["motor_angle_error"]).max(axis=0).tolist(),
                       motor_torque_peak_nm=np.abs(physics["motor_applied"]).max(axis=0).tolist(),
                       motor_torque_rms_nm=np.sqrt(np.mean(physics["motor_applied"]**2,axis=0)).tolist(),
                       motor_saturation_fraction=np.mean(np.abs(physics["motor_requested"]-physics["motor_applied"])>1e-5,axis=0).tolist(),
                       max_j_condition=float(np.linalg.cond(physics["jacobian"]).max()),
                       mean_body_vx=float(rows["base_velocity_b"][:,3].mean()))
    if len(rows["qpos"]):
        quaternions=rows["qpos"][:,3:7]
        w,x,y,z=quaternions.T
        yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        summary.update(policy_frames=len(rows["time_s"]), physics_steps=len(physics["time_s"]),
                       final_yaw_degrees=float(np.rad2deg(yaw[-1]-yaw[0])),
                       min_base_height=float(rows["qpos"][:,2].min()), automatic_resets=0)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    assert sha(policy)==manifest["policy_sha256"], "Policy changed during replay"
    if failure: raise RuntimeError(failure)


if __name__=="__main__": main()
