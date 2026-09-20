"""Calibrated standalone PR/AB demonstration; no encoder re-calibration."""
import argparse
import json
import time

import mujoco
import numpy as np

if __package__:
    from .real_ankle import Geometry, RobotCore, closure_gaps, standalone_xml
else:
    from real_ankle import Geometry, RobotCore, closure_gaps, standalone_xml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("right","left"), default="right")
    parser.add_argument("--mode", choices=("PR","AB"), default="PR")
    parser.add_argument("--no-view", action="store_true")
    args=parser.parse_args()
    core=RobotCore(); geometry=Geometry(args.side,core)
    model=mujoco.MjModel.from_xml_string(standalone_xml(geometry)); data=mujoco.MjData(model)
    geometry.initialize(model,data,np.zeros(2),""); mujoco.mj_forward(model,data)
    viewer=None
    if not args.no_view:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(model,data)
        viewer.cam.lookat[:]=[0,0,.6]; viewer.cam.distance=1.1
    extremes=np.deg2rad([[0,0],[34,0],[-34,0],[0,29],[0,-29],[34,29],[34,-29],[-34,29],[-34,-29]])
    previous=np.zeros(2); errors=[]; gaps=[]; conditions=[]
    try:
        for step in range(18000):
            start=time.perf_counter(); t=step*.001
            target=extremes[min(int(t/2),len(extremes)-1)]*np.sin(np.pi*(t%2)/2)**2
            motors=np.array([data.joint(f"motor_{i}").qpos[0] for i in (1,2)])
            motor_speed=np.array([data.joint(f"motor_{i}").qvel[0] for i in (1,2)])
            q=geometry.inverse(motors,previous); previous=q
            _,jac=geometry.forward(q)
            if args.mode=="PR":
                torque=np.linalg.solve(jac.T,120*(target-q)-2*np.linalg.solve(jac,motor_speed))
            else:
                target_motor,_=geometry.forward(target)
                torque=120*(target_motor-motors)-2*motor_speed
            data.ctrl[:]=np.clip(torque,-20,20)
            actual=np.array([data.joint(n).qpos[0] for n in ("ankle_pitch","ankle_roll")])
            errors.append(q-actual); gaps.append(closure_gaps(model,data,[""])); conditions.append(np.linalg.cond(jac))
            mujoco.mj_step(model,data)
            if viewer:
                if not viewer.is_running(): break
                if step%20==0: viewer.sync()
                time.sleep(max(0,.001-(time.perf_counter()-start)))
        print(json.dumps(dict(side=args.side,mode=args.mode,seconds=data.time,
                             fk_error_max_rad=np.max(np.abs(errors),axis=0).tolist(),
                             closure_gap_max_m=np.max(gaps,axis=0).tolist(),
                             max_condition=max(conditions)),indent=2))
    finally:
        if viewer: viewer.close()


if __name__=="__main__": main()
