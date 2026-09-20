"""Validate completed recordings without touching the running trainers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    import imageio_ffmpeg
    import numpy as np
    import onnxruntime as ort
    from t1_heading_ablation import load_saved_config, configuration_differences
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    selection = json.loads((output/'selection.json').read_text())
    validated = []
    for row in selection['recordings']:
        folders = [p.parent for p in output.glob(f"{row['key']}_model{row['iteration']}_*/manifest.json")]
        if not folders and args.completed_only:
            continue
        if len(folders) != 1:
            raise ValueError(f"Missing/ambiguous completed recording: {row['key']}")
        folder = folders[0]
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['command'] == selection['command'] and manifest['nominal_eval']
        assert manifest['checkpoint_sha256'] == row['checkpoint_sha256']
        assert manifest['checkpoint'] == row['checkpoint']
        assert manifest['iteration'] == row['iteration']
        assert hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest() == row['checkpoint_sha256']
        assert manifest['automatic_resets'] == 0 and not manifest['yaw_compensation']
        assert manifest['pd_profile'] == 'nominal'
        saved = load_saved_config(Path(row['checkpoint']).parent/'params/env.yaml')
        actual = load_saved_config(folder/'effective_env.yaml')
        robot_expected = saved['scene']['robot']
        robot_actual = actual['scene']['robot']
        diffs = configuration_differences(robot_expected, robot_actual)
        # Only simulation output paths and nominal (zero) command delays change.
        allowed = {'prim_path', 'spawn.usd_dir'}
        assert all(d['path'] in allowed or d['path'].endswith(('.min_delay','.max_delay')) for d in diffs), diffs
        assert actual['actions'] == saved['actions']
        assert actual['sim']['dt'] == saved['sim']['dt']
        assert actual['decimation'] == saved['decimation']
        assert manifest['physics_dt'] == saved['sim']['dt']
        for key, motor in manifest['actual_controller']['actuators'].items():
            for field, gain in [('stiffness','kp'),('damping','kd')]:
                value = robot_expected['actuators'][key][field]
                expected = [next(v for pattern,v in value.items() if re.fullmatch(pattern,joint))
                            if isinstance(value,dict) else value for joint in motor['joint_names']]
                np.testing.assert_allclose(motor[gain],expected,atol=1e-5,rtol=1e-6)
        data = np.load(folder/'trajectory.npz')
        frames = round(selection['seconds']/manifest['policy_dt'])
        assert len(data['time']) == manifest['frames_per_view'] == frames
        assert all(np.isfinite(data[key]).all() for key in data.files)
        np.testing.assert_allclose(data['time'],np.arange(1,frames+1)*manifest['policy_dt'],atol=1e-7)
        assert np.all(data['command'] == selection['command'])
        opts=ort.SessionOptions(); opts.intra_op_num_threads=opts.inter_op_num_threads=1
        engine=ort.InferenceSession(str(folder/'export/policy.onnx'),opts,providers=['CPUExecutionProvider'])
        max_error=0.
        for index in np.linspace(0,frames-1,100,dtype=int):
            action=engine.run(None,{engine.get_inputs()[0].name:data['obs'][index:index+1]})[0][0]
            max_error=max(max_error,float(np.abs(action-data['action'][index]).max()))
            np.testing.assert_allclose(action,data['action'][index],atol=1e-6,rtol=1e-6)
        videos=[]
        for video in sorted(folder.glob('*.mp4')):
            reader=imageio_ffmpeg.read_frames(str(video)); meta=next(reader); reader.close()
            count,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
            assert count==frames and abs(seconds-selection['seconds'])<1e-3
            assert meta['size']==(1280,720) and meta['fps']==50
            process=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2',
                '-i',str(video),'-f','null','-'],capture_output=True,text=True,check=True)
            if process.stderr.strip():
                raise RuntimeError(process.stderr)
            videos.append(dict(path=str(video),frames=count,seconds=seconds,decode_ok=True))
        assert len(videos)==2
        w,x,y,z=data['root_quat'].astype(np.float64).T
        yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        initial_yaw=yaw[-1]-np.radians(manifest['final_yaw_drift_degrees'])
        dx,dy=manifest['displacement_xy']
        valid=np.ones(frames,dtype=bool) if manifest['first_fall_time'] is None else data['time']<manifest['first_fall_time']
        metric=dict(key=row['key'],iteration=row['iteration'],mean_body_vx=manifest['mean_body_vx'],
            final_yaw_drift_degrees=manifest['final_yaw_drift_degrees'],first_fall_time=manifest['first_fall_time'],
            max_abs_yaw_drift_degrees=float(np.max(np.abs(np.degrees(yaw-initial_yaw)))),
            displacement_initial_heading_xy=[float(dx*np.cos(initial_yaw)+dy*np.sin(initial_yaw)),
                                             float(-dx*np.sin(initial_yaw)+dy*np.cos(initial_yaw))],
            valid_seconds=float(valid.sum()*manifest['policy_dt']),
            vx_rmse=float(np.sqrt(np.mean((data['root_lin_vel_b'][valid,0]-selection['command'][0])**2))) if valid.any() else None,
            saved_controller_verified=True,trajectory_finite=True,onnx_replay_max_error=max_error,videos=videos)
        validated.append(metric)
        print(json.dumps(metric),flush=True)
    filename='validation_partial.json' if args.completed_only else 'validation.json'
    (output/filename).write_text(json.dumps(validated,indent=2)+"\n")


if __name__ == '__main__':
    main()
