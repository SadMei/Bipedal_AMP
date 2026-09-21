"""Freeze and record the seven recent BSRL AMP comparisons; never touches training.

CPU ONNX export and closed-chain simulation, GPU offline rendering. Falls and
failed rollouts remain visible. No automatic resets or overwrite of old output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

from ._paths import CLOSED_ANKLE_ROOT, REPO_ROOT

RUNS = [
    ('serial_mirror','2026-09-14_17-51-40_','Serial / mirror 0.1'),
    ('serial_nomirror','2026-09-15_19-55-26_','Serial / no mirror'),
    ('reduced_mirror','2026-09-15_15-14-55_','Reduced parallel / mirror 0.1'),
    ('reduced_heading','2026-09-16_10-51-46_','Reduced + heading / mirror 0.1'),
    ('explicit_closed','2026-09-15_16-30-14_','Explicit closed training / mirror 0.1'),
    ('accad_nomirror','2026-09-16_21-06-48_','ACCAD34 / no mirror'),
    ('accad_mirror','2026-09-16_21-07-06_','ACCAD34 / mirror 0.1'),
]


def save_json(path,value):
    content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with Path(path).open('x') as stream:stream.write(content)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run_command(arguments,log,extra_env=None):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.update(extra_env or {})
    with log.open('x') as stream:
        process=subprocess.run([sys.executable,'-m',*arguments],cwd=REPO_ROOT,
                               env=env,stdout=stream,stderr=subprocess.STDOUT)
    return process.returncode


def simulate(record,output):
    folder=output/record['label'];folder.mkdir()
    rc=run_command(['scripts.tools.parallel_ankle.export_checkpoint',record['checkpoint'],str(folder/'export')],folder/'export.log')
    if rc:
        return dict(label=record['label'],stage='export_failed',returncode=rc)
    rc=run_command(['scripts.tools.parallel_ankle.replay_policy','--run',record['run'],
        '--onnx',str(folder/'export/policy.onnx'),'--checkpoint',record['checkpoint'],
        '--model-xml',str(CLOSED_ANKLE_ROOT/'xml/bsrl_closed_ankles.xml'),
        '--topology','closed','--command','0.6','0','0','--seconds','30','--label',record['description'],
        '--output',str(folder/'replay')],folder/'replay.log')
    summary=folder/'replay/summary.json'
    return dict(label=record['label'],stage='simulated' if rc==0 else 'replay_stopped_with_error',
                returncode=rc,summary=json.loads(summary.read_text()) if summary.exists() else None)


def render(records,output,videos,gpu):
    results=[]
    for record in records:
        folder=output/record['label']
        if not (folder/'replay/summary.json').exists():
            continue
        rc=run_command(['scripts.tools.parallel_ankle.render_replay',str(folder/'replay'),
            str(videos/(record['label']+'.mp4')),'--body-and-feet','--width','1440','--height','720',
            '--preview',str(folder/'preview.png')],folder/'render.log',
            dict(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(gpu)))
        results.append(dict(label=record['label'],stage='rendered' if rc==0 else 'render_failed',returncode=rc,gpu=gpu))
        print('[RENDER]',record['label'],rc,flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video-dir',type=Path,required=True)
    parser.add_argument('--render-gpus',type=int,nargs=2,default=[3,6])
    args=parser.parse_args();args.output=args.output.resolve();args.video_dir=args.video_dir.resolve()
    if args.output.exists() or args.video_dir.exists():raise FileExistsError('Use new output/video directories')
    records=[]
    for label,prefix,description in RUNS:
        runs=list((REPO_ROOT/'logs/rsl_rl/bsrl_amp').glob(prefix+'*'))
        if len(runs)!=1:raise ValueError('Ambiguous run '+prefix)
        run=runs[0]
        candidates=sorted([p for p in run.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)
                           and p.stat().st_size>100000],key=lambda p:int(p.stem.split('_')[1]))
        checkpoint=candidates[-1]
        records.append(dict(label=label,description=description,run=str(run),checkpoint=str(checkpoint),
            iteration=int(checkpoint.stem.split('_')[1]),checkpoint_sha256=sha(checkpoint),
            env_sha256=sha(run/'params/env.yaml'),agent_sha256=sha(run/'params/agent.yaml'),
            is_final_checkpoint=checkpoint.stem=='model_29999'))
    args.output.mkdir(parents=True);args.video_dir.mkdir(parents=True)
    save_json(args.output/'selection.json',dict(selected_at=datetime.now().isoformat(),records=records,
        scope='Seven recent BSRL experiments; excludes older archival runs and T1',
        fixed_command=[.6,0,0],requested_seconds=30,physics_pd_policy_hz=[1000,1000,50],
        warnings=['Actual frozen gains restored; common MuJoCo deployment is not exact Isaac training dynamics.',
                  'Latest checkpoint fixed at selection; unfinished runs have different training iterations.',
                  'No training process, checkpoint, original export, or asset edited.']))
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ('export_checkpoint.py','replay_policy.py','render_replay.py','record_recent_amp.py'):
        shutil.copy2(Path(__file__).parent/name,snapshot/name)
    print('[SELECTED]',json.dumps(records,ensure_ascii=False),flush=True)
    statuses=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(simulate,r,args.output):r for r in records}
        for future in as_completed(futures):
            try:result=future.result()
            except Exception:result=dict(label=futures[future]['label'],stage='exception',error=traceback.format_exc())
            statuses.append(result);print('[SIM]',json.dumps(result,ensure_ascii=False),flush=True)
    save_json(args.output/'simulation_status.json',statuses)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(render,records[i::2],args.output,args.video_dir,gpu) for i,gpu in enumerate(args.render_gpus)]
        rendering=[r for f in futures for r in f.result()]
    save_json(args.output/'render_status.json',rendering)
    print('[DONE] batch replay and render; run independent verification next',flush=True)


if __name__=='__main__':main()
