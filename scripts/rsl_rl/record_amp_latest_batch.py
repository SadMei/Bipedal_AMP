"""Freeze five active latent AMP checkpoints and record them on two idle GPUs.

No training/checkpoint changes; each rollout uses its own saved plant/controller.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ("bsrl_scale01", "BSRL scale0.1 originalPD", "bsrl_amp",
     "2026-09-19_02-30-51_accad34_latent_scale01_anklefixed_nomirror_noest_40000it_8192_formal_20260919"),
    ("bsrl_matched_bigpd", "BSRL matched-reference bigPD", "bsrl_amp",
     "2026-09-19_02-30-50_accad34_latent_matched_bigpd_anklefixed_nomirror_noest_40000it_8192_formal_20260919"),
    ("bsrl_clean_bigpd", "BSRL cleanAMP bigPD", "bsrl_amp",
     "2026-09-19_02-30-51_accad34_latent_clean_bigpd_anklefixed_nomirror_noest_40000it_8192_formal_20260919"),
    ("t1_resumed", "T1 resumed from24999", "men_t1_amp",
     "2026-09-19_00-15-40_latent20_resume24999_to40000_sameconfig_nomirror_noactorheading_8192"),
    ("t1_clean", "T1 cleanAMP", "men_t1_amp",
     "2026-09-18_17-51-20_latent20_cleanamp_uniform08_06_15_noheading_nocond_nomirror_40k_8192"),
)


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[1,3])
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--speed", type=float, default=.6)
    parser.add_argument("--selection-file", type=Path,
                        help="Reuse previously frozen checkpoints, changing only rollout settings")
    parser.add_argument("--launch-interval", type=float, default=30.,
                        help="Minimum seconds between simulator launches to stagger graphics initialization")
    args = parser.parse_args()
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.splitlines()}
    if len(set(args.gpus)) != 2 or any(memory.get(gpu,999999) > 1500 for gpu in args.gpus):
        raise RuntimeError(f"Select two idle GPUs, without stopping training: {memory}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = []
    frozen = None
    if args.selection_file:
        frozen = json.loads(args.selection_file.read_text())["recordings"]
        if [row["key"] for row in frozen] != [row[0] for row in RUNS]:
            raise ValueError("Frozen selection must contain the same five runs in order")
    for key, label, task, name in RUNS:
        run = ROOT / "logs/rsl_rl" / task / name
        previous = next((row for row in frozen if row["key"] == key), None) if frozen else None
        if previous:
            checkpoint = Path(previous["checkpoint"])
            if checkpoint.parent.resolve() != run.resolve():
                raise ValueError(f"Frozen checkpoint is from a different run: {checkpoint}")
        else:
            candidates = sorted((p for p in run.glob("model_*.pt") if p.stem[6:].isdigit()),
                                key=lambda p: int(p.stem[6:]), reverse=True)
            checkpoint = next(p for p in candidates if time.time() - p.stat().st_mtime > 5)
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if previous and (checksum != previous["checkpoint_sha256"]
                         or int(checkpoint.stem[6:]) != previous["iteration"]):
            raise RuntimeError(f"Frozen checkpoint changed: {checkpoint}")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (loaded["iter"] != int(checkpoint.stem[6:])
                or not all(torch.isfinite(v).all() for v in loaded["model_state_dict"].values())
                or checksum != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            raise RuntimeError(f"Unstable or invalid checkpoint {checkpoint}")
        selected.append(dict(key=key, label=label, checkpoint=str(checkpoint),
                             iteration=int(loaded["iter"]), checkpoint_sha256=checksum))
        del loaded
    manifest = dict(selected_at=datetime.now().isoformat(), command=[args.speed,0.,0.], seconds=args.seconds,
                    simulator="Isaac nominal saved-run controller", recordings=selected)
    if args.selection_file:
        manifest["selection_source"] = str(args.selection_file.resolve())
    (output / "selection.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)

    launch_lock = threading.Lock()
    last_launch = [float('-inf')]

    def worker(gpu, rows):
        results = []
        for row in rows:
            target = output / f"{row['key']}_model{row['iteration']}_forward{round(args.speed*10):02d}"
            command = ["/root/miniconda3/envs/isaaclab/bin/python", "-u", "scripts/rsl_rl/record_latent.py",
                "--checkpoint", row['checkpoint'], "--output-dir", str(target), "--label", row['label'],
                "--command", str(args.speed), "0", "0", "--seconds", str(args.seconds),
                "--pd-profile", "nominal", "--headless", "--device", f"cuda:{gpu}"]
            environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            environment.pop("CUDA_VISIBLE_DEVICES", None)  # absolute CUDA/Vulkan GPU IDs for rendering
            log = output / f"{row['key']}.log"
            with log.open("x") as stream:
                with launch_lock:
                    delay = max(0., args.launch_interval - (time.monotonic() - last_launch[0]))
                    if delay:
                        time.sleep(delay)
                    print(f"[BATCH_START] gpu={gpu} {row['key']} model{row['iteration']}",flush=True)
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    last_launch[0] = time.monotonic()
                returncode = process.wait()
            result = dict(**row, gpu=gpu, output=str(target), log=str(log), exit_code=returncode)
            if returncode == 0:
                actual = json.loads((target/'manifest.json').read_text())
                if actual['checkpoint_sha256'] != row['checkpoint_sha256']:
                    raise RuntimeError("Recording used a different checkpoint than selection")
                result.update(first_fall_time=actual['first_fall_time'],
                              mean_body_vx=actual['mean_body_vx'], yaw=actual['final_yaw_drift_degrees'])
            results.append(result)
            print("[BATCH_RESULT] "+json.dumps(result),flush=True)
        return results

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker,gpu,selected[index::2]) for index,gpu in enumerate(args.gpus)]
        results = [row for future in futures for row in future.result()]
    (output/'batch_results.json').write_text(json.dumps(results,indent=2)+"\n")
    if any(row['exit_code'] for row in results):
        raise RuntimeError("Some recordings failed; inspect batch_results.json")
    print(f"[BATCH_COMPLETE] {output}",flush=True)


if __name__ == "__main__":
    main()
