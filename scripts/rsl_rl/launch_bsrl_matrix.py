"""Launch exactly three independent fresh BSRL trainings on idle GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from bsrl_training_matrix import VARIANTS


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=3, default=[0,4,5])
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if len(set(args.gpus)) != 3 or not 1 <= args.max_iterations <= 40000:
        parser.error("Three distinct GPUs and at most 40000 iterations required")
    # Read-only preflight: never evict another training to free resources.
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.splitlines()}
    if any(memory.get(gpu, 999999) > 1500 for gpu in args.gpus):
        raise RuntimeError(f"Selected GPU is occupied: {memory}")
    manifest_path = root / "logs/launches" / f"bsrl_matrix_{args.tag}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = root / "logs/rsl_rl/bsrl_amp/2026-09-17_18-18-30_accad34_reduced_latent20_h10_recent2_predict33_kl001_nomirror_8192"
    runs = []
    for gpu, variant in zip(args.gpus, VARIANTS):
        run_name = f"accad34_latent_{variant}_anklefixed_nomirror_noest_{args.max_iterations}it_{args.num_envs}_{args.tag}"
        log_path = manifest_path.with_name(f"bsrl_matrix_{args.tag}_{variant}.log")
        command = ["/root/miniconda3/envs/isaaclab/bin/python", "-u", "scripts/rsl_rl/train_bsrl_matrix.py",
            "--baseline", str(baseline), "--variant", variant, "--run_name", run_name,
            "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
            "--headless", "--device", "cuda:0"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            BSRL_EXPLICIT_CLOSED_ANKLES="0", BSRL_ANKLE_TRANSMISSION_DR="0")
        with log_path.open("x") as output:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        runs.append(dict(gpu=gpu, variant=variant, pid=process.pid, run_name=run_name,
                         log=str(log_path), command=command))
        # Persist every successful spawn, even if a later launch fails.
        manifest_path.write_text(json.dumps(dict(baseline=str(baseline), max_iterations=args.max_iterations,
            num_envs=args.num_envs, started_at=datetime.now().isoformat(), runs=runs), indent=2)+"\n")
        print(json.dumps(runs[-1]), flush=True)
    print(f"Launch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
