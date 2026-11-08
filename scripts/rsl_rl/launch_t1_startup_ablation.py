"""Launch the formal T1 startup-reset AMP ablation in a detached session."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 40000:
        parser.error("max-iterations must be 1..40000")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.splitlines()}
    if memory.get(args.gpu, 999999) > 1500:
        raise RuntimeError(f"Selected GPU is occupied: {memory}")
    baseline = args.baseline.resolve(strict=True)
    run_name = (
        "latent20_commandref_defaultstart20_nomirror_noactorheading_"
        f"{args.max_iterations}it_{args.num_envs}_{args.tag}"
    )
    launch_dir = root / "logs/launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    log_path = launch_dir / f"t1_startup_ablation_{args.tag}.log"
    manifest_path = launch_dir / f"t1_startup_ablation_{args.tag}.json"
    command = ["/root/miniconda3/envs/isaaclab/bin/python", "-u",
               "scripts/rsl_rl/train_t1_startup_ablation.py",
               "--baseline", str(baseline), "--run_name", run_name,
               "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
               "--headless", "--device", f"cuda:{args.gpu}"]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with log_path.open("x") as output:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    manifest = dict(gpu=args.gpu, pid=process.pid, baseline=str(baseline), run_name=run_name,
                    num_envs=args.num_envs, max_iterations=args.max_iterations,
                    log=str(log_path), command=command, started_at=datetime.now().isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
