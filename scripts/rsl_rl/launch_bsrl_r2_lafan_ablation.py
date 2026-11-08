"""Launch the three R2-based LAFAN/ACCAD style-data ablations."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from bsrl_r2_lafan_ablation import BASELINE_R2_RUN, VARIANTS


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or len(args.gpus) != len(args.variants)
            or not 1 <= args.max_iterations <= 40000 or args.num_envs < 7):
        parser.error("Distinct GPUs and variants must be supplied one-to-one with valid sizes")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.splitlines()}
    if any(memory.get(gpu, 999999) > 1500 for gpu in args.gpus):
        raise RuntimeError(f"Selected GPU is occupied: {memory}")
    baseline = (args.baseline or root / BASELINE_R2_RUN).resolve()
    if not (baseline / "params/env.yaml").is_file():
        raise FileNotFoundError(baseline)
    manifest_path = root / "logs/launches" / f"bsrl_r2_lafan_ablation_{args.tag}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for gpu, variant in zip(args.gpus, args.variants):
        run_name = f"r2plant_{variant}_{args.max_iterations}it_{args.num_envs}env_{args.tag}"
        log_path = manifest_path.with_name(f"bsrl_r2_lafan_ablation_{args.tag}_{variant}.log")
        command = [
            "/root/miniconda3/envs/isaaclab/bin/python", "-u",
            "scripts/rsl_rl/train_bsrl_r2_lafan_ablation.py",
            "--baseline", str(baseline), "--variant", variant, "--run_name", run_name,
            "--num_envs", str(args.num_envs), "--max_iterations", str(args.max_iterations),
            "--headless", "--device", f"cuda:{gpu}",
        ]
        environment = dict(
            os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            BSRL_EXPLICIT_CLOSED_ANKLES="0", BSRL_ANKLE_TRANSMISSION_DR="0",
        )
        with log_path.open("x") as output:
            process = subprocess.Popen(
                command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        runs.append({
            "gpu": gpu, "variant": variant, "pid": process.pid,
            "run_name": run_name, "log": str(log_path), "command": command,
        })
        manifest_path.write_text(json.dumps({
            "baseline": str(baseline), "fresh_training": True,
            "max_iterations": args.max_iterations, "num_envs": args.num_envs,
            "started_at": datetime.now().isoformat(), "runs": runs,
        }, indent=2) + "\n")
        print(json.dumps(runs[-1]), flush=True)
    print(f"Launch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
