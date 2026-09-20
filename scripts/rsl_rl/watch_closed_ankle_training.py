"""Read-only, bounded acceptance monitor; never stops or changes training.

Writes a result at the requested iteration (including checkpoint validation),
or records an earlier failure. It does not call a 100-step environment check
a 100-iteration PPO pass. Invoke only for the explicitly selected new run.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time


def parse_log(contents):
    contents = re.sub(r"\x1b\[[0-9;]*m", "", contents)
    iterations = re.findall(r"Learning iteration\s+(\d+)/(\d+)", contents)
    audits = [dict((key, float(value)) for key, value in re.findall(r"(\w+)=([\deE+.-]+)", line))
              for line in contents.splitlines() if line.startswith("[ClosedAnkleAudit] physics_steps=")]
    metrics = {}
    nonfinite_metric = False
    for key, value in re.findall(r"^\s*((?:Mean |Episode_Reward/|Episode_Termination/|Metrics/)[^:\n]+):\s*([\w+.-]+)\s*$",
                                 contents, flags=re.M):
        try:
            number = float(value)
            nonfinite_metric |= not math.isfinite(number)
            metrics[key] = number if math.isfinite(number) else value
        except ValueError:
            pass
    failures = []
    if re.search(r"Traceback \(most recent call last\)|CUDA out of memory|Closed ankle physical audit failed", contents):
        failures.append("Training log contains an exception/physical-audit failure.")
    if nonfinite_metric:
        failures.append("A reported loss/reward/metric is not finite.")
    for audit in audits:
        # Old runs used a 10mm acceptance gate. New logs explicitly mark finite
        # closure residual as monitoring-only, while retaining other guards.
        strict_gap_failure = (not audit.get("gap_monitor_only", 0)
                              and audit.get("closure_max_mm", math.inf) > 10)
        if (strict_gap_failure or audit.get("motor_peak_Nm", math.inf) > 112.001
                or audit.get("passive_direct_Nm", math.inf) > 1e-6
                or audit.get("motor_power_peak_W", math.inf) > 1250.01
                or audit.get("nonfinite", 1) or audit.get("singular", 1)):
            failures.append("A physical audit exceeded its acceptance bound.")
            break
    return {
        "iteration": int(iterations[-1][0]) if iterations else -1,
        "audit_count": len(audits),
        "max_closure_mm": max((a.get("closure_max_mm", 0) for a in audits), default=None),
        "max_motor_Nm": max((a.get("motor_peak_Nm", 0) for a in audits), default=None),
        "max_passive_direct_Nm": max((a.get("passive_direct_Nm", 0) for a in audits), default=None),
        "last_metrics": metrics, "failures": failures,
    }


def validate_checkpoint(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    count = 0
    def visit(value):
        nonlocal count
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            count += 1
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Non-finite tensor in checkpoint")
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(state)
    if not count:
        raise ValueError("No model tensors found in checkpoint")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.run / f"model_{args.target}.pt"
    while True:
        result = parse_log(args.log.read_text(errors="replace"))
        result.update(status="running", target_iteration=args.target, run=str(args.run),
                      checked_at=datetime.now().astimezone().isoformat())
        cmdline = Path(f"/proc/{args.pid}/cmdline")
        try:
            command = cmdline.read_bytes()
            alive = (b"scripts/rsl_rl/train.py" in command
                     and args.run.name.split("_", 2)[-1].encode() in command)
        except FileNotFoundError:
            alive = False
        if result["failures"]:
            result["status"] = "failed"
        elif result["iteration"] >= args.target and checkpoint.is_file():
            # Save follows the iteration's logging; retry once if read raced
            # with the final file write. Never count mere file existence.
            try:
                result["finite_checkpoint_tensors"] = validate_checkpoint(checkpoint)
                result["checkpoint"] = str(checkpoint)
                result["status"] = "passed" if result["audit_count"] >= args.target else "failed"
                if result["status"] == "failed":
                    result["failures"].append("Missing rollout-level physical audits")
            except (OSError, RuntimeError, EOFError):
                time.sleep(5)
                try:
                    result["finite_checkpoint_tensors"] = validate_checkpoint(checkpoint)
                    result["checkpoint"] = str(checkpoint)
                    result["status"] = "passed" if result["audit_count"] >= args.target else "failed"
                except Exception as exc:
                    result["status"] = "failed"
                    result["failures"].append(f"Checkpoint validation failed: {exc}")
            except ValueError as exc:
                result["status"] = "failed"
                result["failures"].append(str(exc))
        elif not alive:
            result["status"] = "failed"
            result["failures"].append("Selected training process ended before acceptance")
        elif result["iteration"] > args.target+2 and not checkpoint.is_file():
            result["status"] = "failed"
            result["failures"].append("Training passed target but its requested checkpoint was not saved")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        print(f"[Acceptance] {result['checked_at']} status={result['status']} "
              f"iteration={result['iteration']} gap_mm={result['max_closure_mm']}", flush=True)
        if result["status"] != "running":
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
