#!/usr/bin/env python3
"""Compare saved door rollouts; no simulation, training or configuration changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    with np.load(folder / "frames.npz") as frames:
        time = frames["time"].copy()
        velocity = frames["body_velocity"][:, 0].copy()
    steady = velocity[time >= 5.0]
    dt = float(np.median(np.diff(time)))
    n = round(1.0 / dt)
    smooth = np.convolve(steady, np.ones(n) / n, mode="valid")
    command = float(manifest["command"][0])
    result = dict(
        folder=str(folder), command_vx=command, first_fall_time=manifest["first_fall_time"],
        mean_vx_after_5s=float(steady.mean()), std_vx_after_5s=float(steady.std()),
        min_max_vx_after_5s=[float(steady.min()), float(steady.max())],
        p05_p95_vx_after_5s=np.quantile(steady, [.05, .95]).tolist(),
        mean_command_error_after_5s=float(steady.mean() - command),
        rmse_command_after_5s=float(np.sqrt(np.mean((steady - command) ** 2))),
        min_max_1s_moving_mean_after_5s=[float(smooth.min()), float(smooth.max())],
        std_1s_moving_mean_after_5s=float(smooth.std()),
        final_yaw_drift_deg=manifest["final_yaw_drift_deg"],
        displacement_xy=manifest["displacement_xy"],
    )
    return result, time, velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path,
                        help="Optional root containing additional forward04 and forward08 rollouts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; do not overwrite an earlier comparison")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    runs = [(args.baseline_root, "06", "tab:blue"),
            (args.comparison_root, "10", "tab:orange")]
    if args.extra_root is not None:
        runs.extend([(args.extra_root, "04", "tab:purple"),
                     (args.extra_root, "08", "tab:green")])
        runs.sort(key=lambda run: int(run[1]))
    results = []
    for row, variant in enumerate(("0827", "0916")):
        for col, (suffix, label) in enumerate((("", "Original plant / explicit PD"),
                                             ("_user_t1", "User T1 plant / vendor gains"))):
            ax = axes[row, col]
            for root, tag, color in runs:
                folder = root / f"{variant}{suffix}_forward{tag}"
                stats, time, velocity = summarize(folder)
                stats.update(variant=variant, environment="original" if not suffix else "user_t1")
                results.append(stats)
                mask = (time >= 20) & (time <= 25)
                ax.plot(time[mask], velocity[mask], color=color,
                        label=f"cmd {stats['command_vx']:.1f}, mean {stats['mean_vx_after_5s']:.3f} m/s")
                ax.axhline(stats["command_vx"], color=color, alpha=.6, linestyle="--", linewidth=1)
            ax.set_title(f"{variant} | {label}")
            ax.set_xlim(20, 25)
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
            if col == 0:
                ax.set_ylabel("Instantaneous base body-frame vx (m/s)")
            if row == 1:
                ax.set_xlabel("Simulation time (s)")
    commands_text = " / ".join(f"{int(tag)/10:.1f}" for _, tag, _ in runs)
    fig.suptitle(f"Door ONNX: {commands_text} m/s command | fixed vy=wz=0 | same gains within each policy\n"
                 "Solid: measured speed; dashed: command. Display: 20-25s. Means: 5-30s. No phase alignment.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(args.output / "speed_comparison.png", dpi=160)
    plt.close(fig)
    if args.extra_root is not None:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        metrics = [("mean_vx_after_5s", "Mean body vx, 5-30s (m/s)"),
                   ("std_vx_after_5s", "Instantaneous vx standard deviation (m/s)"),
                   ("final_yaw_drift_deg", "Final yaw drift at 30s (deg)")]
        for row, variant in enumerate(("0827", "0916")):
            for col, (metric, ylabel) in enumerate(metrics):
                ax = axes[row, col]
                for env, color, label in (("original", "tab:blue", "Original environment"),
                                           ("user_t1", "tab:orange", "User T1 environment")):
                    subset = sorted((r for r in results if r["variant"] == variant and r["environment"] == env),
                                    key=lambda r: r["command_vx"])
                    ax.plot([r["command_vx"] for r in subset], [r[metric] for r in subset],
                            "o-", color=color, label=label)
                if col == 0:
                    ax.plot([.4, 1.], [.4, 1.], "k--", alpha=.5, label="Command")
                if col == 2:
                    ax.axhline(0, color="k", linestyle="--", alpha=.3)
                ax.set_title(f"{variant} | {ylabel}", fontsize=10)
                ax.set_xlabel("Forward command (m/s)")
                ax.set_xticks([.4, .6, .8, 1.])
                ax.grid(alpha=.25)
                ax.legend(fontsize=8)
        fig.suptitle("Door ONNX command sweep | same policy and gains, two environments\n"
                     "One deterministic 30s rollout per setting; 1.0 exceeds the original 0.8 joystick limit.", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(args.output / "command_response.png", dpi=160)
        plt.close(fig)
    report = dict(statistics_window="frame time >= 5s (50Hz samples), to the end of 30s rollout",
                  speed_definition="floating-base instantaneous body-frame vx; not whole-body COM speed",
                  smoothing="50 consecutive samples = 1s window; valid windows within the statistics interval",
                  caveat="1.0 m/s exceeds the delivered 0.8 m/s joystick forward limit; training range unknown",
                  results=results)
    (args.output / "statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for result in results:
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
