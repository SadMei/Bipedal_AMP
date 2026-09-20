"""Validate isolated 100STYLE/BSRL outputs and plot source-versus-robot paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget_100style_bsrl import LABELS, MOVEMENTS, source_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path("/root/mjw/GMR/motion_data/100STYLE_Neutral_source"))
    parser.add_argument("--gmr-dir", type=Path, default=Path("/root/mjw/GMR"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.gmr_dir))
    from general_motion_retargeting.params import ROBOT_VIEWER_XML_DICT
    model = mujoco.MjModel.from_xml_path(str(ROBOT_VIEWER_XML_DICT["bsrl"]))
    data = mujoco.MjData(model)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    summary = {}
    for movement, axis in zip(MOVEMENTS, axes.flat):
        stem = f"Neutral_{movement}"
        with (args.run / "gmr" / f"{stem}.pkl").open("rb") as stream:
            motion = pickle.load(stream)
        metrics = json.loads((args.run / "metrics" / f"{stem}.json").read_text())
        source_q, source_x, skeleton, source_info = source_motion(
            args.source_dir / f"{stem}.bvh", args.source_dir / "Frame_Cuts.csv")
        if len(source_q) != len(motion["root_pos"]):
            raise ValueError(f"Incomplete retargeted clip: {stem}")
        if not metrics["finite"] or metrics["joint_limit_violation_rad"] > 1e-5:
            raise ValueError(f"Invalid retargeted data: {stem}")
        np.testing.assert_allclose(np.linalg.norm(motion["root_rot"], axis=1), 1., atol=1e-6)
        max_fk_error = 0.
        for index in np.linspace(0, len(source_q)-1, 13, dtype=int):
            data.qpos[:3] = motion["root_pos"][index]
            data.qpos[3:7] = motion["root_rot"][index, [3, 0, 1, 2]]
            for joint_idx, name in enumerate(motion["dof_names"]):
                data.joint(name).qpos[0] = motion["dof_pos"][index, joint_idx]
            mujoco.mj_forward(model, data)
            for body_idx, body in enumerate(motion["key_body_names"]):
                max_fk_error = max(max_fk_error, float(np.linalg.norm(
                    data.body(body).xpos - motion["key_body_pos"][index, body_idx])))
        if max_fk_error > 1e-5:
            raise ValueError(f"Viewer and retargeter FK disagree: {stem}, {max_fk_error}")
        root_idx = skeleton.bones.index("Hips")
        source_matrix = Rotation.from_quat(source_q[:, root_idx], scalar_first=True).as_matrix()
        source_yaw = np.unwrap(np.arctan2(source_matrix[:, 1, 0], source_matrix[:, 0, 0]))
        yaw_rate = np.gradient(source_yaw, 1./source_info["fps"])
        # Window-scale turning ignores natural instantaneous pelvis oscillation.
        window = round(source_info["fps"])
        averaged_yaw_rate = np.convolve(yaw_rate, np.ones(window)/window, mode="valid")
        metrics["source_mean_abs_yaw_rate_rad_s"] = float(np.abs(yaw_rate).mean())
        metrics["source_turning_fraction_1s_above_0p2_rad_s"] = float(np.mean(np.abs(averaged_yaw_rate) > .2))
        metrics["viewer_key_body_fk_max_error_m"] = max_fk_error
        summary[stem] = metrics
        src = source_x[:, root_idx, :2] * motion["human_to_robot_scale"]
        robot = motion["root_pos"][:, :2]
        axis.plot(src[:, 0], src[:, 1], "--", color="0.5", lw=1.2, label="Source (scaled)")
        axis.plot(robot[:, 0], robot[:, 1], color="#1878b5", lw=.9, label="BSRL IK")
        axis.scatter(*robot[0], color="green", s=22, zorder=3, label="Start")
        axis.scatter(*robot[-1], color="red", s=22, zorder=3, label="End")
        axis.set_title(f"{movement}: {LABELS[movement]}\n{metrics['duration_s']:.1f}s; turning windows {metrics['source_turning_fraction_1s_above_0p2_rad_s']:.0%}")
        axis.set_xlabel("X (m)"); axis.set_ylabel("Y (m)")
        axis.set_aspect("equal", adjustable="datalim"); axis.grid(alpha=.25)
        print(f"[AUDIT] {stem}: {metrics['frames']} frames, FK max error={max_fk_error:.2g}m, source turning={metrics['source_turning_fraction_1s_above_0p2_rad_s']:.1%}", flush=True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("100STYLE Neutral: movement categories are NOT straight-line trajectory labels", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    image_path, json_path = args.run / "trajectory_audit.png", args.run / "audit.json"
    if image_path.exists() or json_path.exists():
        raise FileExistsError("Audit outputs already exist")
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    json_path.write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
