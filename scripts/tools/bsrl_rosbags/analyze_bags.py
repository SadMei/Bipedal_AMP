#!/usr/bin/env python3
"""Offline numerical checks, plots and CSV summaries for exported BSRL bags.

No bridge access or robot configuration changes. Frequencies use monotonic
measurement timestamps; ROS receive times are never treated as the servo clock.
Enum values are kept numeric because the bridge enum definitions were not uploaded.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import yaml

if __package__:
    from ._paths import BAG_ROOT, CORE_ROOT, RESULTS_ROOT
else:
    from _paths import BAG_ROOT, CORE_ROOT, RESULTS_ROOT

ROOT = Path(__file__).resolve().parent
KEY_JOINTS = [3, 4, 5, 9, 10, 11]
SHORT_NAMES = ["R yaw", "R hip roll", "R hip pitch", "R knee", "R ankle pitch", "R ankle roll",
               "L yaw", "L hip roll", "L hip pitch", "L knee", "L ankle pitch", "L ankle roll"]


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def builtin(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def uniques(x):
    vals, counts = np.unique(x, return_counts=True)
    return {str(val): int(count) for val, count in zip(vals, counts)}


def compile_core(core, build):
    core = Path(core)
    missing = [name for name in ("AnkleKinematics.hpp", "AnkleKinematics.cpp") if not (core / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Original BSRL solver missing from {core}: {missing}. "
            "Restore the original files or set BSRL_ANKLE_CORE_DIR / --core; no substitute solver is used."
        )
    include = build / "include" / "elmo_ethercat_sdk" / "bsrl"
    include.mkdir(parents=True)
    (include / "AnkleKinematics.hpp").symlink_to(core / "AnkleKinematics.hpp")
    shared = build / "ankle_batch.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC",
                    "-I", str(build / "include"), "-I", str(core),
                    str(core / "AnkleKinematics.cpp"), str(ROOT / "ankle_batch.cpp"),
                    "-o", str(shared)], check=True)
    lib = C.CDLL(str(shared))
    dptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.ankle_forward_batch.argtypes = [C.c_int, C.c_size_t, dptr, dptr, dptr, iptr]
    lib.ankle_forward_batch.restype = None
    return lib


def segments(t, keys):
    changes = np.any(keys[1:] != keys[:-1], axis=1) | (np.diff(t) <= 0) | (np.diff(t) > .01)
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    ends = np.r_[starts[1:], len(t)]
    return list(zip(starts, ends))


def analyze(folder, bag_root, lib, core_config):
    d = load_npz(folder / "state.npz")
    p = load_npz(folder / "policy.npz")
    imu = load_npz(folder / "imu.npz")
    meta = json.loads((folder / "metadata.json").read_text())
    cfg = yaml.safe_load((bag_root / folder.name / "snapshots" / "bsrl_bridge_bringup.yaml").read_text())["bridge"]
    t = (d["monotonic_time_ns"] - d["monotonic_time_ns"][0]) * 1e-9
    dt = np.diff(t)
    q = d["joint_position_rad"].astype(float)
    v = d["joint_velocity_raw_rad_s"].astype(float)
    vf = d["joint_velocity_filtered_rad_s"].astype(float)
    kp, kd = d["active_kp_nm_per_rad"], d["active_kd_nm_per_rad_s"]
    enabled = np.all((d["statusword_raw"] & 0x6f) == 0x27, axis=1)
    active = enabled & np.any(kp > 0, axis=1)
    valid_bus = d["working_counter"] == d["expected_working_counter"]
    # Stable current-cycle data are used for algebra checks. Non-full WKC cycles
    # may intentionally replay last torque/state and are reported separately.
    check = active & valid_bus
    check[1:] &= valid_bus[:-1]
    tau_motor_from_joint = np.array(d["commanded_joint_torque_nm"], dtype=float)
    tau_joint_from_measured = np.array(d["actuator_torque_nm"], dtype=float)
    tau_joint_from_target = np.array(d["actuator_target_torque_nm"], dtype=float)
    mapping = {}
    for side, inds in [("right", [4, 5]), ("left", [10, 11])]:
        trans = cfg["ankle_transmissions"][side]
        if trans["geometry"] != core_config["ankle_transmissions"][side]["geometry"]:
            raise ValueError(f"Geometry differs from uploaded core config: {folder.name} {side}")
        mi = trans["motor_indices"]
        assert inds == trans["joint_indices"]
        motors = np.zeros((len(t), 2), dtype=float)
        jac = np.zeros((len(t), 2, 2), dtype=float)
        status = np.zeros(len(t), dtype=np.int32)
        lib.ankle_forward_batch(int(side == "right"), len(t),
                               np.ascontiguousarray(q[:, inds]), motors, jac, status)
        good = check & (status == 0)
        inv = np.linalg.inv(jac)
        joint_v = np.einsum("nij,nj->ni", inv, d["actuator_velocity_rad_s"][:, mi])
        # actuator_position_rad is signed gearbox-output angle BEFORE zero offset,
        # verified below against both raw encoder counts and forward kinematics.
        motor_position = d["actuator_position_rad"][:, mi] - np.array(trans["encoder_zero_offset_rad"])
        tau_motor_from_joint[:, mi] = np.einsum("nji,nj->ni", inv, d["commanded_joint_torque_nm"][:, inds])
        tau_joint_from_measured[:, inds] = np.einsum("nji,nj->ni", jac, d["actuator_torque_nm"][:, mi])
        tau_joint_from_target[:, inds] = np.einsum("nji,nj->ni", jac, d["actuator_target_torque_nm"][:, mi])
        mapping[side] = {
            "forward_status": uniques(status[check]),
            "position_residual_p99_rad": np.percentile(np.abs(motors[good] - motor_position[good]), 99, axis=0),
            "velocity_mapping_error_rms": rms(joint_v[good] - v[good][:, inds]),
            "velocity_mapping_error_p99": np.percentile(np.abs(joint_v[good] - v[good][:, inds]), 99, axis=0),
            "joint_to_motor_torque_error_p99": np.percentile(np.abs(tau_motor_from_joint[good][:, mi] - d["actuator_target_torque_nm"][good][:, mi]), 99, axis=0),
            "max_condition": float(np.linalg.cond(jac[good]).max()),
        }
    limits = np.array([j["hardware"]["max_current_a"] for j in cfg["joints"]])
    torque_limits = np.array([j["safety"]["max_torque_nm"] for j in cfg["joints"]])
    encoder_scale = np.array([j["hardware"]["direction"] * 2*np.pi /
                             (j["hardware"]["position_encoder_resolution"] * j["hardware"]["gear_ratio"])
                             for j in cfg["joints"]])
    position_from_counts = d["position_actual_counts"] * encoder_scale
    velocity_from_units = d["velocity_actual_units"] * encoder_scale
    # Separate coordinate identity from physical accuracy: a correct algebraic
    # mapping does not validate encoder zero, linkage dimensions, or lack of play.
    # Telemetry contains newly received q[n], but the torque sent on this cycle
    # was computed from q[n-1]. This is established from the bag algebra, not
    # interpreted as a wrong controller rate or changed firmware timing.
    previous_q = np.vstack([q[:1], q[:-1]])
    pd = kp * (d["filtered_target_rad"] - previous_q) - kd * vf
    summary = {
        "session": folder.name, "bag_duration_s": meta["duration_s"],
        "state_duration_s": float(t[-1]), "state_samples": len(t),
        "state_start_bag_relative_s": (int(d["header_time_ns"][0]) - meta["start_time_ns"]) * 1e-9,
        "dt_ms_p0_p50_p99_p100": np.percentile(dt * 1000, [0, 50, 99, 100]),
        "cycle_sequence_missing": int(np.maximum(np.diff(d["cycle_sequence"]) - 1, 0).sum()),
        "phase_counts": uniques(d["phase"]), "mode_counts": uniques(d["command_mode"]),
        "stop_reason_counts": uniques(d["stop_reason"]),
        "active_samples": int(active.sum()), "active_wkc_mismatch": int((active & ~valid_bus).sum()),
        "active_max_consecutive_low_wkc": int(d["consecutive_low_working_counter"][active].max()),
        "active_exchange_duration_us_p50_p99_max": np.percentile(d["ethercat_exchange_duration_ns"][active]*.001, [50,99,100]),
        "active_wake_lateness_us_p50_p99_max": np.percentile(d["wake_lateness_ns"][active]*.001, [50,99,100]),
        "receiver_missing_samples_max": int(d["receiver_missing_samples"].max()),
        "receiver_crc_errors_max": int(d["receiver_crc_errors"].max()),
        "bridge_dropped_samples_max": int(d["bridge_dropped_samples"].max()),
        "drive_error_code_counts": uniques(d["error_code_raw"]),
        "encoder_position_conversion_error_max": np.max(np.abs(position_from_counts - d["actuator_position_rad"]), axis=0),
        "encoder_velocity_conversion_error_max": np.max(np.abs(velocity_from_units - d["actuator_velocity_rad_s"]), axis=0),
        "ankle_mapping": mapping,
        "pd_no_clip_residual_rms": rms(pd[check] - d["commanded_joint_torque_nm"][check]),
        "pd_joint_clip_residual_p99": np.percentile(np.abs(np.clip(pd[check], -torque_limits, torque_limits) - d["commanded_joint_torque_nm"][check]), 99, axis=0),
        "imu_accel_min": imu["linear_acceleration"].min(0), "imu_accel_max": imu["linear_acceleration"].max(0),
        "imu_accel_covariance_00": uniques(imu["linear_acceleration_covariance"][:,0]),
        "policy_inference_attempts": int(p["inference_attempted"].sum()),
        "policy_inference_failures": int((p["inference_attempted"] & ~p["inference_ok"]).sum()),
        "policy_inference_ms_p50_p99_max": np.percentile(p["inference_time_ms"][p["inference_attempted"]], [50,99,100]),
        "policy_requested_velocity_min": p["requested_velocity"].min(0),
        "policy_requested_velocity_max": p["requested_velocity"].max(0),
        "policy_applied_velocity_min": p["applied_velocity"].min(0),
        "policy_applied_velocity_max": p["applied_velocity"].max(0),
    }
    keys = np.column_stack([d["command_mode"], d["phase"], kp, kd, enabled])
    segs = segments(t, keys)
    seg_rows, joint_rows, windows = [], [], []
    for a,b in segs:
        if t[b-1] - t[a] < .2 or not np.all(active[a:b]): continue
        mode = int(d["command_mode"][a])
        seg_rows.append(dict(start_s=t[a], end_s=t[b-1], mode=mode, phase=int(d["phase"][a]),
                             samples=b-a, kp=kp[a].tolist(), kd=kd[a].tolist()))
        ts = np.arange(t[a], t[b-1], .001)
        qu = np.column_stack([np.interp(ts, t[a:b], q[a:b,i]) for i in range(12)])
        vu = np.column_stack([np.interp(ts, t[a:b], v[a:b,i]) for i in range(12)])
        # Position derivative with centered 11 ms smoothing for quantization.
        if len(ts) >= 101:
            qdot = signal.savgol_filter(qu, 11, 3, deriv=1, delta=.001, axis=0)
            for i in KEY_JOINTS:
                corr = float(np.corrcoef(qdot[10:-10,i], vu[10:-10,i])[0,1])
                err = d["filtered_target_rad"][a:b,i] - q[a:b,i]
                freqs, psd = signal.welch(vu[:,i], fs=1000, nperseg=min(4096,len(ts)))
                band = (freqs >= 5) & (freqs <= 100)
                peak = float(freqs[band][np.argmax(psd[band])]) if band.any() else None
                joint_rows.append(dict(start_s=t[a], end_s=t[b-1], mode=mode,
                    joint=SHORT_NAMES[i], kp=float(kp[a,i]), kd=float(kd[a,i]),
                    position_error_rms_rad=float(rms(err)), position_error_max_rad=float(np.max(np.abs(err))),
                    raw_velocity_rms=float(rms(v[a:b,i])), filtered_velocity_rms=float(rms(vf[a:b,i])),
                    velocity_derivative_correlation=corr, velocity_highfreq_peak_hz=peak,
                    current_abs_max_a=float(np.max(np.abs(d["actual_current_a"][a:b,i]))),
                    current_near_limit_fraction=float(np.mean(np.abs(d["actual_current_a"][a:b,i]) >= .98*limits[i])),
                    joint_torque_abs_max_nm=float(np.max(np.abs(d["commanded_joint_torque_nm"][a:b,i]))),
                    motor_torque_abs_max_nm=float(np.max(np.abs(d["actuator_target_torque_nm"][a:b,i])))))
        if len(ts) >= 5000:
            vh = signal.sosfiltfilt(signal.butter(3, 5, fs=1000, btype="highpass", output="sos"), vu, axis=0)
            # Find a 4 s window with high-frequency ankle velocity; avoid start/end.
            score = np.mean(vh[:,[4,5,10,11]]**2, axis=1)
            cumulative = np.r_[0, np.cumsum(score)]
            energy = (cumulative[4000:] - cumulative[:-4000])/4000
            energy[:500] = -1
            energy[-500:] = -1
            start = int(energy.argmax())
            windows.append(dict(start_s=float(ts[start]), end_s=float(ts[start+3999]),
                                score=float(energy[start]), mode=mode))
    summary["segments"] = seg_rows
    summary["joint_statistics"] = joint_rows
    summary["selected_windows"] = windows

    # Check one-cycle ordering: vf[n]-vf[n-1] = alpha*(v[n-1]-vf[n-1]).
    # The implementation matches alpha=1-exp(-2*pi*fc*0.001), not backward Euler.
    filter_stats = []
    for mode in [1,2]:
        pair = check[1:] & check[:-1] & (d["command_mode"][1:]==mode) & (d["command_mode"][:-1]==mode)
        for i in KEY_JOINTS:
            x = (v[:-1,i]-vf[:-1,i])[pair]; y = np.diff(vf[:,i])[pair]
            alpha = float(x@y/(x@x)) if x@x>0 else 0
            residual = float(rms(y-alpha*x))
            implied = -np.log1p(-alpha)/(2*np.pi*.001) if 0<alpha<1 else None
            filter_stats.append(dict(mode=mode,joint=SHORT_NAMES[i],alpha=alpha,
                                     exponential_equivalent_cutoff_hz=implied,residual_rms=residual))
    summary["joint_filter_fit"] = filter_stats
    slew_stats=[]
    pair=check[1:] & check[:-1] & (d["command_mode"][1:]==2) & (d["command_mode"][:-1]==2)
    delta=np.abs(np.diff(d["filtered_target_rad"].astype(float),axis=0))[pair]
    for i in KEY_JOINTS:
        slew_stats.append(dict(joint=SHORT_NAMES[i],delta_rad_p50_p95_max=np.percentile(delta[:,i],[50,95,100]),
                               fraction_at_0_00025_rad_per_cycle=float(np.mean(np.isclose(delta[:,i],.00025,atol=2e-7)))))
    summary["policy_filtered_target_slew"]=slew_stats
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=builtin) + "\n")
    with (folder / "joint_statistics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(joint_rows[0])); writer.writeheader(); writer.writerows(joint_rows)
    np.savez_compressed(folder / "derived.npz", time_s=t, active=active,
                        measured_joint_torque_nm=tau_joint_from_measured,
                        delivered_joint_target_torque_nm=tau_joint_from_target)

    # Overview, all points subsampled for display only; exported NPZ is full rate.
    fig,axes=plt.subplots(4,2,figsize=(17,12),sharex=True)
    for col,side in enumerate(["right","left"]):
        inds=[3,4,5] if side=="right" else [9,10,11]
        for axidx,i in enumerate(inds):
            ax=axes[axidx,col]
            sel=np.arange(0,len(t),10)
            ax.plot(t[sel],q[sel,i],label="actual",lw=.8)
            ax.plot(t[sel],d["filtered_target_rad"][sel,i],label="filtered target",lw=.8)
            ax.set_title(SHORT_NAMES[i]); ax.set_ylabel("rad"); ax.grid(alpha=.25)
            ax.legend(loc="upper right",fontsize=8)
        ax=axes[3,col]
        for i in inds:
            ax.plot(t[::10],d["actual_current_a"][::10,i],label=SHORT_NAMES[i],lw=.6)
        ax.set_ylabel("motor current A"); ax.set_xlabel("seconds from first realtime state")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(folder.name+" | full recording (startup/stop included)")
    fig.tight_layout(); fig.savefig(folder/"overview.png",dpi=130); plt.close(fig)

    for window in windows:
        sel=np.flatnonzero((t>=window["start_s"]) & (t<=window["end_s"]))
        fig,axes=plt.subplots(4,3,figsize=(19,12),sharex=True)
        for col,(ri,li) in enumerate([(3,9),(4,10),(5,11)]):
            for i,color in [(ri,"tab:blue"),(li,"tab:orange")]:
                label=SHORT_NAMES[i]
                axes[0,col].plot(t[sel],q[sel,i],c=color,label=label,lw=.8)
                axes[0,col].plot(t[sel],d["filtered_target_rad"][sel,i],c=color,ls="--",lw=.8)
                axes[1,col].plot(t[sel],v[sel,i],c=color,lw=.7,label=label+" raw")
                axes[1,col].plot(t[sel],vf[sel,i],c=color,ls="--",lw=.8,label=label+" filtered")
                axes[2,col].plot(t[sel],d["commanded_joint_torque_nm"][sel,i],c=color,lw=.8,label=label+" command")
                axes[2,col].plot(t[sel],tau_joint_from_measured[sel,i],c=color,ls=":",lw=.8,label=label+" measured (J^T)")
                motor_label=label if i in [3,9] else (("R" if i<6 else "L") + (" short motor" if i in [4,10] else " long motor"))
                axes[3,col].plot(t[sel],d["actual_current_a"][sel,i],c=color,lw=.8,label=motor_label)
            for row,ylabel in enumerate(["joint position rad", "joint velocity rad/s", "joint torque Nm", "motor current A"]):
                axes[row,col].set_ylabel(ylabel); axes[row,col].grid(alpha=.25); axes[row,col].legend(fontsize=7)
            axes[3,col].set_xlabel("seconds from first realtime state")
        fig.suptitle(f'{folder.name} | mode {window["mode"]} | strongest 4 s ankle HF window; dashed position = target')
        fig.tight_layout(); fig.savefig(folder/f'oscillation_mode{window["mode"]}.png',dpi=130); plt.close(fig)
    print(json.dumps({"session":folder.name,"mapping":mapping,"windows":windows,
                      "segments":seg_rows,"pd_residual":summary["pd_joint_clip_residual_p99"]},default=builtin),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=RESULTS_ROOT)
    parser.add_argument("--bag-root",type=Path,default=BAG_ROOT)
    parser.add_argument("--core",type=Path,default=CORE_ROOT)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="bsrl-bag-kinematics-") as build:
        lib=compile_core(args.core,Path(build))
        core_config=yaml.safe_load((args.core/"bsrl_bridge_bringup.yaml").read_text())["bridge"]
        for folder in sorted(args.results.iterdir()):
            if folder.is_dir() and (folder/"metadata.json").exists():
                analyze(folder,args.bag_root,lib,core_config)


if __name__=="__main__": main()
