#!/usr/bin/env python3
"""Focused, reproducible analysis of the stationary-target ringing interval."""
import argparse
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __package__:
    from .analyze_bags import load_npz, builtin, KEY_JOINTS, SHORT_NAMES
    from ._paths import RESULTS_ROOT
else:
    from analyze_bags import load_npz, builtin, KEY_JOINTS, SHORT_NAMES
    from _paths import RESULTS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    folder = args.results / "20260904T181457-f11d9119"
    d = load_npz(folder / "state.npz")
    extra = load_npz(folder / "derived.npz")
    t = extra["time_s"]
    a,b=77.8,78.3
    sel=(t>=a)&(t<b)
    ts=np.arange(a,b,.001)
    def interp(array):
        return np.interp(ts,t,array)
    vel=interp(d["joint_velocity_raw_rad_s"][:,11])
    freqs, psd=signal.periodogram(vel,fs=1000,window="hann",detrend="linear")
    band=(freqs>=10)&(freqs<=100)
    peak=float(freqs[band][psd[band].argmax()])
    delays={}
    for i in [10,11]:
        actual=interp(d["actuator_torque_nm"][:,i])
        command=interp(d["actuator_target_torque_nm"][:,i])
        # Sweep a local delay only; do not confuse a whole oscillation period
        # with a delay. This is telemetry command-vs-current-inferred torque lag.
        errors=[]
        for lag in range(-10,11):
            x=command[20-lag:len(command)-20-lag]
            y=actual[20:-20]
            fit=np.column_stack([x,np.ones_like(x)])
            gain,bias=np.linalg.lstsq(fit,y,rcond=None)[0]
            errors.append(dict(lag_ms=lag,rmse_nm=float(np.sqrt(np.mean((y-gain*x-bias)**2))),
                               fitted_gain=float(gain),fitted_bias=float(bias)))
        best=min(errors,key=lambda x:x["rmse_nm"])
        delays[str(i)]={"all_lags":errors,"best":best}
    summary={
        "session":folder.name,"time_origin":"first realtime_state monotonic timestamp",
        "interval_s":[a,b], "sample_count":int(sel.sum()),
        "bag_relative_interval_s":[a+(int(d["header_time_ns"][0])-json.loads((folder/"metadata.json").read_text())["start_time_ns"])*1e-9,
                                   b+(int(d["header_time_ns"][0])-json.loads((folder/"metadata.json").read_text())["start_time_ns"])*1e-9],
        "left_roll_position_peak_to_peak_rad":float(np.ptp(d["joint_position_rad"][sel,11])),
        "left_roll_raw_speed_abs_max":float(np.max(np.abs(d["joint_velocity_raw_rad_s"][sel,11]))),
        "left_roll_target_peak_to_peak_rad":float(np.ptp(d["filtered_target_rad"][sel,11])),
        "left_roll_kp":np.unique(d["active_kp_nm_per_rad"][sel,11]),
        "left_roll_kd":np.unique(d["active_kd_nm_per_rad_s"][sel,11]),
        "left_roll_spectral_peak_hz":peak,
        "low_wkc_samples":int(np.sum(d["working_counter"][sel]!=d["expected_working_counter"][sel])),
        "left_motor_abs_current_max_a":np.max(np.abs(d["actual_current_a"][sel][:,[10,11]]),axis=0),
        "motor_current_inferred_torque_delay_fits":delays,
        "caution":"Delay includes drive response, filtering and telemetry timing; it is not an independently measured mechanical torque delay. Position is encoder/kinematics-derived, not an independent foot sensor.",
    }
    (folder/"ringing_detail.json").write_text(json.dumps(summary,indent=2,default=builtin)+"\n")
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True)
    axes[0].plot(t[sel],d["joint_position_rad"][sel,11],label="Left roll reconstructed angle")
    axes[0].plot(t[sel],d["filtered_target_rad"][sel,11],"--",label="Target (constant)")
    axes[0].set_ylabel("rad")
    axes[1].plot(t[sel],d["joint_velocity_raw_rad_s"][sel,11],label="Raw roll velocity")
    axes[1].plot(t[sel],d["joint_velocity_filtered_rad_s"][sel,11],label="10 Hz filtered roll velocity")
    axes[1].set_ylabel("rad/s")
    axes[2].plot(t[sel],d["commanded_joint_torque_nm"][sel,11],label="Roll torque command")
    axes[2].plot(t[sel],extra["measured_joint_torque_nm"][sel,11],label="Roll torque inferred from motor currents + J^T")
    axes[2].set_ylabel("Nm")
    for i,branch in [(10,"short"),(11,"long")]:
        axes[3].plot(t[sel],d["actual_current_a"][sel,i],label="Left "+branch+" motor")
    axes[3].set_ylabel("A");axes[3].set_xlabel("seconds from first realtime state")
    for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=9,loc="upper right")
    fig.suptitle("Stationary-target ringing: left ankle roll, Kp=100 / Kd=2, ~38-40 Hz\nNo low-WKC sample in this interval; currents below configured 10.8 A limit")
    fig.tight_layout();fig.savefig(folder/"left_roll_ringing_detail.png",dpi=150);plt.close(fig)
    print(json.dumps(summary,default=builtin),flush=True)

    for path in sorted(args.results.glob("*/summary.json")):
        s=json.loads(path.read_text());folder=path.parent
        state=load_npz(folder/"state.npz");extra=load_npz(folder/"derived.npz");t=extra["time_s"]
        for window in s["selected_windows"]:
            ix=np.flatnonzero((t>=window["start_s"])&(t<=window["end_s"]))
            columns={"time_s":t[ix],"cycle_sequence":state["cycle_sequence"][ix],
                     "working_counter":state["working_counter"][ix]}
            for i in KEY_JOINTS:
                prefix=SHORT_NAMES[i].replace(" ","_")
                for key in ["joint_position_rad","joint_velocity_raw_rad_s","joint_velocity_filtered_rad_s",
                            "filtered_target_rad","commanded_joint_torque_nm"]:
                    columns[prefix+"_"+key]=state[key][ix,i]
                columns[prefix+"_current_inferred_joint_torque_nm"]=extra["measured_joint_torque_nm"][ix,i]
            for i in [4,5,10,11]:
                side="right" if i<6 else "left";branch="short" if i in [4,10] else "long"
                columns[f"{side}_{branch}_motor_current_a"]=state["actual_current_a"][ix,i]
            with (folder/f'oscillation_mode{window["mode"]}.csv').open("w",newline="") as f:
                writer=csv.writer(f);writer.writerow(columns)
                writer.writerows(zip(*columns.values()))


if __name__=="__main__": main()
