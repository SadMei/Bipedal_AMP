#!/usr/bin/env python3
"""Summarize physical-step Isaac drive efforts and implied parallel motor loads.

This does NOT re-run physics with coupled motor limits. Motor-side quantities
are inverse-transmission requirements of the existing independently driven policy.
RMS/current/copper-loss ratios are load proxies, not a validated thermal model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def longest_run(mask, dt):
    transitions=np.diff(np.r_[False,mask,False].astype(np.int8))
    lengths=np.flatnonzero(transitions==-1)-np.flatnonzero(transitions==1)
    return float(lengths.max()*dt) if len(lengths) else 0.


def rolling_rms_max(torque, dt, seconds):
    width=int(round(seconds/dt))
    if len(torque)<width: return None
    sums=np.r_[0.,np.cumsum(torque**2)]
    return float(np.sqrt(np.max((sums[width:]-sums[:-width])/width)))


def stats(torque, speed, rated, peak, conservative, peak_speed, peak_power, dt):
    absolute=np.abs(torque)
    rms=float(np.sqrt(np.mean(torque**2)))
    mechanical=torque*speed
    return dict(rated_nm=rated, datasheet_peak_nm=peak, rms_nm=rms,
                rms_over_rated=rms/rated, copper_loss_ratio_proxy=(rms/rated)**2,
                abs_mean_nm=float(absolute.mean()), abs_p95_nm=float(np.percentile(absolute,95)),
                abs_p99_nm=float(np.percentile(absolute,99)), abs_max_nm=float(absolute.max()),
                above_rated_pct=float(np.mean(absolute>rated)*100),
                longest_above_rated_s=longest_run(absolute>rated,dt),
                above_conservative_pct=float(np.mean(absolute>conservative)*100),
                above_datasheet_peak_pct=float(np.mean(absolute>peak)*100),
                rolling_1s_rms_max_nm=rolling_rms_max(torque,dt,1),
                rolling_10s_rms_max_nm=rolling_rms_max(torque,dt,10),
                abs_speed_p99_rad_s=float(np.percentile(np.abs(speed),99)),
                abs_speed_max_rad_s=float(np.max(np.abs(speed))),
                above_peak_speed_pct=float(np.mean(np.abs(speed)>peak_speed)*100),
                positive_mech_power_mean_w=float(np.maximum(mechanical,0).mean()),
                negative_mech_power_mean_w=float(np.minimum(mechanical,0).mean()),
                abs_mech_power_max_w=float(np.max(np.abs(mechanical))),
                above_peak_mech_power_pct=float(np.mean(np.abs(mechanical)>peak_power)*100))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    repo_root=Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.tools.bsrl_rosbags import analyze_bags as module
    names=manifest["joint_names"];dt=manifest["physics_dt"]
    all_rows=[];summaries=[]
    with tempfile.TemporaryDirectory(prefix="bsrl-isaac-torque-") as build:
        lib=module.compile_core(module.CORE_ROOT,Path(build))
        for profile in manifest["profiles"]:
            label=profile["name"]
            with np.load(root/f"{label}.npz") as z: d={key:z[key].astype(float) for key in z.files}
            n=len(d["time_s"])
            assert n==profile["samples"] and n>0
            assert np.allclose(d["command"],profile["command"],atol=1e-6)
            assert all(np.all(np.isfinite(array)) for array in d.values())
            motor_torque=d["applied_torque"].copy();motor_speed=d["joint_vel"].copy()
            motor_names=names.copy();max_condition=0.;virtual_work_error=0.
            for side in ["right","left"]:
                ids=[names.index(f"joint_{side}_ankle_{axis}") for axis in ["pitch","roll"]]
                motor_angle=np.zeros((n,2));jac=np.zeros((n,2,2));status=np.zeros(n,dtype=np.int32)
                lib.ankle_forward_batch(int(side=="right"),n,np.ascontiguousarray(d["joint_pos"][:,ids]),motor_angle,jac,status)
                if np.any(status): raise ValueError(f"Unreachable/invalid linkage posture in {label}/{side}: {np.unique(status,return_counts=True)}")
                motor_torque[:,ids]=np.linalg.solve(jac.transpose(0,2,1),d["applied_torque"][:,ids,None])[:,:,0]
                motor_speed[:,ids]=np.einsum("nij,nj->ni",jac,d["joint_vel"][:,ids])
                max_condition=max(max_condition,float(np.linalg.cond(jac).max()))
                virtual_work_error=max(virtual_work_error,float(np.abs(np.sum(motor_torque[:,ids]*motor_speed[:,ids],axis=1)-np.sum(d["applied_torque"][:,ids]*d["joint_vel"][:,ids],axis=1)).max()))
                # Math branch order is long, short; output names intentionally
                # replace the slot names to avoid calling a motor 'pitch/roll'.
                for j,branch in zip(ids,["long","short"]):motor_names[j]=f"motor_{side}_ankle_{branch}"
            np.savez_compressed(root/f"{label}_motor.npz",time_s=d["time_s"],motor_torque_nm=motor_torque,
                                motor_speed_rad_s=motor_speed,motor_names=np.array(motor_names))
            stiffness=np.zeros(len(names));damping=np.zeros(len(names));limits=np.zeros(len(names))
            for cfg in manifest["actuators"].values():
                ix=cfg["joint_indices"];stiffness[ix]=cfg["stiffness"];damping[ix]=cfg["damping"];limits[ix]=cfg["effort_limit"]
            pd=stiffness*(d["joint_pos_target"]-d["joint_pos"])-damping*d["joint_vel"]
            pd_error=float(np.max(np.abs(pd-d["computed_torque"])))
            speed_mean=d["root_lin_vel_b"][d["time_s"]>=5].mean(axis=0) if n>1000 else d["root_lin_vel_b"].mean(axis=0)
            sm=dict(profile=label,command=profile["command"],duration_s=n*dt,termination=profile["termination"],
                    mean_body_speed_after5s=speed_mean.tolist(),root_displacement=(d["root_pos_w"][-1]-d["root_pos_w"][0]).tolist(),
                    minimum_base_height=float(d["root_pos_w"][:,2].min()),
                    max_linkage_condition=max_condition,virtual_work_max_error_w=virtual_work_error,
                    nominal_PD_reconstruction_max_error_nm=pd_error,
                    simulated_any_joint_clipped_pct=float(np.mean(np.any(np.abs(d["computed_torque"]-d["applied_torque"])>1e-4,axis=1))*100),
                    actual_drives_equal_configured_effort_caps_pct=float(np.mean(np.any(np.abs(d["applied_torque"])>=.999*limits,axis=1))*100))
            assert pd_error<.002
            for scope,mask in [("all",np.ones(n,dtype=bool)),("after5s",d["time_s"]>=5)]:
                if not mask.any(): continue
                for j,name in enumerate(motor_names):
                    small=("ankle" in name or "hip_yaw" in name)
                    params=(20.,112.,24.,250*2*np.pi/60,1250.) if small else (45.,216.,54.,144.5*2*np.pi/60,1050.)
                    row=dict(profile=label,scope=scope,motor=name,
                             **stats(motor_torque[mask,j],motor_speed[mask,j],*params,dt))
                    row["estimated_current_rms_a"]=row["rms_nm"]/(20/9 if small else 45/10)
                    row["estimated_current_peak_a"]=row["abs_max_nm"]/(20/9 if small else 45/10)
                    all_rows.append(row)
            summaries.append(sm)
            # Plots use motor-side loads at the unchanged original 200 Hz rate.
            fig,axes=plt.subplots(3,2,figsize=(17,10),sharex=True)
            for col,side in enumerate(["right","left"]):
                keys=[f"joint_{side}_knee_pitch",f"motor_{side}_ankle_long",f"motor_{side}_ankle_short"]
                for row,name in enumerate(keys):
                    ax=axes[row,col];j=motor_names.index(name);rated=45 if row==0 else 20
                    ax.plot(d["time_s"],motor_torque[:,j],lw=.5,label="Implied motor output torque")
                    ax.axhline(rated,c="tab:orange",ls="--",label="Rated +/-")
                    ax.axhline(-rated,c="tab:orange",ls="--")
                    ax.set_title(name);ax.set_ylabel("Nm");ax.grid(alpha=.25);ax.legend(fontsize=7)
                    if row==2:ax.set_xlabel("seconds")
            fig.suptitle(f"BSRL model29999 | {label} | motor-side torque, nominal Isaac replay")
            fig.tight_layout();fig.savefig(root/f"{label}_motor_torque.png",dpi=140);plt.close(fig)
            print(json.dumps(sm),flush=True)
    with (root/"motor_statistics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (root/"summary.json").write_text(json.dumps({"profiles":summaries,"motor_statistics":all_rows},indent=2)+"\n")
    lines=["# BSRL Isaac 力矩实测（2026-09-14）", "", f"Checkpoint：`{manifest['checkpoint']}`", "",
           "固定指令分别回放 60 s；名义条件、默认站姿启动，无噪声/推力/域随机化/执行器延迟。每个 200 Hz 物理步记录限幅前和限幅后的驱动力矩。未更改 PD、限值或策略权重。", "",
           "踝部电机数据是用上传 C++ 机构模型的 J⁻ᵀ 对 Isaac 关节驱动力矩换算得到的所需减速器输出力矩，不是已加入并联机构的重新仿真。不包含摩擦损耗、柔性、转子惯量差异或独立测量的热模型。", "",
           "## 回放有效性", "", "| 指令 | 时长 s | 实际 body vx 均值（5s后） | 提前终止 | 任一关节执行器限幅占比 |", "|---|---:|---:|---|---:|"]
    for s in summaries:lines.append(f"| {s['command']} | {s['duration_s']:.1f} | {s['mean_body_speed_after5s'][0]:.3f} | {s['termination']} | {s['simulated_any_joint_clipped_pct']:.3f}% |")
    lines += ["", "## 电机侧持续负载（去掉启动前 5 s）", "", "RMS=均方根，非有正负抵消的平均力矩；额定为小电机20、大电机45 N·m。左右分别统计。", "",
              "| 指令 | 电机 | RMS Nm | P99绝对值 Nm | 最大绝对值 Nm | 超额定占比 | 最长连续超额定 s | 最大10s RMS Nm |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        if row["scope"]!="after5s":continue
        ten=row["rolling_10s_rms_max_nm"]
        lines.append(f"| {row['profile']} | {row['motor']} | {row['rms_nm']:.2f} | {row['abs_p99_nm']:.2f} | {row['abs_max_nm']:.2f} | {row['above_rated_pct']:.2f}% | {row['longest_above_rated_s']:.3f} | {ten:.2f} |" if ten is not None else "")
    lines += ["", "## 如何解释", "",
              "- 最大值说明瞬时需求，不能当作持续力矩要求；1s/10s滑动RMS用于识别局部高负载。完整CSV同时保留含启动的数据。",
              "- 在相同转矩常数/电阻及冷却条件下，(RMS力矩/额定力矩)²只能作为铜耗相对额定的粗略代理。真实温升还取决于电阻温漂、散热、转速、驱动器和占空比，不能据此批准持续电流或峰值持续时长。",
              "- 若电机换算结果超过峰值，说明原策略在独立关节仿真中要求了硬件无法实现的组合，不能靠放宽实机安全限值补偿。",
              "- 名义单轨迹不覆盖随机推力、跌倒救援、载荷和所有速度指令；没有在该测试中打满不代表峰值设计无意义。",
              "- 执行器驱动力矩不是接触约束反力矩，也不是脚底外力。", "",
              "原始数据：`zero.npz`、`forward06.npz`、`forward10.npz`；电机换算：`*_motor.npz`；统计：`motor_statistics.csv`、`summary.json`；配置/资产/权重校验：`manifest.json`。", ""]
    (root/"README.md").write_text("\n".join(lines))


if __name__=="__main__":main()
