"""Closed-loop assembly on reset and runtime audit of the physical constraints."""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import axis_angle_from_quat, quat_from_matrix, quat_apply
from legged_lab.assets.bsrl_closed_ankle_geometry import CADAnkleGeometry


class ClosedAnkleArticulation(Articulation):
    def begin_physics_step(self):
        """Permit one actuator evaluation for the next actual simulation step.

        AMP also writes the scene for reset/forward kinematics without stepping
        physics. Those writes must not tick a held PD controller or its delay.
        Standalone component simulations must call this before write_data_to_sim.
        """
        self._physics_step_pending = True

    def _apply_actuator_model(self):
        if getattr(self, "_physics_step_pending", False):
            self._physics_step_pending = False
            super()._apply_actuator_model()

    def _initialize_impl(self):
        super()._initialize_impl()
        self.ankle_geometry = CADAnkleGeometry(device=self.device)
        self.ankle_q_ids = [[self.joint_names.index(f"joint_{s}_ankle_{a}") for a in ("pitch", "roll")]
                           for s in ("left", "right")]
        self.ankle_motor_ids = [[self.joint_names.index(f"motor_{s}_{b}") for b in ("long", "short")]
                               for s in ("left", "right")]
        self.ankle_ball_ids = [[[self.joint_names.index(f"rod_ball_{s}_{b}:{k}") for k in range(3)]
                               for b in ("long", "short")] for s in ("left", "right")]
        self.ankle_rod_ids = [[self.body_names.index(f"rod_{s}_{b}") for b in ("long", "short")]
                             for s in ("left", "right")]
        self.ankle_foot_ids = [self.body_names.index(f"link_{s}_ankle_roll") for s in ("left", "right")]
        self._rod_end_local = torch.zeros((self.num_instances, 2, 2, 3), device=self.device)
        self._rod_end_local[..., 2] = self.ankle_geometry.rod_lengths
        self._assemble(self.data.default_joint_pos, self.data.default_joint_vel)
        self._audit_step = 0
        self._audit_reset_count = 0
        self._audit = torch.zeros(7, device=self.device)
        self._audit_gap_totals = torch.zeros(2, device=self.device)
        self._audit_total_closure_max = 0.0
        self._audit_peak_state = torch.zeros(1+7+2*self.num_joints, device=self.device)
        self._ankle_ready = True

    def _assemble(self, pos, vel):
        q, dq = pos[:, self.ankle_q_ids], vel[:, self.ankle_q_ids]
        motor, valid = self.ankle_geometry.inverse_kinematics(q)
        if not bool(valid.all()) or not bool(torch.isfinite(q).all() & torch.isfinite(dq).all()):
            raise RuntimeError("Closed ankle reset received an unreachable/non-finite reference pose or velocity.")
        ball = axis_angle_from_quat(quat_from_matrix(self.ankle_geometry.rod_relative_rotation(q, motor)))
        dm, ball_velocity = self.ankle_geometry.reset_velocity(q, dq, motor)
        pos[:, self.ankle_motor_ids] = motor
        pos[:, self.ankle_ball_ids] = ball
        vel[:, self.ankle_motor_ids] = dm
        vel[:, self.ankle_ball_ids] = ball_velocity

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        if not getattr(self, "_ankle_ready", False):
            return super().write_joint_state_to_sim(position, velocity, joint_ids, env_ids)
        rows = slice(None) if env_ids is None else env_ids
        cols = slice(None) if joint_ids is None else joint_ids
        pos, vel = self.data.joint_pos[rows].clone(), self.data.joint_vel[rows].clone()
        pos[:, cols], vel[:, cols] = position, velocity
        self._assemble(pos, vel)
        # One complete, constraint-consistent state write. Never teleport links
        # during the physics rollout to conceal a disconnected mechanism.
        super().write_joint_state_to_sim(pos, vel, joint_ids=None, env_ids=env_ids)
        self._audit_reset_count += pos.shape[0]

    def closure_gaps(self):
        rod_end = self.data.body_pos_w[:, self.ankle_rod_ids] + quat_apply(
            self.data.body_quat_w[:, self.ankle_rod_ids], self._rod_end_local)
        foot_end = self.data.body_pos_w[:, self.ankle_foot_ids].unsqueeze(-2) + quat_apply(
            self.data.body_quat_w[:, self.ankle_foot_ids].unsqueeze(-2).expand(-1, -1, 2, -1),
            self.ankle_geometry.mounts.unsqueeze(0).expand(self.num_instances, -1, -1, -1))
        return (rod_end-foot_end).norm(dim=-1)

    def update(self, dt):
        super().update(dt)
        if dt == 0. or not getattr(self, "_ankle_ready", False) or "closed_ankles" not in self.actuators:
            return
        actuator = self.actuators["closed_ankles"]
        gaps = self.closure_gaps()
        env_gap = gaps.amax(dim=(1, 2))
        self._audit_gap_totals += torch.stack((env_gap.sum(), (env_gap > .01).sum()))
        # Accumulate on GPU, including ALL decimation substeps and states before
        # termination/reset; a policy-step-only audit can hide closure failures.
        sample = torch.stack((
            gaps.max(), actuator.motor_torque.abs().max(),
            self.data.applied_torque[:, self.ankle_q_ids].abs().max(),
            (~actuator.jacobian_valid).sum().float(),
            (~torch.isfinite(self.data.joint_pos)).sum().float()
            + (~torch.isfinite(self.data.joint_vel)).sum().float()
            + (~torch.isfinite(gaps)).sum().float(),
            (actuator.motor_torque*actuator.motor_velocity).abs().max(),
            actuator.motor_velocity.abs().max(),
        ))
        worst_env = gaps.flatten().argmax() // 4
        peak_state = torch.cat((worst_env.reshape(1).float(), self.data.root_pose_w[worst_env],
                                self.data.joint_pos[worst_env], self.data.joint_vel[worst_env]))
        self._audit_peak_state = torch.where(sample[0] > self._audit[0], peak_state, self._audit_peak_state)
        self._audit = torch.maximum(self._audit, sample)
        self._audit_step += 1
        # 24 policy steps at 50 Hz per PPO collection, independent of substeps.
        audit_interval = round(.48/dt)
        if self._audit_step % audit_interval == 0:
            gap, torque, passive, singular, nonfinite, power, speed = self._audit.tolist()
            gap_sum, over10_count = self._audit_gap_totals.tolist()
            gap_mean_mm = gap_sum / (audit_interval * self.num_instances) * 1000
            gap_over10_percent = over10_count / (audit_interval * self.num_instances) * 100
            self._audit_total_closure_max = max(self._audit_total_closure_max, gap)
            print(f"[ClosedAnkleAudit] physics_steps={self._audit_step} resets={self._audit_reset_count} "
                  f"closure_max_mm={gap*1000:.6f} closure_alltime_mm={self._audit_total_closure_max*1000:.6f} "
                  f"closure_mean_mm={gap_mean_mm:.6f} closure_over10_percent={gap_over10_percent:.6f} "
                  f"gap_monitor_only=1 "
                  f"motor_peak_Nm={torque:.6f} passive_direct_Nm={passive:.6f} "
                  f"singular={singular:.0f} nonfinite={nonfinite:.0f} "
                  f"motor_power_peak_W={power:.6f} motor_speed_peak_rad_s={speed:.6f}", flush=True)
            # User-requested residual-tolerant experiment: finite closure gaps
            # are telemetry, NOT a stop/reset/reward threshold. Keep the old
            # criterion for traceability: ... or gap > .01
            # Nonfinite states, invalid mapping and actuator violations remain
            # real failures; accepting solver residual is not accepting NaNs.
            if nonfinite or singular or passive > 1e-6 or torque > 112.001 or power > 1250.01:
                print("[ClosedAnkleAudit] current gaps first16 mm", (gaps[:16]*1000).tolist(), flush=True)
                print("[ClosedAnkleAudit] peak env/root/q/dq", self._audit_peak_state.tolist(), flush=True)
                print("[ClosedAnkleAudit] joint_names", self.joint_names, flush=True)
                raise RuntimeError("Closed ankle physical audit failed; inspect ClosedAnkleAudit (only this run stops).")
            self._audit.zero_()
            self._audit_gap_totals.zero_()
