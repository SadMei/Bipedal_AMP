"""Opt-in early-failure recorder. No optimizer and no changes to physics state."""
from pathlib import Path
import csv
import json
import time

import torch


class EarlyClosureRecorder:
    def __init__(self, env, directory, threshold=.01, history=32):
        self.env = env
        self.robot = robot = env.scene["robot"]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.history_length = history
        self.step = 0
        self.initial = None
        self.samples = None
        self.max_gap = 0.
        self.reset_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.int64)
        self.last_reset = None
        original_write = robot.write_joint_state_to_sim
        original_update = robot.update

        def write(position, velocity, joint_ids=None, env_ids=None):
            original_write(position, velocity, joint_ids, env_ids)
            ids = slice(None) if env_ids is None else env_ids
            state = self.state()
            if self.last_reset is None:
                self.last_reset = state.clone()
            else:
                self.last_reset[ids] = state[ids]
            self.reset_step[ids] = self.step

        def update(dt):
            original_update(dt)
            self.observe()

        robot.write_joint_state_to_sim = write
        robot.update = update

    def state(self):
        robot = self.robot
        ankle = robot.actuators["closed_ankles"]
        values = {
            "root_state_w": robot.data.root_state_w,
            "joint_pos": robot.data.joint_pos,
            "joint_vel": robot.data.joint_vel,
            "applied_torque": robot.data.applied_torque,
            "joint_target": robot.data.joint_pos_target,
            "requested_motor": ankle._requested_motor.flatten(1),
            "closure_gaps": robot.closure_gaps().flatten(1),
            "action": self.env.action_manager.action,
        }
        self.fields = {key: value.shape[1] for key, value in values.items()}
        return torch.cat(tuple(values.values()), dim=1)

    def observe(self):
        state = self.state()
        if self.samples is None:
            self.samples = torch.empty((self.history_length, *state.shape), device=state.device)
            self.initial = state.clone()
        self.samples[self.step % self.history_length].copy_(state)
        self.step += 1
        gap = self.robot.closure_gaps().flatten(1).max(dim=1).values
        maximum = gap.max().item()
        self.max_gap = max(self.max_gap, maximum)
        if self.step <= 5 or self.step % 100 == 0 or maximum > self.threshold:
            print(f"[EarlyClosure] substep={self.step} max_mm={maximum*1000:.6f} "
                  f"env={(gap.argmax()).item()} alltime_mm={self.max_gap*1000:.6f}", flush=True)
        if maximum > self.threshold or not torch.isfinite(state).all():
            self.save(gap)
            raise RuntimeError("Diagnostic stopped at FIRST closure threshold crossing; see failure.pt.")

    def save(self, gap):
        ids = torch.topk(gap, min(16, gap.numel())).indices
        count = min(self.step, self.history_length)
        times = torch.arange(self.step-count, self.step, device=self.env.device)
        selected = self.samples[times % self.history_length][:, ids]
        robot = self.robot
        properties = {}
        for name in ("masses", "inertias", "coms", "material_properties", "dof_limits",
                     "dof_armatures", "dof_friction_properties", "dof_max_velocities"):
            try:
                value = getattr(robot.root_physx_view, "get_"+name)()
                properties[name] = value[ids.to(value.device)].cpu()
            except (AttributeError, RuntimeError) as exc:
                properties[name] = str(exc)
        actuators = {}
        for name, actuator in robot.actuators.items():
            values = {key: getattr(actuator, key)[ids].cpu() for key in
                      ("stiffness", "damping", "armature", "friction")}
            values["joint_indices"] = actuator.joint_indices
            for key in ("_control_tick",):
                values[key] = getattr(actuator, key, None)
            if hasattr(actuator, "positions_delay_buffer"):
                values["delay"] = actuator.positions_delay_buffer.time_lags[ids].cpu()
            actuators[name] = values
        payload = {
            "fields": self.fields, "physics_steps": times.cpu()+1, "env_ids": ids.cpu(),
            "states": selected.cpu(), "initial": self.initial[ids].cpu(),
            "last_reset": self.last_reset[ids].cpu() if self.last_reset is not None else None,
            "reset_step": self.reset_step[ids].cpu(), "properties": properties,
            "actuators": actuators, "joint_names": robot.joint_names, "body_names": robot.body_names,
            "env_origins": self.env.scene.env_origins[ids].cpu(), "dt": self.env.physics_dt,
            "seed": self.env.cfg.seed, "tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        torch.save(payload, self.directory / "failure.pt")
        print(f"[EarlyClosure] saved {self.directory / 'failure.pt'}", flush=True)


class ClosureStatistics:
    """Per-environment maxima; resets censor a streak, never count as recovery."""

    def __init__(self, num_envs, device):
        self.steps = 0
        self.over10_samples = 0
        self.recovered = 0
        self.reset_censored = 0
        self.resets = 0
        self.longest_steps = 0
        self.max_gap = 0.
        self.ever10 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.streak = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def reset(self, ids):
        self.reset_censored += int((self.streak[ids] > 0).sum())
        self.resets += self.streak[ids].numel()
        self.streak[ids] = 0

    def observe(self, gap):
        over = gap > .01
        self.recovered += int(((self.streak > 0) & ~over).sum())
        self.streak = torch.where(over, self.streak + 1, 0)
        self.ever10 |= over
        self.over10_samples += int(over.sum())
        self.longest_steps = max(self.longest_steps, int(self.streak.max()))
        self.max_gap = max(self.max_gap, float(gap.max()))
        self.steps += 1

    def summary(self, dt):
        return {
            "physics_steps": self.steps,
            "simulated_seconds": self.steps * dt,
            "max_gap_mm": self.max_gap * 1000,
            "env_time_over10_percent": 100 * self.over10_samples / max(1, self.steps * self.streak.numel()),
            "ever_over10_env_percent": float(self.ever10.float().mean()) * 100,
            "longest_over10_ms": self.longest_steps * dt * 1000,
            "recovered_without_reset_streaks": self.recovered,
            "reset_censored_streaks": self.reset_censored,
            "still_open_over10_streaks": int((self.streak > 0).sum()),
            "reset_env_count_including_initial": self.resets,
        }


class ClosureDurationRecorder(EarlyClosureRecorder):
    """Diagnostic-only replacement audit: do not stop at the first 10 mm gap.

    This instance uses Articulation.update for buffer refresh, skipping only the
    production audit. Assembly, PD clock, forces, solver, and normal episode
    resets are unchanged. Every physical substep is sampled BEFORE any reset.
    No physical state is projected or sanitized. Extreme states still stop the
    diagnostic; the 0.5 m emergency bound is NOT a passing tolerance.
    """

    def __init__(self, env, directory, hard_stop_gap=.5):
        super().__init__(env, directory, threshold=hard_stop_gap)
        from isaaclab.assets import Articulation
        self.stats = ClosureStatistics(env.num_envs, env.device)
        self.started = time.monotonic()
        self.finished = False
        self.watch_ids = None
        self.trace = []
        self.last_row = None
        self.csv_file = (self.directory / "substeps.csv").open("w", newline="")
        self.writer = None
        self.active_ids = [i for i, name in enumerate(self.robot.joint_names)
                           if name.startswith(("joint_", "motor_"))]
        previous_write = self.robot.write_joint_state_to_sim

        def write(position, velocity, joint_ids=None, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            self.stats.reset(ids)
            previous_write(position, velocity, joint_ids, env_ids)

        def update(dt):
            Articulation.update(self.robot, dt)
            self.observe()

        self.robot.write_joint_state_to_sim = write
        self.robot.update = update
        print(f"[ClosureMonitor] diagnostic-only audit; log 10mm excursions, "
              f"hard_stop_gap={hard_stop_gap}m; production defaults unchanged", flush=True)

    def observe(self):
        robot, env = self.robot, self.env
        state = self.state()
        if self.samples is None:
            self.samples = torch.empty((self.history_length, *state.shape), device=state.device)
            self.initial = state.clone()
        self.samples[self.step % self.history_length].copy_(state)
        self.step += 1
        raw_gap = robot.closure_gaps().flatten(1).max(dim=1).values
        nonfinite = int((~torch.isfinite(state).all(dim=1)).sum())
        # Only sanitize STATISTICS so a failure can be serialized, never state.
        gap = torch.nan_to_num(raw_gap, nan=self.threshold, posinf=self.threshold, neginf=self.threshold)
        self.stats.observe(gap)
        if self.watch_ids is None:
            self.watch_ids = torch.topk(gap, min(16, env.num_envs)).indices
        ankle = robot.actuators["closed_ankles"]
        quantiles = torch.quantile(gap, torch.tensor([.5, .95, .99], device=gap.device)).tolist()
        row = {
            "step": self.step, "time_s": self.step * env.physics_dt,
            "max_mm": float(gap.max()) * 1000, "mean_mm": float(gap.mean()) * 1000,
            "p50_mm": quantiles[0] * 1000, "p95_mm": quantiles[1] * 1000, "p99_mm": quantiles[2] * 1000,
            "over1_envs": int((gap > .001).sum()), "over5_envs": int((gap > .005).sum()),
            "over10_envs": int((gap > .01).sum()), "over33_envs": int((gap > .033).sum()),
            "over10_current_streak_max_ms": int(self.stats.streak.max()) * env.physics_dt * 1000,
            "reset_count": self.stats.resets, "recovered_streaks": self.stats.recovered,
            "reset_censored_streaks": self.stats.reset_censored,
            "active_speed_max": float(robot.data.joint_vel[:, self.active_ids].abs().max()),
            "all_joint_speed_max": float(robot.data.joint_vel.abs().max()),
            "root_linear_speed_max": float(robot.data.root_lin_vel_b.abs().max()),
            "root_angular_speed_max": float(robot.data.root_ang_vel_b.abs().max()),
            "motor_torque_max": float(ankle.motor_torque.abs().max()),
            "motor_power_max": float((ankle.motor_torque * ankle.motor_velocity).abs().max()),
            "passive_q_torque_max": float(robot.data.applied_torque[:, robot.ankle_q_ids].abs().max()),
            "rod_damping_torque_max": float(robot.actuators['passive_rod_balls'].applied_effort.abs().max()),
            "singular_ankles": int((~ankle.jacobian_valid).sum()), "nonfinite_envs": nonfinite,
        }
        self.last_row = row
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.trace.append(torch.stack((gap[self.watch_ids], self.reset_step[self.watch_ids].float()), dim=1).cpu())
        if self.step <= 10 or self.step % 100 == 0:
            self.csv_file.flush()
            print(f"[ClosureMonitor] step={self.step} max={row['max_mm']:.3f}mm "
                  f"p99={row['p99_mm']:.3f}mm over10={row['over10_envs']}/{env.num_envs} "
                  f"streak={row['over10_current_streak_max_ms']:.1f}ms "
                  f"active_speed={row['active_speed_max']:.2f}rad/s", flush=True)
        reasons = []
        for condition, reason in (
            (nonfinite, "nonfinite state"), (row['singular_ankles'], "singular ankle Jacobian"),
            (float(gap.max()) > self.threshold, "catastrophic gap bound"),
            (row['active_speed_max'] > 500, "active joint speed >500rad/s"),
            (row['root_linear_speed_max'] > 100, "root linear component >100m/s"),
            (row['root_angular_speed_max'] > 200, "root angular component >200rad/s"),
            (row['motor_torque_max'] > 112.001, "motor torque limit violation"),
            (row['motor_power_max'] > 1250.01, "motor power limit violation"),
            (row['passive_q_torque_max'] > 1e-6, "direct roll/pitch torque"),
        ):
            if condition:
                reasons.append(reason)
        if reasons:
            self.save(gap)
            raise RuntimeError("ClosureMonitor safety stop: " + "; ".join(reasons))

    def finish(self, status, reason=None):
        if self.finished:
            return
        self.finished = True
        self.csv_file.close()
        result = self.stats.summary(self.env.physics_dt)
        result.update(status=status, reason=reason, wall_seconds=time.monotonic()-self.started,
                      last_substep=self.last_row, hard_stop_gap_m=self.threshold,
                      note="Completion is not physical acceptance; no optimizer. Quantiles are over per-env max of four closures.")
        # NaN in a failure's diagnostic scalars is encoded as null in JSON.
        def clean(value):
            import math
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return value
        result = clean(result)
        (self.directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        if self.trace:
            torch.save({"env_ids": self.watch_ids.cpu(), "gap_m_and_last_reset_step": torch.stack(self.trace),
                        "dt": self.env.physics_dt}, self.directory / "first_peak_trajectories.pt")
        print("[ClosureMonitor] SUMMARY " + json.dumps(result), flush=True)


def replay_failure(env, path, index=0, center=False, lift=0., aux_inertia_scale=1., no_direct_speed_clamp=False):
    """Replay recorded generalized forces to isolate physics, not a policy test.

    Restore exact assembled reset state without recomputing ball coordinates.
    Intended for traces captured before the first post-startup episode reset.
    """
    from isaaclab.assets import Articulation
    payload = torch.load(path, map_location="cpu", weights_only=False)
    robot = env.scene["robot"]
    assert robot.joint_names == payload["joint_names"]
    assert payload["reset_step"][index] == 0 and payload["physics_steps"][0] == 1
    assert abs(env.physics_dt-payload["dt"]) < 1e-10
    slices, offset = {}, 0
    for key, size in payload["fields"].items():
        slices[key] = slice(offset, offset+size)
        offset += size
    env.reset()
    for name, value in payload["properties"].items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Cannot restore property {name}: {value}")
        if name == "dof_friction_coefficients":
            raise RuntimeError("Old capture lacks Isaac Sim 5 joint friction properties; capture again.")
        existing = getattr(robot.root_physx_view, "get_"+name)()
        saved = value[index:index+1].expand(env.num_envs, *value.shape[1:]).clone().to(existing.device)
        if name == "dof_limits":
            # Restoring +/-FLT_MAX through setLimitParams is illegal for FREE
            # spherical axes. The same asset already has exactly these limits.
            torch.testing.assert_close(existing, saved)
            continue
        if name == "inertias" and aux_inertia_scale != 1.:
            aux_ids = [i for i, n in enumerate(robot.body_names) if n.startswith(("rod_", "crank_"))]
            saved[:, aux_ids] *= aux_inertia_scale
        if name == "dof_max_velocities" and no_direct_speed_clamp:
            direct_ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("joint_") and "ankle" not in n]
            saved[:, direct_ids] = 1000.
        getattr(robot.root_physx_view, "set_"+name)(saved, robot._ALL_INDICES.to(existing.device))
    reset = payload["last_reset"][index:index+1].to(env.device).expand(env.num_envs, -1).clone()
    root = reset[:, slices["root_state_w"]].clone()
    if center:
        root[:, :2] -= root[:1, :2].clone()
        root[:, :2] += env.scene.env_origins[:, :2]
    else:
        root[:, :2] += env.scene.env_origins[:, :2]-env.scene.env_origins[:1, :2]
    root[:, 2] += lift
    robot.write_root_state_to_sim(root)
    Articulation.write_joint_state_to_sim(robot, reset[:, slices["joint_pos"]], reset[:, slices["joint_vel"]])
    env.sim.forward()
    print(f"[ClosureReplay] source_env={payload['env_ids'][index].item()} clones={env.num_envs} "
          f"center={center} lift={lift} DIAGNOSTIC_inertia_scale={aux_inertia_scale} "
          f"reset_gap_mm={robot.closure_gaps().max().item()*1000:.6f}", flush=True)
    for step, sample in zip(payload["physics_steps"], payload["states"][:, index]):
        torque = sample[slices["applied_torque"]].to(env.device).expand(env.num_envs, -1).contiguous()
        robot.root_physx_view.set_dof_actuation_forces(torque, robot._ALL_INDICES)
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        gap = robot.closure_gaps().max().item()*1000
        dq = robot.data.joint_vel.abs().max().item()
        original_gap = sample[slices["closure_gaps"]].max().item()*1000
        print(f"[ClosureReplay] step={step.item()} gap_mm={gap:.6f} recorded_gap_mm={original_gap:.6f} "
              f"max_dq={dq:.6f}", flush=True)
