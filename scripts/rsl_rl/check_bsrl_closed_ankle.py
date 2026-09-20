"""Component-level GPU topology/reset/dynamics validation, not PPO training."""
import argparse
import builtins
import sys
import time
from types import SimpleNamespace

_import = builtins.__import__
def _compat(name, globals=None, locals=None, fromlist=(), level=0):
    module = _import(name, globals, locals, fromlist, level)
    if name == "isaaclab.utils.version":
        version = sys.modules.get(name)
        if version is not None and not hasattr(version, "get_isaac_sim_version"):
            version.get_isaac_sim_version = lambda: SimpleNamespace(major=5, minor=1, patch=0)
    return module
builtins.__import__ = _compat
from local_source import prefer_local_source_tree
prefer_local_source_tree(__file__)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--powered", action="store_true")
parser.add_argument("--position_iterations", type=int, default=8)
parser.add_argument("--velocity_iterations", type=int, default=4)
parser.add_argument("--forces_every_iteration", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--no_replication", action="store_true")
parser.add_argument("--amp_env", action="store_true", help="Diagnose the real environment after a failed training launch; no optimizer.")
parser.add_argument("--nominal", action="store_true")
parser.add_argument("--contact_last", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--physics_dt", type=float, default=None)
parser.add_argument("--world_offset", type=float, default=0.)
parser.add_argument("--action_std", type=float, default=.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--inference", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--capture_dir", type=str)
parser.add_argument("--stop_gap", type=float, default=.01)
parser.add_argument("--amp_velocity_iterations", type=int, default=None)
parser.add_argument("--amp_position_iterations", type=int, default=None)
parser.add_argument("--disable_joint_friction", action="store_true")
parser.add_argument("--no_self_collision", action="store_true")
parser.add_argument("--lift", type=float, default=0.)
parser.add_argument("--replay_failure", type=str)
parser.add_argument("--replay_index", type=int, default=0)
parser.add_argument("--replay_center", action="store_true")
parser.add_argument("--solver_type", type=int, choices=(0, 1), default=1)
parser.add_argument("--aux_inertia_scale", type=float, default=1., help="DIAGNOSTIC sensitivity only, not measured hardware inertia.")
parser.add_argument("--no_direct_speed_clamp", action="store_true")
parser.add_argument("--monitor_closure", action="store_true", help="Diagnostic only: log closure distributions/durations instead of stopping at 10 mm.")
parser.add_argument("--hard_stop_gap", type=float, default=.5, help="Catastrophic diagnostic stop in metres, NOT an acceptance tolerance.")
parser.add_argument("--passive_damping", type=float, default=0., help="Diagnostic rod-ball damping; does not change production defaults.")
parser.add_argument("--passive_armature", type=float, default=0., help="Diagnostic rod-ball armature; not a measured hardware parameter.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.monitor_closure and (not args.amp_env or not args.capture_dir or args.replay_failure):
    parser.error("--monitor_closure requires --amp_env --capture_dir and cannot be used with --replay_failure")
if args.passive_damping < 0 or args.passive_armature < 0 or args.hard_stop_gap <= 0:
    parser.error("Diagnostic damping/armature must be nonnegative and hard_stop_gap positive")
launcher = AppLauncher(args)
app = launcher.app
builtins.__import__ = _import

import torch
torch.backends.cuda.matmul.allow_tf32 = args.tf32
torch.backends.cudnn.allow_tf32 = args.tf32
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_from_matrix, matrix_from_quat, quat_apply
from legged_lab.assets.bsrl import BSRL_CFG
from legged_lab.assets.bsrl_closed_ankle_spawn import spawn_bsrl_closed_ankles
from legged_lab.assets.bsrl_closed_ankle_geometry import CADAnkleGeometry
from legged_lab.assets.bsrl_closed_ankle import with_closed_ankles

if args.amp_env:
    import gymnasium as gym
    from legged_lab.tasks.locomotion.amp.config.bsrl.bsrl_amp_env_cfg import BSRLAmpEnvCfg
    ec = BSRLAmpEnvCfg()
    ec.seed = args.seed
    ec.scene.num_envs = args.num_envs
    ec.scene.replicate_physics = not args.no_replication
    ec.sim.device = args.device
    ec.sim.physx.solver_type = args.solver_type
    if args.contact_last is not None:
        ec.sim.physx.solve_articulation_contact_last = args.contact_last
    ec.sim.dt = args.physics_dt or .001
    ec.sim.physx.enable_external_forces_every_iteration = (
        True if args.forces_every_iteration is None else args.forces_every_iteration)
    if args.contact_last is None:
        ec.sim.physx.solve_articulation_contact_last = True
    ec.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/(base_link|link_.*)"
    ec.decimation = round(.02/ec.sim.dt)
    control_decimation = round(.005/ec.sim.dt)
    assert abs(control_decimation*ec.sim.dt-.005) < 1e-10
    ec.scene.robot = with_closed_ankles(BSRL_CFG, control_decimation).replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Isaac's timestamp + seeded-random default can collide when diagnostics
    # with the same seed start in parallel. Only this process uses this cache.
    import tempfile
    ec.scene.robot.spawn.usd_dir = tempfile.mkdtemp(prefix="bsrl_closed_diag_usd_")
    passive = ec.scene.robot.actuators["passive_rod_balls"]
    passive.damping = args.passive_damping
    passive.armature = args.passive_armature
    if args.passive_damping:
        # Keep these joints passive: Kp=0, zero velocity target, damping only.
        passive.effort_limit = passive.effort_limit_sim = 50.
    if args.amp_velocity_iterations is not None:
        ec.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.amp_velocity_iterations
    if args.amp_position_iterations is not None:
        ec.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.amp_position_iterations
    if args.no_self_collision:
        ec.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.no_direct_speed_clamp:
        for name in ("large_motors", "small_motors"):
            ec.scene.robot.actuators[name].velocity_limit_sim = 1000.
    ec.sim.render_interval = ec.decimation
    ec.scene.contact_forces.update_period = ec.sim.dt
    if args.nominal:
        for name in ("physics_material", "base_com", "body_mass", "actuator_gains", "joint_params",
                     "joint_friction", "add_joint_default_pos", "push_robot"):
            setattr(ec.events, name, None)
        for ac in ec.scene.robot.actuators.values():
            if hasattr(ac, "max_delay"):
                ac.min_delay = ac.max_delay = 0
    if args.capture_dir:
        from pathlib import Path
        import json
        from isaaclab.utils.io import dump_yaml
        capture_path = Path(args.capture_dir)
        capture_path.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(capture_path / "env.yaml"), ec)
        (capture_path / "arguments.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    env = gym.make("LeggedLab-Isaac-AMP-BSRL-v0", cfg=ec).unwrapped
    articulation_props = ec.scene.robot.spawn.articulation_props
    print(f"[ClosedAnkleCheck] solver={ec.sim.physx.solver_type} "
          f"iterations={articulation_props.solver_position_iteration_count}/"
          f"{articulation_props.solver_velocity_iteration_count} "
          f"physics_dt={ec.sim.dt} pd_decimation={control_decimation} policy_decimation={ec.decimation} "
          f"contact_last={ec.sim.physx.solve_articulation_contact_last} "
          f"forces_every_iteration={ec.sim.physx.enable_external_forces_every_iteration} "
          f"ankle_sim_speed_guard={ec.scene.robot.actuators['closed_ankles'].velocity_limit_sim} "
          f"clock_callbacks={len(env._before_physics_callbacks)} "
          f"actuator_class={type(env.scene['robot'].actuators['closed_ankles']).__name__}", flush=True)
    env.scene.env_origins[:, 0] += args.world_offset
    env.scene.env_origins[:, 2] += args.lift
    if args.replay_failure:
        from closed_ankle_diagnostics import replay_failure
        replay_failure(env, args.replay_failure, args.replay_index, args.replay_center, args.lift,
                       args.aux_inertia_scale, args.no_direct_speed_clamp)
        env.close()
        app.close(wait_for_replicator=False)
        sys.exit(0)
    if args.disable_joint_friction:
        robot = env.scene["robot"]
        robot.write_joint_friction_coefficient_to_sim(
            torch.zeros_like(robot.data.joint_friction_coeff),
            joint_dynamic_friction_coeff=torch.zeros_like(robot.data.joint_friction_coeff),
            joint_viscous_friction_coeff=torch.zeros_like(robot.data.joint_friction_coeff))
    if args.monitor_closure:
        from closed_ankle_diagnostics import ClosureDurationRecorder
        recorder = ClosureDurationRecorder(env, args.capture_dir, args.hard_stop_gap)
    elif args.capture_dir:
        from closed_ankle_diagnostics import EarlyClosureRecorder
        recorder = EarlyClosureRecorder(env, args.capture_dir, args.stop_gap)
    generator = torch.Generator(device=args.device).manual_seed(args.seed+1000)
    print(f"[ClosedAnkleCheck] seed={args.seed} tf32={args.tf32} inference={args.inference}", flush=True)
    with torch.inference_mode(args.inference):
        obs, info = env.reset()
        robot = env.scene["robot"]
        rod_offset = quat_apply(robot.data.body_quat_w[:, robot.ankle_rod_ids], robot._rod_end_local)
        foot_offset = quat_apply(
            robot.data.body_quat_w[:, robot.ankle_foot_ids].unsqueeze(-2).expand(-1, -1, 2, -1),
            robot.ankle_geometry.mounts.unsqueeze(0).expand(env.num_envs, -1, -1, -1))
        rod_velocity = (robot.data.body_link_lin_vel_w[:, robot.ankle_rod_ids]
                        + torch.linalg.cross(robot.data.body_ang_vel_w[:, robot.ankle_rod_ids], rod_offset))
        foot_velocity = (robot.data.body_link_lin_vel_w[:, robot.ankle_foot_ids].unsqueeze(-2)
                         + torch.linalg.cross(robot.data.body_ang_vel_w[:, robot.ankle_foot_ids].unsqueeze(-2), foot_offset))
        print(f"[ClosedAnkleCheck] initial_reset_gap_mm={robot.closure_gaps().max().item()*1000:.6f} "
              f"initial_reset_velocity_error_m_s={(rod_velocity-foot_velocity).norm(dim=-1).max().item():.9f}", flush=True)
        rollout_started = time.perf_counter()
        try:
            for i in range(args.steps):
                env.step(torch.randn((args.num_envs, 12), device=args.device, generator=generator)*args.action_std)
                if i % 20 == 0:
                    print("[ClosedAnkleCheck] AMP_ENV step", i, "maxgap mm", env.scene["robot"].closure_gaps().max().item()*1000, flush=True)
        except Exception as exc:
            if args.monitor_closure:
                recorder.finish("stopped", str(exc))
            raise
        print(f"[ClosedAnkleCheck] rollout_seconds={time.perf_counter()-rollout_started:.6f} "
              f"policy_steps={args.steps} physics_steps={args.steps*ec.decimation} "
              f"monitor_closure={args.monitor_closure}", flush=True)
    if args.monitor_closure:
        recorder.finish("complete")
    print("[ClosedAnkleCheck] AMP_ENV COMPLETE", flush=True)
    env.close()
    app.close()
    sys.exit(0)

args.physics_dt = args.physics_dt or .005
sim_cfg = sim_utils.SimulationCfg(dt=args.physics_dt, device=args.device, gravity=(0., 0., 0.))
sim_cfg.physx.enable_external_forces_every_iteration = bool(args.forces_every_iteration)
sim_cfg.physx.solve_articulation_contact_last = bool(args.contact_last)
sim = sim_utils.SimulationContext(sim_cfg)
cfg = BSRL_CFG.replace(prim_path="/World/Robot")
cfg.spawn.func = spawn_bsrl_closed_ankles
cfg.spawn.activate_contact_sensors = False
cfg.spawn.fix_base = True
if args.powered:
    cfg = with_closed_ankles(cfg)
else:
    cfg.actuators = {"all": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0, damping=0,
                                            effort_limit=1e6, effort_limit_sim=1e6,
                                            velocity_limit=1000, velocity_limit_sim=1000, armature=.01)}
cfg.init_state.joint_pos = {".*": 0.0}
cfg.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
if args.num_envs > 1:
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    @configclass
    class CheckSceneCfg(InteractiveSceneCfg):
        robot = cfg
    scene = InteractiveScene(CheckSceneCfg(num_envs=args.num_envs, env_spacing=2.5,
                                          replicate_physics=not args.no_replication))
    robot = scene["robot"]
else:
    robot = cfg.class_type(cfg)
sim.reset()
print("[ClosedAnkleCheck] JOINTS", robot.joint_names, flush=True)
print("[ClosedAnkleCheck] BODIES", robot.body_names, flush=True)
print("[ClosedAnkleCheck] STATE", robot.data.joint_pos, flush=True)
geometry = CADAnkleGeometry(device=args.device)
q_ids = [[robot.joint_names.index(f"joint_{s}_ankle_{a}") for a in ("pitch", "roll")] for s in ("left", "right")]
m_ids = [[robot.joint_names.index(f"motor_{s}_{b}") for b in ("long", "short")] for s in ("left", "right")]
b_ids = [[[robot.joint_names.index(f"rod_ball_{s}_{b}:{k}") for k in range(3)]
          for b in ("long", "short")] for s in ("left", "right")]
rod_ids = [[robot.body_names.index(f"rod_{s}_{b}") for b in ("long", "short")] for s in ("left", "right")]
foot_ids = [robot.body_names.index(f"link_{s}_ankle_roll") for s in ("left", "right")]
q = torch.tensor([[[.3, .25], [-.4, -.35]]], device=args.device).expand(args.num_envs, -1, -1).clone()
motor, valid = geometry.inverse_kinematics(q)
ball_rot = geometry.rod_relative_rotation(q, motor)
ball_pos = axis_angle_from_quat(quat_from_matrix(ball_rot))
pos = torch.zeros_like(robot.data.joint_pos)
pos[:, q_ids] = q
pos[:, m_ids] = motor
pos[:, b_ids] = ball_pos
robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
sim.forward()
rod_end_local = torch.zeros((args.num_envs, 2, 2, 3), device=args.device)
rod_end_local[..., 2] = geometry.rod_lengths
def gaps():
    rod_end = robot.data.body_pos_w[:, rod_ids] + quat_apply(robot.data.body_quat_w[:, rod_ids], rod_end_local)
    foot_end = robot.data.body_pos_w[:, foot_ids].unsqueeze(-2) + quat_apply(
        robot.data.body_quat_w[:, foot_ids].unsqueeze(-2).expand(-1, -1, 2, -1), geometry.mounts.unsqueeze(0).expand(args.num_envs,-1,-1,-1))
    return (rod_end-foot_end).norm(dim=-1)
print("[ClosedAnkleCheck] NONZERO assembled closure m", gaps(), flush=True)
print("[ClosedAnkleCheck] BALL desired", ball_pos, flush=True)
dq = torch.tensor([[[1.2, -.8], [-.7, 1.1]]], device=args.device).expand(args.num_envs, -1, -1).clone()
dm = (geometry.jacobian(q, motor) @ dq.unsqueeze(-1)).squeeze(-1)
eps = 1e-3
qp, qm = q+eps*dq, q-eps*dq
mp, _ = geometry.inverse_kinematics(qp)
mm, _ = geometry.inverse_kinematics(qm)
rp, rm = geometry.rod_relative_rotation(qp, mp), geometry.rod_relative_rotation(qm, mm)
rdot = (rp-rm)/(2*eps)
omega_parent = rdot @ ball_rot.transpose(-1, -2)
omega_child = ball_rot.transpose(-1, -2) @ rdot
def vee(skew):
    return torch.stack((skew[...,2,1]-skew[...,1,2], skew[...,0,2]-skew[...,2,0],
                        skew[...,1,0]-skew[...,0,1]), -1)*.5
for label, bv in (("exp_derivative", (axis_angle_from_quat(quat_from_matrix(rp))-axis_angle_from_quat(quat_from_matrix(rm)))/(2*eps)),
                  ("omega_parent", vee(omega_parent)), ("omega_child", vee(omega_child)),
                  ("analytic_child", geometry.reset_velocity(q, dq, motor)[1])):
    vel = torch.zeros_like(pos)
    vel[:, q_ids] = dq
    vel[:, m_ids] = dm
    vel[:, b_ids] = bv
    # Deliberately bypass the production assembly for the convention comparison.
    Articulation.write_joint_state_to_sim(robot, pos, vel)
    sim.forward()
    rod_offset = quat_apply(robot.data.body_quat_w[:, rod_ids], rod_end_local)
    foot_offset = quat_apply(robot.data.body_quat_w[:, foot_ids].unsqueeze(-2).expand(-1,-1,2,-1), geometry.mounts.unsqueeze(0).expand(args.num_envs,-1,-1,-1))
    rod_vel = robot.data.body_link_lin_vel_w[:, rod_ids] + torch.linalg.cross(robot.data.body_ang_vel_w[:, rod_ids], rod_offset)
    foot_vel = robot.data.body_link_lin_vel_w[:, foot_ids].unsqueeze(-2) + torch.linalg.cross(
        robot.data.body_ang_vel_w[:, foot_ids].unsqueeze(-2).expand(-1,-1,2,-1), foot_offset)
    print("[ClosedAnkleCheck] VELOCITY convention", label, "residual m/s", (rod_vel-foot_vel).norm(dim=-1), flush=True)
for step in range(args.steps):
    if args.powered:
        target = torch.zeros_like(pos)
        target[:, q_ids] = q * torch.cos(torch.tensor(step*.005*3, device=args.device))
        robot.set_joint_position_target(target)
        if step % 100 == 0:
            robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            robot.reset()
            print("[ClosedAnkleCheck] POWERED RESET gap m", gaps(), flush=True)
    if hasattr(robot, "begin_physics_step"):
        robot.begin_physics_step()
    robot.write_data_to_sim()
    sim.step()
    robot.update(args.physics_dt)
    if step in (0, args.steps-1):
        print("[ClosedAnkleCheck] STEP", step, robot.data.joint_pos, "GAPS", gaps(), flush=True)
print("[ClosedAnkleCheck] COMPLETE", flush=True)
sim.stop()
sim.clear()
app.close(wait_for_replicator=False)
