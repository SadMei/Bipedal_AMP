import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="View the BSRL asset in its configured default pose.")
parser.add_argument("--usd_path", default=None, help="Optional USD asset override.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab.assets.bsrl import BSRL_CFG


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view(eye=(2.2, 2.2, 1.4), target=(0.0, 0.0, 0.55))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = BSRL_CFG.replace(prim_path="/World/Robot")
    if args.usd_path is not None:
        robot_cfg.spawn.usd_path = args.usd_path
    robot = Articulation(robot_cfg)
    sim.reset()

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    print("BSRL GMR-derived default pose loaded. Close the Isaac window to exit.", flush=True)
    while simulation_app.is_running():
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        robot.update(sim.get_physics_dt())
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
    simulation_app.close()
