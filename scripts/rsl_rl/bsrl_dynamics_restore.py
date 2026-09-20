"""Restore the saved BSRL plant, not merely its identically named URDF."""


def restore_bsrl_dynamics(env_cfg, saved_env):
    saved_robot = saved_env.get("scene", {}).get("robot", {})
    saved_actuators = saved_robot.get("actuators", {})
    special = {"closed_ankles", "parallel_ankles"}
    if not (special.intersection(env_cfg.scene.robot.actuators) or special.intersection(saved_actuators)):
        return

    from legged_lab.assets.bsrl import BSRL_CFG

    saved_sim = saved_env.get("sim", {})
    saved_physx = saved_sim.get("physx", {})
    env_cfg.sim.dt = float(saved_sim.get("dt", .005))
    env_cfg.decimation = int(saved_env.get("decimation", 4))
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.physx.solver_type = int(saved_physx.get("solver_type", 1))
    for flag in ("enable_external_forces_every_iteration", "solve_articulation_contact_last"):
        setattr(env_cfg.sim.physx, flag, str(saved_physx.get(flag, "false")).lower() == "true")
    prim_path = env_cfg.scene.robot.prim_path
    if "closed_ankles" in saved_actuators:
        from legged_lab.assets.bsrl_closed_ankle import with_closed_ankles
        pd_decimation = int(saved_actuators["closed_ankles"].get("control_decimation", 1))
        robot_cfg = with_closed_ankles(BSRL_CFG, pd_decimation)
        topology = "explicit physical loops"
    elif "parallel_ankles" in saved_actuators:
        from legged_lab.assets.bsrl_reduced_ankle import with_reduced_ankles
        robot_cfg = with_reduced_ankles(BSRL_CFG, env_cfg.sim.dt)
        topology = "serial skeleton + analytical motor-limited transmission"
    else:
        robot_cfg = BSRL_CFG.copy()
        topology = "legacy direct-PD serial ankles"
    env_cfg.scene.robot = robot_cfg.replace(prim_path=prim_path)
    saved_props = saved_robot.get("spawn", {}).get("articulation_props", {})
    for key in ("solver_position_iteration_count", "solver_velocity_iteration_count"):
        if key in saved_props:
            setattr(env_cfg.scene.robot.spawn.articulation_props, key, int(saved_props[key]))
    env_cfg.scene.contact_forces.prim_path = (
        "{ENV_REGEX_NS}/Robot/(base_link|link_.*)" if "closed_ankles" in saved_actuators
        else "{ENV_REGEX_NS}/Robot/.*")
    env_cfg.scene.contact_forces.update_period = env_cfg.sim.dt

    # Current source switches must never leak new DR into old videos.
    env_cfg.events.ankle_transmission = None
    saved_event = saved_env.get("events", {}).get("ankle_transmission")
    if "parallel_ankles" in saved_actuators and isinstance(saved_event, dict):
        from isaaclab.managers import EventTermCfg
        from legged_lab.assets.bsrl_reduced_ankle import randomize_ankle_transmission
        params = {key: tuple(float(v) for v in values)
                  for key, values in saved_event.get("params", {}).items()}
        env_cfg.events.ankle_transmission = EventTermCfg(
            func=randomize_ankle_transmission, mode=saved_event.get("mode", "startup"), params=params)
    # New actuator fields and delay settings are plant parameters too.
    for name, actuator in env_cfg.scene.robot.actuators.items():
        for key in ("motor_effort", "motor_speed", "motor_power", "physics_dt",
                    "control_decimation", "min_delay", "max_delay"):
            value = saved_actuators.get(name, {}).get(key)
            if value is not None and hasattr(actuator, key):
                number = int(value) if key in ("control_decimation", "min_delay", "max_delay") else float(value)
                setattr(actuator, key, number)
    print(f"[INFO] Restored checkpoint BSRL dynamics: {topology}.")
