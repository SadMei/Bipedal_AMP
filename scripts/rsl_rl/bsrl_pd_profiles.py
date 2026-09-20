"""Simulator-only BSRL joint-space PD candidates; never edits training assets."""

AXES = ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll")
PROFILES = {
    "nominal": ([150, 150, 150, 200, 100, 100], [2, 2, 2, 2, 1, 1]),
    "damped2": ([150, 150, 150, 200, 100, 100], [4, 4, 4, 4, 2, 2]),
    "balanced15": ([225, 225, 225, 300, 150, 150], [4, 4, 4, 4, 2, 2]),
    "stiff2damped3": ([300, 300, 300, 400, 200, 200], [6, 6, 6, 6, 3, 3]),
    "damped15": ([150, 150, 150, 200, 100, 100], [3, 3, 3, 3, 1.5, 1.5]),
    "hipdamped": ([150, 150, 150, 200, 100, 100], [3, 6, 6, 4, 2, 2]),
    "distaldamped": ([150, 150, 150, 200, 100, 100], [2, 2, 2, 4, 2, 2]),
    # Hardware constraint reported by the user: never increase ankle damping.
    # The following candidates keep BOTH ankle Kp/Kd at 100/1.
    "proximal15": ([150, 150, 150, 200, 100, 100], [3, 3, 3, 3, 1, 1]),
    "proximal2": ([150, 150, 150, 200, 100, 100], [4, 4, 4, 4, 1, 1]),
    "hip6_anklefixed": ([150, 150, 150, 200, 100, 100], [3, 6, 6, 4, 1, 1]),
    "knee4_anklefixed": ([150, 150, 150, 200, 100, 100], [2, 2, 2, 4, 1, 1]),
}

# Retain earlier sweeps for reproducibility, but exclude ankle-D increases from
# the default search after the user reported a real-hardware stability limit.
DEFAULT_PROFILES = ("nominal", "proximal15", "proximal2", "hip6_anklefixed", "knee4_anklefixed")


def gain_vectors(profile, joint_names):
    gains = PROFILES[profile]
    axes = [name.removeprefix("joint_left_").removeprefix("joint_right_") for name in joint_names]
    return tuple([values[AXES.index(axis)] for axis in axes] for values in gains)


def apply_bsrl_pd_profile(cfg, robot_name, profile):
    # Nominal means the SAVED run, not today's asset defaults.
    if profile == "nominal":
        return
    if robot_name != "BSRL":
        raise ValueError("PD candidate overrides are restricted to BSRL")
    kp, kd = PROFILES[profile]
    groups = {
        "large_motors": ("hip_pitch", "hip_roll", "knee_pitch"),
        "small_motors": ("hip_yaw",),
        "parallel_ankles": ("ankle_pitch", "ankle_roll"),
    }
    for name, axes in groups.items():
        actuator = cfg.scene.robot.actuators[name]
        actuator.stiffness = {f"joint_.*_{axis}": kp[AXES.index(axis)] for axis in axes}
        actuator.damping = {f"joint_.*_{axis}": kd[AXES.index(axis)] for axis in axes}
