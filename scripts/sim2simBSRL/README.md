# BSRL sim2sim

This runner uses Bipedal_AMP's canonical BSRL assets under
`source/legged_lab/legged_lab/data/Robots/BSRL_urdf`. Robot XML, URDF, and
meshes are not duplicated under `scripts/sim2simBSRL`.

The current BSRL XML contains the rigid-body and collision model but no
`<actuator>` section. `run_sim2sim_standalone.py` injects twelve MuJoCo motors
at load time and configures them as implicit position-PD servos, matching the
IsaacLab play configuration:

- physics timestep: `1 ms`
- policy/control decimation: `20` (`50 Hz` policy)
- action scale: `0.25`
- nominal gains: hip/yaw/roll/pitch `Kp=100, Kd=2`; knee `150/4`; ankle `40/2`
- armature: `0.01`
- effort limits: read from the selected XML (`112 Nm` small motors, `216 Nm` large motors in the current `BSRL_urdf` revision)
- velocity limits: read from the canonical `BSRL_urdf/urdf/export_floating.urdf` when available
- play command: `(vx, vy, wz) = (1.0, 0.0, 0.0)`
- play observation corruption and randomized actuator delay: disabled
- contact override: friction `1.0/0.02/0.001`, `condim=6`,
  `solref=0.005/1.0`, `solimp=0.99/0.9999/0.0002/0.5/2.0`
- joint-limit override: `solref=0.02/1.0`,
  `solimp=0.95/0.99/0.001/0.5/2.0`, zero margin

Run the exported policy:

```bash
python scripts/sim2simBSRL/run_sim2sim_standalone.py \
  --policy logs/rsl_rl/bsrl_amp/bsrl_amp_no_actor_feet/2026-08-24_00-16-13_no_actor_feet_formal1_8192/exported/policy.onnx
```

The policy remains at 50 Hz and
physics at 1 kHz; display refresh is limited to 25 Hz so the 1440p viewer can
maintain real time. Press `Ctrl+1/2/3/4/5` for
`1x/2x/3x/4x/8x` wall-clock speed and `Space` to pause or resume. Use
`--no-realtime` for unlimited speed and `--no-follow-camera` for a fixed camera.
Keyboard movement is disabled by default, so the runner uses the fixed
`--command` value (`1.0 0.0 0.0` by default). Add `--keyboard-control` to start
from zero velocity and enable held-key movement. Hold the up/down arrows for
forward/backward motion, left/right arrows for lateral motion, and the bare
`,`/`.` keys for yaw velocity. Each held axis ramps at `0.2` per second and is
clamped to `1.2 m/s` forward/backward, `0.8 m/s` laterally, and `0.5 rad/s` in
yaw. Releasing the keys immediately zeros that axis. Press `0` to stop all
commands.

For a finite headless replay:

```bash
python scripts/sim2simBSRL/run_sim2sim_standalone.py \
  --headless --no-realtime --max_control_steps 1000 --record
```

Run the fixed-target PD hold without loading ONNX:

```bash
python scripts/sim2simBSRL/run_sim2sim_standalone.py --fixed-target-pd
```

This diagnostic uses dedicated static-hold gains (`hip=150/5`, `knee=300/8`,
`ankle=200/5`). The lower policy gains, especially ankle `40/2`, are intended
for closed-loop policy control and do not hold this free-base model statically.

The default XML is
`source/legged_lab/legged_lab/data/Robots/BSRL_urdf/xml/bsrl_scene.xml`; velocity
limits are read from the sibling `urdf/` directory. The loader resolves the
XML's mesh directory and injects any missing policy motors, so both
actuator-free and partially actuated model revisions can be used directly.

The runner supports both BSRL actor layouts. A 270-D policy includes the
`key_body_pos_b` actor term; a 240-D `no_actor_feet` policy omits that term.
Policies with an additional two-value heading term per frame are also
recognized (280-D with key bodies, 250-D without). This compatibility does
not add heading observations to existing 240-D/270-D policies.
Both inputs follow IsaacLab `ObservationManager` history packing: each active
observation term contains its five oldest-to-newest frames, then the term
histories are concatenated. It is not five complete frames. With this
layout, both the original 270-D actor and the 240-D no-actor-feet actor can be
loaded directly.

Record MP4 and state data with an explicit, fixed command:

```bash
MUJOCO_GL=egl python scripts/sim2simBSRL/run_sim2sim_standalone.py \
  --policy /path/to/exported/policy.onnx \
  --headless --no-realtime --command 0.6 0 0 --max_control_steps 1500 \
  --video --video_path videos/bsrl_sim2sim.mp4 \
  --video_fps 50 --record
```

`1500` control steps are 30 simulated seconds. Video frames are scheduled
using simulated time, so `--video_fps 25` changes the frame rate, not playback
speed. Resolution is configurable with `--video_width`/`--video_height`.
Do not enable `--keyboard-control` when evaluating a fixed command. The
runner does not automatically reset after a fall. `imageio` and
`imageio-ffmpeg` are needed only for MP4 recording.

The nominal PD, action scale and armature are runner constants, not loaded
from a training run's YAML. Select a matching policy/model; the defaults are
historical compatibility settings, not an automatic selection of the latest run.

Use `--model` to select another compatible BSRL MuJoCo XML explicitly.
