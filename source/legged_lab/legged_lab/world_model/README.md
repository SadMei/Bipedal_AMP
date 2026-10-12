# Legged Lab residual world model

This package is the AMP-side port of the user-developed WM in
`/root/mjw/wbt_est/source/whole_body_tracking/whole_body_tracking/world_model`.
The imported snapshot and every original file hash are recorded in
`PROVENANCE.json`. Project adaptations are deliberately kept in named adapter
modules rather than folded into the numerical core.

## Stable contracts

`WorldModelStateSpec` is the only authority for robot-dependent dimensions.
A frame is:

```text
dynamic:
  projected gravity(3), body angular velocity(3), q(N), dq(N),
  root quaternion wxyz(4), body linear velocity(3), root height(1),
  foot contact(F), foot height(F)
context:
  reference q(N), reference dq(N), previous action(N)
```

The active residual network input is the current privileged physical state
with yaw removed from the quaternion, followed by the current action. It does
not consume reference context or history directly. History remains required
for causal replay and autoregressive rollout.

| contract | joints | dynamic | context | frame | model input | force output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 | 25 | 68 | 75 | 143 | 86 | 31 |
| BSRL | 12 | 42 | 36 | 78 | 47 | 18 |

`pack_frame` rejects missing fields, wrong order/width and inconsistent batch
shapes. `FrameHistory.push(reset=...)` prevents history leakage across episode
boundaries. `IsaacLabDataCollectionEnv` stores the real terminal state
separately from IsaacLab's post-reset continuation state.

## AMP adapter

Use `amp_world_model_frame` to read physical state from the robot/contact
sensor and reference q/dq from AMP's `AnimationTerm`. It never registers an
observation or changes PPO/AMP configuration. Run
`validate_amp_world_model_contract` before writing replay; it audits actual
joint order, action dimension, frame dimension and finiteness.

For BSRL, use `bsrl_world_model_state_spec()` and pass
`BSRL_12_JOINT_NAMES` as the expected order. Reference state is never silently
filled with zeros.

The optional SAC path uses the full 144-D three-frame BSRL observation for its
8-D estimator and `bsrl_estimator_actor_indices()` to select only the latest
48-D frame for the actor, yielding the same 56-D augmented actor contract as
AMP PPO. `load_estimator_from_ppo` accepts both standalone wbtest keys and
AMP/RSL-RL's nested `model_state_dict -> actor.*` keys.

## MJX and BSRL scope

`bsrl_serial_mjx_cfg` provides a strict 12-DoF serial-joint nominal backend for
framework and dimensional-generalization tests. The BSRL XML contains no
MuJoCo actuators, so this backend applies PD joint torque explicitly through
`qfrc_applied`.

It is **not** the R2 plant. R2 uses `ReducedAnkleActuator` and the analytical
parallel-ankle transmission with motor-space effort/speed/power limits and
transmission randomization. Calling `bsrl_r2_mjx_cfg` raises
`NotImplementedError` until that exact plant is implemented in MJX. This
guard prevents a serial approximation from being mislabeled as sim2real R2.

T1 factories and checkpoint formats remain source-compatible with wbtest.
Checkpoint loading continues to reject mismatched backend, integrator,
horizon, force frame, orientation input and variance parameterization.

## Dependencies and verification

Core state/model/replay tests require PyTorch. Isaac collection requires the
project's IsaacLab environment. MJX construction additionally requires
`mujoco`, `mujoco-mjx` and a JAX build matching the PyTorch device; CUDA input
is rejected if JAX is CPU-only.

Run the focused tests with:

```bash
LEGGED_LAB_SKIP_TASK_REGISTRATION=1 PYTHONPATH=source/legged_lab \
  /root/miniconda3/envs/isaaclab/bin/python -m unittest \
  source/legged_lab/test/test_world_model_port.py -v
```

The ported end-to-end replay/WM/SAC pipeline smoke is:

```bash
LEGGED_LAB_SKIP_TASK_REGISTRATION=1 PYTHONPATH=source/legged_lab \
  /root/miniconda3/envs/isaaclab/bin/python \
  scripts/world_model/lift_pipeline_smoke.py
```

## Migration verification (2026-10-11)

- Existing AMP 4-D/8-D estimator regression: 22 tests passed.
- WM/BSRL contracts: 8 tests passed.
- End-to-end toy collection, replay, residual training and SAC: passed
  (`384` simulation, `64` target and `320` synthetic transitions).
- BSRL serial MJX: real project XML compiled as `nq=19, nv=18, nu=0`;
  one 20 ms step returned a finite 78-D frame.
- Frozen R2 Isaac audit: 8 environments, exact saved-config restoration,
  joint order equal to `BSRL_12_JOINT_NAMES`, action dim 12, finite frame
  shape `[8, 78]`, policy dt 20 ms.
- AMP wbtest estimator checkpoint import: 14 estimator/normalizer tensors
  matched bit-for-bit; deterministic SAC action shape `[3, 12]` was finite.
