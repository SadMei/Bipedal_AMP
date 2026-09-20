# BSRL predictive latent experiment

Same latent-only, DreamWaQ-inspired method as [T1](README.md), not an exact
reproduction of DreamWaQ or the supplier's undisclosed training objective.

## Architecture

| Part | BSRL | T1 |
| --- | --- | --- |
| Physical/action DoFs | 12 | 25 |
| Per-frame Actor observations | 48 | 87 |
| Ten-frame input | 480 | 870 |
| Encoder | 512/256/128 -> mu20 | identical hidden widths |
| Recent two frames + mu20 | 116 | 194 |
| Actor hidden widths | 2048/1024/512/128 | identical |
| Auxiliary next-frame prediction | 33 | 59 |
| Critic width (unchanged per robot) | 370 | 595 |

BSRL frame order: angular velocity3, local rotation6, command3, absolute joint
position12, joint velocity12, last action12. Input remains Isaac's term-major,
oldest-first history; model reorders it internally. Target33 is angular velocity3,
local rotation6, q12 and dq12, excluding next commands and last action. No heading
integral/privileged state enters Actor. All reset/timeout transitions are masked.

Unchanged T1 method: normalized next-observation MSE + 0.01 * summed latent KL,
auxiliary coefficient1, shared PPO optimizer, PPO gradients into encoder, deterministic
mu for rollout/inference, sampled z for training-only decoder20/128/256/33. Shared
frame normalizer updates only between rollouts; deterministic Actor activation
checkpointing reduces peak memory without changing batches or numerical objective.

## Frozen baseline and launch

Baseline: `2026-09-16_21-06-48_accad34_reduced_parallel_headingpriv_noestimator_nomirror_8192`.
Fresh run, **no mirror or data augmentation**, no supervised estimator. Uses serial
12-DoF PhysX skeleton plus **analytical parallel ankle transmission, not explicit
closed loops**. Same ACCAD34 files/weights (manifest hashes checked), same randomization,
rewards (including heading-frame velocity), command distribution, Critic privileges,
PD/action settings and simulator timestep. Does not borrow T1's task/reward settings.

BSRL retains its own PPO settings (including 4 minibatches, 5 epochs, scalar action
std, entropy0.01) and AMP replay/discriminator parameters. Formal launch: GPU7,
8192 environments, seed42, 25000 iterations cap. Saves every100 iterations.
Existing T1/BSRL runs are untouched.

```bash
bash scripts/rsl_rl/launch_bsrl_latent.sh
BSRL_LATENT_ENVS=512 BSRL_LATENT_ITERS=3 BSRL_LATENT_RUN=bsrl_latent_smoke bash scripts/rsl_rl/launch_bsrl_latent.sh
```

Pre- and post-initialization full configuration guards allow only the documented
architecture/history/normalization changes, output metadata and run-length/env-count
overrides. `params/latent_runtime_audit.yaml` verifies actual480/370/4x39 observation
widths and ankle implementation; `latent_source/` snapshots exact source for replay.

Export uses the same `latent/export.py` standalone script and emits raw480 -> actions12
ONNX with embedded frozen normalization. Older240-input BSRL replay/sim2sim scripts
need ten-frame input support before this policy can be deployed. Training improvement
is **not yet established**; compare matched fixed-command trajectories after training.
