# T1 predictive latent branch

Independent of `state_est`; inspired by DreamWaQ (https://arxiv.org/abs/2301.10602),
**not** an exact reproduction of DreamWaQ or the closed-source supplier policy.
DreamWaQ also estimates velocity explicitly; this first experiment has **latent only**.

The same implementation now supports the 12-DoF BSRL contract; see
[BSRL baseline, dimensions and launcher](README_bsrl.md). T1 checkpoint shapes,
configuration defaults and numerical behavior are unchanged.

## Architecture

```text
Isaac term-major history870 -> chronological [10,87] -> frame normalization
  encoder870 -> 512 -> 256 -> 128 -> mu20
  latest two normalized frames174 + mu20 -> 194
  Actor194 -> 2048 -> 1024 -> 512 -> 128 -> 25 actions

Training only:
  encoder hidden128 -> logvar20
  z = mu + exp(0.5*logvar)*epsilon -> decoder20/128/256/59
  target: next-step [angular velocity3, local rotation6, joint position25, joint velocity25]
```

T1 keeps its existing 87 features/frame (including 6D local orientation), hence
870/194, not the supplier's 840/188. History is oldest-first, 50Hz: ten samples
span 0.18s. Isaac repeats the initial observation after reset; history is not
allowed to retain the previous episode. No recurrent hidden state is needed.

The prediction target is the next **observed** proprioception (same noise/scaling
as Actor), not privileged true states. Next commands and previous actions are
excluded as trivial/exogenous targets. This simple decoder takes z alone, no
action conditioning; it models one-step behavior under the current policy,
not arbitrary action-conditioned world dynamics. Thus it is not a drop-in WM.

`L = L_PPO + 1.0 * (mean(MSE_59) + 0.01 * mean(sum(KL_20)))`.
KL is against N(0,I), log-variance clamped to [-10,5]. These weights are initial
experiment choices, not recovered supplier settings. PPO and auxiliary losses
both update the encoder (no detach). Actor always uses **mu**, during rollout,
PPO likelihood recomputation, and deployment; Gaussian exploration is applied
only at the normal action distribution. Decoder sampling does not enter Actor.
All parameters share the existing PPO Adam and gradient clipping; AMP keeps its
existing separate discriminator optimizer. Checkpoints retain decoder/optimizer.
Training uses non-reentrant activation checkpointing for the deterministic Actor
forward to fit 8192 environments on 32GB without changing minibatches or replay
capacity. Export/rollout bypass recomputation; there are no stochastic layers in
this path.

Next-frame labels and valid masks travel only in rollout storage and are absent
from Actor/Critic/discriminator groups. Both failure and timeout transitions are
excluded **before** computing auxiliary losses, so no prediction across reset.
Normalizer statistics are shared across ten frames, updated initially and after
each PPO update from the rollout's current frames, frozen during rollout/update,
and embedded/frozen in export. Critic keeps its original 595 inputs and no
normalization. No explicit state estimation labels, heading integral, or external
heading-to-yaw-rate controller enter the Actor.

## Comparison and launch

Restore saved **2026-09-15_19-40-00** T1 no-heading/no-estimator parameters exactly;
retain reward weights, 18 reference clips, straight-command sampling, command
conditioning, domain randomization, PD/action settings, and Critic privileges.
Per user request, **mirror loss and data augmentation are both disabled**.
This is therefore an architecture + no-mirror experiment, not an architecture-only
ablation against the mirror-enabled 9/15 run. No guarantee of eliminating yaw drift.

From repository root:

```bash
bash scripts/rsl_rl/launch_t1_latent.sh
# Short diagnostic launch (separate checkpoint, not used to initialize formal run):
T1_LATENT_ENVS=512 T1_LATENT_ITERS=3 T1_LATENT_RUN=latent_smoke bash scripts/rsl_rl/launch_t1_latent.sh
```

Default: physical GPU6, 8192 environments, seed42, 25000 iterations maximum.
Saved `params/latent_ablation_audit.yaml` records allowed config differences;
`latent_runtime_audit.yaml` verifies actual term order and Actor/Critic widths.
Monitor `latent/reconstruction`, `latent/kl`, `latent/valid_fraction`, and
`latent/mu_std` as well as normal PPO/AMP metrics. A nonzero latent variance alone
does not prove useful control; later evaluate heading drift, speed, foot heights,
and an inference latent ablation on matched commands/seeds.

Deployment: export `policy.actor` only (input **raw term-major870**, output25).
It includes layout conversion, normalization, encoder, and deterministic mu;
decoder and random latent sampling are training-only. Existing five-frame T1
sim2sim input builders must be adapted to ten frames before deploying this ONNX.

Standalone export (no Isaac launch, verifies ONNX Runtime parity, refuses overwrite):

```bash
/root/miniconda3/envs/isaaclab/bin/python source/legged_lab/legged_lab/latent/export.py /path/to/run/model_500.pt
```

Outputs `exported_latent/model_500/{policy.onnx,policy.pt,contract.json}` in the run.
