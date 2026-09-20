"""Reset-safe next-frame targets for the standard PPO/AMP optimizer."""

import math
import torch
from rsl_rl.algorithms import PPOAMP

from .models import LatentActorCritic, NEXT_FRAME_KEY, VALID_KEY


class LatentPPOAMP(PPOAMP):
    def __init__(self, policy, storage, *args, **kwargs):
        if not isinstance(policy, LatentActorCritic):
            raise TypeError("LatentPPOAMP requires LatentActorCritic")
        if kwargs.get("symmetry_cfg") or kwargs.get("state_est_loss_required"):
            raise ValueError("This latent experiment has neither mirror nor supervised estimator")
        if kwargs.get("multi_gpu_cfg"):
            raise ValueError("First latent branch is single-GPU only")
        super().__init__(policy, storage, *args, **kwargs)
        shape = (storage.num_transitions_per_env, storage.num_envs)
        storage.observations[NEXT_FRAME_KEY] = torch.zeros(*shape, policy.actor.layout.frame_dim, device=self.device)
        storage.observations[VALID_KEY] = torch.zeros(*shape, 1, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        # Add labels to the *previous* observation without mutating env data.
        transition_obs = self.transition.observations.clone(recurse=False)
        frame = self.policy.actor.layout.latest(obs["policy"]).detach()
        valid = ~dones.reshape(-1).bool()
        transition_obs[NEXT_FRAME_KEY] = torch.where(valid[:, None], frame, torch.zeros_like(frame))
        transition_obs[VALID_KEY] = valid[:, None].float()
        self.transition.observations = transition_obs
        # Both timeout and failure resets are excluded. Terminal AMP handling
        # remains exactly the existing PPOAMP implementation.
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        metrics = super().update()
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Nonfinite latent/PPO/AMP update: {metrics}")
        # clear() resets only the storage cursor; data still contains the rollout.
        self.policy.commit_normalization(self.storage.observations["policy"].flatten(0, 1))
        return metrics
