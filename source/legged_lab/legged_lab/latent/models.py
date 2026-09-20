"""DreamWaQ-inspired latent-only policy (not an exact DreamWaQ reproduction)."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


T1_TERM_DIMS = (3, 6, 3, 25, 25, 25)
BSRL_TERM_DIMS = (3, 6, 3, 12, 12, 12)
NEXT_FRAME_KEY = "latent_next_frame"
VALID_KEY = "latent_transition_valid"


def mlp(input_dim, hidden_dims, output_dim=None):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    if output_dim is not None:
        layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class HistoryLayout(nn.Module):
    """Isaac term-major flattened history -> chronological frame-major history."""

    def __init__(self, history_length=10, term_dims=T1_TERM_DIMS):
        super().__init__()
        self.history_length = history_length
        self.frame_dim = sum(term_dims)
        offsets, offset = [], 0
        for width in term_dims:
            offsets.append(offset)
            offset += history_length * width
        order = [start + frame * width + col
                 for frame in range(history_length)
                 for start, width in zip(offsets, term_dims)
                 for col in range(width)]
        self.register_buffer("frame_order", torch.tensor(order, dtype=torch.long))
        self.register_buffer("latest_order", self.frame_order[-self.frame_dim:].clone())

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.index_select(-1, self.frame_order).reshape(-1, self.history_length, self.frame_dim)

    def latest(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.index_select(-1, self.latest_order)


class LatentActor(nn.Module):
    """Deployable deterministic path; decoder/log-variance never enter inference."""

    def __init__(self, num_actions, history_length=10, recent_frames=2, latent_dim=20,
                 encoder_dims=(512, 256, 128), actor_dims=(2048, 1024, 512, 128),
                 term_dims=T1_TERM_DIMS):
        super().__init__()
        self.layout = HistoryLayout(history_length, term_dims)
        self.input_dim = history_length * self.layout.frame_dim
        self.recent_frames = recent_frames
        self.normalizer = EmpiricalNormalization(self.layout.frame_dim)
        self.encoder = mlp(self.input_dim, encoder_dims)
        self.latent_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.policy = mlp(latent_dim + recent_frames * self.layout.frame_dim, actor_dims, num_actions)

    def encode(self, raw: torch.Tensor):
        frames = self.normalizer(self.layout(raw))
        hidden = self.encoder(frames.flatten(1))
        return frames, hidden, self.latent_mu(hidden)

    def _forward_impl(self, raw: torch.Tensor) -> torch.Tensor:
        frames, _, mu = self.encode(raw)
        recent = frames[:, -self.recent_frames:, :].flatten(1)
        return self.policy(torch.cat((mu, recent), dim=-1))

    @torch.jit.unused
    def _checkpointed_forward(self, raw):
        return checkpoint(self._forward_impl, raw, use_reentrant=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_impl(raw)
        if self.training and torch.is_grad_enabled():
            # Pure deterministic recomputation; changes memory, not PPO batches.
            return self._checkpointed_forward(raw)
        return self._forward_impl(raw)


class LatentActorCritic(ActorCritic):
    """PPO and predictive VAE jointly train the encoder, with no privileged input.

    The Actor always uses mu, including during stochastic PPO action sampling.
    Only the auxiliary decoder samples z. Normalizer updates are explicit at
    rollout boundaries so PPO recomputes likelihoods under identical statistics.
    """

    term_dims = T1_TERM_DIMS
    action_count = 25
    robot_name = "T1"

    def __init__(self, obs, obs_groups, num_actions, history_length=10, recent_frames=2,
                 latent_dim=20, encoder_dims=(512, 256, 128), decoder_dims=(128, 256),
                 latent_loss_coef=1.0, latent_kl_beta=0.01,
                 actor_hidden_dims=(2048, 1024, 512, 128), **kwargs):
        frame_dim = sum(self.term_dims)
        if obs_groups["policy"] != ["policy"] or obs["policy"].shape[-1] != history_length * frame_dim:
            raise ValueError(f"Latent {self.robot_name} requires only the {frame_dim}-feature/frame policy history (no labels/heading).")
        if num_actions != self.action_count or not 1 <= recent_frames <= history_length:
            raise ValueError(f"Invalid {self.robot_name} action/history dimensions")
        if any(key in sum(obs_groups.values(), []) for key in (NEXT_FRAME_KEY, VALID_KEY)):
            raise ValueError("Auxiliary targets must not enter Actor, Critic or discriminator groups")
        if kwargs.get("actor_obs_normalization") or kwargs.get("state_dependent_std"):
            raise ValueError("Use internal frame normalization and state-independent action noise")
        if kwargs.get("activation", "elu") != "elu":
            raise ValueError("The first latent branch uses ELU")
        super().__init__(obs, obs_groups, num_actions, actor_hidden_dims=actor_hidden_dims, **kwargs)
        self.actor = LatentActor(num_actions, history_length, recent_frames, latent_dim,
                                 encoder_dims, actor_hidden_dims, term_dims=self.term_dims)
        self.latent_logvar = nn.Linear(encoder_dims[-1], latent_dim)
        # Predict physical proprioception only: omega3, rotation6, qN, dqN.
        # Do not waste bottleneck capacity copying next commands/previous actions.
        self.register_buffer("prediction_indices", torch.tensor(list(range(9)) + list(range(12, 12 + 2 * num_actions))))
        self.decoder = mlp(latent_dim, decoder_dims, 9 + 2 * num_actions)
        self.latent_loss_coef = float(latent_loss_coef)
        self.latent_kl_beta = float(latent_kl_beta)
        # Runner passes GPU observations before its final policy.to(device).
        self.to(obs["policy"].device)
        self.commit_normalization(obs["policy"])
        print(f"Latent Actor: {self.actor}\nAuxiliary decoder (training only): {self.decoder}")

    @torch.no_grad()
    def update_normalization(self, obs):
        # PPO calls this per environment step. Keep rollout statistics fixed.
        pass

    @torch.no_grad()
    def commit_normalization(self, raw):
        self.actor.normalizer.update(self.actor.layout.latest(raw))

    def compute_auxiliary_loss(self, obs):
        valid = obs[VALID_KEY].squeeze(-1).bool()
        # Index before the forward/loss: a reset target (even NaN) is never used.
        if not torch.any(valid):
            zero = self.actor.latent_mu.weight.sum() * 0.0
            return zero, {"latent/reconstruction": zero.detach(), "latent/kl": zero.detach(),
                          "latent/valid_fraction": zero.detach(), "latent/mu_std": zero.detach()}
        _, hidden, mu = self.actor.encode(self.get_actor_obs(obs)[valid])
        logvar = self.latent_logvar(hidden).clamp(-10.0, 5.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        prediction = self.decoder(z)
        with torch.no_grad():
            target = self.actor.normalizer(obs[NEXT_FRAME_KEY][valid]).index_select(-1, self.prediction_indices)
        reconstruction = (prediction - target).square().mean()
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(-1).mean()
        loss = self.latent_loss_coef * (reconstruction + self.latent_kl_beta * kl)
        return loss, {"latent/reconstruction": reconstruction.detach(), "latent/kl": kl.detach(),
                      "latent/valid_fraction": valid.float().mean(),
                      "latent/mu_std": mu.detach().std(dim=0, unbiased=False).mean()}


class BSRLLatentActorCritic(LatentActorCritic):
    """The identical predictive objective, for BSRL's 12-joint observation contract."""

    term_dims = BSRL_TERM_DIMS
    action_count = 12
    robot_name = "BSRL"
