from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .collector import TransitionCollector
from .dynamics import PhysicsInformedWorldModel
from .replay import ReplayBuffer, TransitionBatch, sample_mixed
from .sac import SACAgent
from .models import variance_parameterization
from .trainer import WorldModelTrainer


@dataclass(frozen=True)
class LIFTLikePipelineCfg:
    sac_batch_size: int = 256
    world_model_updates_per_cycle: int = 500
    target_collect_steps_per_cycle: int = 1000
    synthetic_rollout_starts: int = 400
    synthetic_horizon: int = 20
    sac_updates_per_cycle: int = 1000
    real_ratio: float = 0.2
    clear_synthetic_after_world_model_update: bool = True

    def validate(self) -> None:
        positive = (
            "sac_batch_size",
            "world_model_updates_per_cycle",
            "target_collect_steps_per_cycle",
            "synthetic_rollout_starts",
            "synthetic_horizon",
            "sac_updates_per_cycle",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.real_ratio <= 1.0:
            raise ValueError("real_ratio must be in [0, 1].")


class LIFTLikePipeline:
    """SAC + physics-informed residual model + Dyna-style fine-tuning loop."""

    def __init__(
        self,
        agent: SACAgent,
        world_model: PhysicsInformedWorldModel,
        world_model_trainer: WorldModelTrainer,
        simulation_buffer: ReplayBuffer,
        target_buffer: ReplayBuffer,
        synthetic_buffer: ReplayBuffer,
        cfg: LIFTLikePipelineCfg | None = None,
    ) -> None:
        self.agent = agent
        self.world_model = world_model
        self.world_model_trainer = world_model_trainer
        self.simulation_buffer = simulation_buffer
        self.target_buffer = target_buffer
        self.synthetic_buffer = synthetic_buffer
        self.cfg = cfg or LIFTLikePipelineCfg()
        self.cfg.validate()
        self._validate_buffers()

    def pretrain_sac(
        self,
        collector: TransitionCollector,
        control_steps: int,
        random_control_steps: int,
        updates_per_control_step: int = 1,
    ) -> dict[str, float]:
        """Collect stochastic simulation data and pretrain SAC off-policy."""

        if control_steps <= 0 or random_control_steps < 0 or updates_per_control_step <= 0:
            raise ValueError("Invalid SAC pretraining schedule.")
        metrics: dict[str, float] = {}
        for step in range(control_steps):
            if step < random_control_steps:
                action_fn = lambda obs: self.agent.random_action(obs.shape[0]).to(obs.device)
            else:
                action_fn = lambda obs: self.agent.act(obs, deterministic=False).to(obs.device)
            collection = collector.collect(self.simulation_buffer, steps=1, action_fn=action_fn)
            metrics.update(
                {
                    "pretrain/transitions": float(len(self.simulation_buffer)),
                    "pretrain/reward_sum": collection.reward_sum,
                }
            )
            if len(self.simulation_buffer) >= self.cfg.sac_batch_size and step >= random_control_steps:
                for _ in range(updates_per_control_step):
                    batch = self.simulation_buffer.sample(self.cfg.sac_batch_size, self.agent.device)
                    metrics.update(self.agent.update(batch))
        return metrics

    def pretrain_world_model(self, updates: int | None = None) -> dict[str, float]:
        """Fit the residual model from the saved simulation replay."""

        return self.world_model_trainer.train(
            self.simulation_buffer,
            self.cfg.world_model_updates_per_cycle if updates is None else updates,
        )

    def save_simulation_replay(self, path: str | Path) -> None:
        self.simulation_buffer.save(path)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 4,
                "residual_force_frame": self.world_model.residual_model.cfg.residual_force_frame,
                "model_orientation_input": self.world_model.residual_model.cfg.model_orientation_input,
                "model_input_mode": self.world_model.residual_model.cfg.model_input_mode,
                "variance_parameterization": (
                    variance_parameterization(self.world_model.residual_model.cfg)
                    if self.world_model.residual_model.cfg.probabilistic
                    else None
                ),
                "agent": self.agent.state_dict(),
                "residual_model": self.world_model.residual_model.state_dict(),
                "world_model_trainer": self.world_model_trainer.state_dict(),
                "pipeline_cfg": self.cfg,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.agent.device, weights_only=False)
        model_cfg = self.world_model.residual_model.cfg
        if (
            model_cfg.prediction_type == "generalized_force"
            and state.get("residual_force_frame") != model_cfg.residual_force_frame
        ):
            raise ValueError(
                "Pipeline checkpoint does not use the body-frame linear-force "
                "residual contract; retrain instead of loading it."
            )
        if state.get("model_orientation_input") != model_cfg.model_orientation_input:
            raise ValueError(
                "Pipeline checkpoint does not use yaw-invariant tilt quaternion "
                "model inputs; retrain instead of loading it."
            )
        if state.get("model_input_mode") != model_cfg.model_input_mode:
            raise ValueError("Pipeline checkpoint model-input mode does not match current model.")
        if (
            model_cfg.probabilistic
            and state.get("variance_parameterization")
            != variance_parameterization(model_cfg)
        ):
            raise ValueError(
                "Pipeline checkpoint uncertainty contract does not match the "
                "current model; retrain the WM."
            )
        self.agent.load_state_dict(state["agent"])
        self.world_model.residual_model.load_state_dict(state["residual_model"])
        self.world_model_trainer.load_state_dict(state["world_model_trainer"])

    def finetune_cycle(self, target_collector: TransitionCollector) -> dict[str, float]:
        """One complete LIFT-like target-data/model-rollout/SAC-update cycle."""

        # Safety-critical invariant: target/real environment sees policy mean only.
        collection = target_collector.collect(
            self.target_buffer,
            steps=self.cfg.target_collect_steps_per_cycle,
            action_fn=lambda obs: self.agent.act(obs, deterministic=True).to(obs.device),
        )
        metrics = {
            "finetune/target_transitions": float(collection.transitions),
            "finetune/target_episodes": float(collection.episodes),
            "finetune/target_reward_sum": collection.reward_sum,
            "finetune/target_buffer_size": float(len(self.target_buffer)),
        }

        metrics.update(
            self.world_model_trainer.train(
                self.target_buffer,
                self.cfg.world_model_updates_per_cycle,
            )
        )
        if self.cfg.clear_synthetic_after_world_model_update:
            self.synthetic_buffer.clear()
        metrics.update(
            self.generate_synthetic_rollouts(
                starts=self.cfg.synthetic_rollout_starts,
                horizon=self.cfg.synthetic_horizon,
            )
        )

        sac_metrics: dict[str, float] = {}
        for _ in range(self.cfg.sac_updates_per_cycle):
            batch = sample_mixed(
                self.target_buffer,
                self.synthetic_buffer,
                self.cfg.sac_batch_size,
                self.cfg.real_ratio,
                self.agent.device,
            )
            sac_metrics = self.agent.update(batch)
        metrics.update(sac_metrics)
        return metrics

    @torch.no_grad()
    def generate_synthetic_rollouts(self, starts: int, horizon: int) -> dict[str, float]:
        """Roll stochastic SAC actions through elite-mean WM dynamics."""

        if len(self.target_buffer) == 0:
            raise ValueError("Target buffer must contain real transitions before model rollout.")
        seed = self.target_buffer.sample(starts, self.agent.device)
        policy_obs = seed.policy_obs
        critic_obs = seed.critic_obs
        history = seed.wm_history
        generated = 0
        uncertainty_sum = 0.0

        for _ in range(horizon):
            if policy_obs.shape[0] == 0:
                break
            action = self.agent.act(policy_obs, deterministic=False)
            result = self.world_model.step(history, action)
            batch = TransitionBatch(
                policy_obs=policy_obs,
                critic_obs=critic_obs,
                action=action,
                reward=result.reward,
                next_policy_obs=result.next_policy_obs,
                next_critic_obs=result.next_critic_obs,
                terminated=result.terminated,
                truncated=result.truncated,
                wm_history=history,
                next_wm_history=result.next_history,
                state_est_label=policy_obs.new_zeros(
                    (policy_obs.shape[0], self.synthetic_buffer.state_est_label_dim)
                ),
                next_state_est_label=policy_obs.new_zeros(
                    (policy_obs.shape[0], self.synthetic_buffer.state_est_label_dim)
                ),
                state_est_label_valid=torch.zeros(
                    (policy_obs.shape[0], 1), device=policy_obs.device, dtype=torch.bool
                ),
                next_state_est_label_valid=torch.zeros(
                    (policy_obs.shape[0], 1), device=policy_obs.device, dtype=torch.bool
                ),
                reference_index=policy_obs.new_zeros((policy_obs.shape[0], 1), dtype=torch.long),
                next_reference_index=policy_obs.new_zeros((policy_obs.shape[0], 1), dtype=torch.long),
            )
            self.synthetic_buffer.add(batch)
            generated += len(batch)
            uncertainty_sum += float(result.uncertainty.sum())

            alive = ~batch.done.reshape(-1)
            policy_obs = result.next_policy_obs[alive]
            critic_obs = result.next_critic_obs[alive]
            history = result.next_history[alive]

        return {
            "finetune/synthetic_transitions": float(generated),
            "finetune/synthetic_buffer_size": float(len(self.synthetic_buffer)),
            "finetune/model_uncertainty_mean": uncertainty_sum / max(generated, 1),
        }

    def _validate_buffers(self) -> None:
        shape = (
            self.simulation_buffer.policy_obs_dim,
            self.simulation_buffer.critic_obs_dim,
            self.simulation_buffer.action_dim,
            self.simulation_buffer.history_length,
            self.simulation_buffer.frame_dim,
            self.simulation_buffer.state_est_label_dim,
        )
        spec = self.world_model.residual_model.cfg.state
        expected = (
            self.agent.cfg.obs_dim,
            self.agent.cfg.resolved_critic_obs_dim,
            self.agent.cfg.action_dim,
            spec.history_length,
            spec.frame_dim,
            self.agent.cfg.estimator_output_dim,
        )
        if shape != expected:
            raise ValueError(f"Simulation buffer shape {shape} does not match agent/world-model shape {expected}.")
        if self.world_model_trainer.world_model is not self.world_model:
            raise ValueError("world_model_trainer must train the same world_model instance used for rollouts.")
        for name, buffer in (("target", self.target_buffer), ("synthetic", self.synthetic_buffer)):
            other = (
                buffer.policy_obs_dim,
                buffer.critic_obs_dim,
                buffer.action_dim,
                buffer.history_length,
                buffer.frame_dim,
                buffer.state_est_label_dim,
            )
            if other != shape:
                raise ValueError(f"{name} buffer shape {other} does not match simulation buffer {shape}.")
