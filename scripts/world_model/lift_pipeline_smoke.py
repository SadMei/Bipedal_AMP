"""End-to-end CPU smoke test for the LIFT-like pipeline.

This deliberately uses a tiny analytic vector environment.  It verifies data
causality and the complete algorithmic loop without launching Isaac Sim.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("LEGGED_LAB_SKIP_TASK_REGISTRATION", "1")

import torch

from legged_lab.world_model import (
    EnvStepResult,
    IsaacLabDataCollectionEnv,
    PhysicsInformedWorldModel,
    PipelineState,
    ReplayBuffer,
    ResidualWorldModel,
    ResidualWorldModelCfg,
    TransitionCollector,
    TransitionBatch,
    WorldModelStateSpec,
    WorldModelTrainer,
    WorldModelTrainerCfg,
    pack_frame,
)
from legged_lab.world_model.pipeline import LIFTLikePipeline, LIFTLikePipelineCfg
from legged_lab.world_model.sac import SACAgent, SACCfg


class FakeIsaacAutoResetEnv:
    """Tiny stand-in that matches Isaac's reset-before-return step order."""

    def __init__(self) -> None:
        self.unwrapped = self
        self.num_envs = 2
        self.device = "cpu"
        self.value = torch.zeros(2, 1)
        self.steps = torch.zeros(2, dtype=torch.long)

    def _reset_idx(self, env_ids) -> None:
        self.value[env_ids] = 0.0
        self.steps[env_ids] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        return {"policy": self.value.clone()}, {}

    def step(self, action):
        self.value += action
        self.steps += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.steps >= 2
        ids = truncated.nonzero().flatten()
        if ids.numel():
            self._reset_idx(ids)
        return {"policy": self.value.clone()}, self.value[:, 0].clone(), terminated, truncated, {}


class ToyDynamicsEnv:
    def __init__(
        self,
        num_envs: int,
        spec: WorldModelStateSpec,
        gain: float,
        friction: float,
        max_episode_steps: int = 5,
    ) -> None:
        self.num_envs = num_envs
        self.action_dim = spec.num_joints
        self.spec = spec
        self.gain = gain
        self.friction = friction
        self.max_episode_steps = max_episode_steps
        self.dt = 0.05
        self.q = torch.zeros(num_envs, self.action_dim)
        self.dq = torch.zeros_like(self.q)
        self.previous_action = torch.zeros_like(self.q)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long)

    def reset(self) -> PipelineState:
        self.q.uniform_(-0.2, 0.2)
        self.dq.zero_()
        self.previous_action.zero_()
        self.episode_step.zero_()
        return self._state()

    def step(self, action: torch.Tensor) -> EnvStepResult:
        action = action.to(self.q).clamp(-1.0, 1.0)
        ddq = self.gain * action - self.friction * self.dq - 0.4 * self.q + 0.08 * torch.sin(2.0 * self.q)
        self.dq = self.dq + self.dt * ddq
        self.q = self.q + self.dt * self.dq
        self.previous_action = action
        self.episode_step += 1
        transition_next = self._state()
        reward = -(self.q.square().sum(-1) + 0.01 * action.square().sum(-1))
        terminated = (self.q.abs() > 2.5).any(dim=-1)
        truncated = self.episode_step >= self.max_episode_steps
        done = torch.logical_or(terminated, truncated)

        if done.any():
            self.q[done].uniform_(-0.2, 0.2)
            self.dq[done].zero_()
            self.previous_action[done].zero_()
            self.episode_step[done] = 0
        continuation = self._state()
        return EnvStepResult(transition_next, continuation, reward, terminated, truncated)

    def _state(self) -> PipelineState:
        frame = make_frame(self.spec, self.q, self.dq, self.previous_action)
        return PipelineState(policy_obs=torch.cat((self.q, self.dq), dim=-1), wm_frame=frame)


class ToyNominalPhysics:
    def __init__(self, spec: WorldModelStateSpec, gain: float = 1.0, friction: float = 0.1) -> None:
        self.spec = spec
        self.gain = gain
        self.friction = friction
        self.dt = 0.05

    def __call__(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        generalized_residual_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame = history[:, -1, :].clone()
        q_slice = slice(6, 6 + self.spec.num_joints)
        dq_slice = slice(6 + self.spec.num_joints, 6 + 2 * self.spec.num_joints)
        q = frame[:, q_slice]
        dq = frame[:, dq_slice]
        joint_residual = (
            0.0
            if generalized_residual_force is None
            else generalized_residual_force[:, -self.spec.num_joints :]
        )
        ddq = self.gain * action - self.friction * dq - 0.4 * q + joint_residual
        next_dq = dq + self.dt * ddq
        next_q = q + self.dt * next_dq
        frame[:, q_slice] = next_q
        frame[:, dq_slice] = next_dq
        frame[:, -self.spec.num_joints :] = action
        return frame


def make_frame(
    spec: WorldModelStateSpec,
    q: torch.Tensor,
    dq: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    batch = q.shape[0]
    zero3 = torch.zeros(batch, 3, device=q.device)
    root_quat = torch.zeros(batch, 4, device=q.device)
    root_quat[:, 0] = 1.0
    return pack_frame(
        spec,
        {
            "projected_gravity": torch.tensor([0.0, 0.0, -1.0], device=q.device).expand(batch, -1),
            "base_ang_vel": zero3,
            "joint_pos": q,
            "joint_vel": dq,
            "root_quat": root_quat,
            "base_lin_vel": zero3,
            "root_height": torch.ones(batch, 1, device=q.device),
            "foot_contact": torch.ones(batch, spec.num_feet, device=q.device),
            "foot_height": torch.zeros(batch, spec.num_feet, device=q.device),
            "reference_joint_pos": torch.zeros_like(q),
            "reference_joint_vel": torch.zeros_like(q),
            "previous_action": previous_action,
        },
    )


def main() -> None:
    torch.manual_seed(7)
    fake_isaac = FakeIsaacAutoResetEnv()
    isaac_adapter = IsaacLabDataCollectionEnv(
        fake_isaac,
        policy_observation_fn=lambda observation: observation["policy"],
        world_model_frame_fn=lambda env: env.value,
        action_dim=1,
        state_estimator_label_fn=lambda env: env.value,
    )
    isaac_adapter.reset()
    isaac_adapter.step(torch.ones(2, 1))
    terminal_step = isaac_adapter.step(torch.ones(2, 1))
    assert torch.equal(terminal_step.transition_next.wm_frame, torch.full((2, 1), 2.0))
    assert torch.equal(terminal_step.transition_next.state_est_label, torch.full((2, 1), 2.0))
    assert torch.equal(terminal_step.continuation.wm_frame, torch.zeros(2, 1))
    isaac_adapter.close()

    spec = WorldModelStateSpec(num_joints=2, num_feet=1, history_length=2)
    obs_dim = 4
    device = "cpu"
    residual = ResidualWorldModel(
        ResidualWorldModelCfg(
            state=spec,
            prediction_type="state_delta",
            hidden_dims=(64, 64),
            ensemble_size=3,
            num_elites=2,
        )
    )
    nominal = ToyNominalPhysics(spec)
    q_slice = slice(6, 6 + spec.num_joints)
    dq_slice = slice(6 + spec.num_joints, 6 + 2 * spec.num_joints)
    observation_fn = lambda history: torch.cat(
        (history[:, -1, q_slice], history[:, -1, dq_slice]), dim=-1
    )
    reward_fn = lambda history, action, next_history: -(
        next_history[:, -1, q_slice].square().sum(-1, keepdim=True)
        + 0.01 * action.square().sum(-1, keepdim=True)
    )
    termination_fn = lambda history: (
        (history[:, -1, q_slice].abs() > 2.5).any(-1, keepdim=True),
        torch.zeros(history.shape[0], 1, dtype=torch.bool, device=history.device),
    )
    world_model = PhysicsInformedWorldModel(
        residual,
        nominal,
        observation_fn,
        reward_fn,
        termination_fn,
        uncertainty_threshold=2.0,
    )
    agent = SACAgent(
        SACCfg(
            obs_dim=obs_dim,
            action_dim=spec.action_dim,
            actor_hidden_dims=(64, 64),
            critic_hidden_dims=(64, 64),
            estimator_output_dim=0,
        ),
        device,
    )
    trainer = WorldModelTrainer(
        world_model,
        WorldModelTrainerCfg(batch_size=32, normalizer_samples=256, rollout_horizon=1),
        device,
    )

    def buffer(capacity: int) -> ReplayBuffer:
        return ReplayBuffer(capacity, obs_dim, spec.action_dim, spec.history_length, spec.frame_dim)

    simulation_buffer = buffer(4096)
    target_buffer = buffer(4096)
    synthetic_buffer = buffer(8192)
    pipeline = LIFTLikePipeline(
        agent,
        world_model,
        trainer,
        simulation_buffer,
        target_buffer,
        synthetic_buffer,
        LIFTLikePipelineCfg(
            sac_batch_size=32,
            world_model_updates_per_cycle=8,
            target_collect_steps_per_cycle=8,
            synthetic_rollout_starts=16,
            synthetic_horizon=20,
            sac_updates_per_cycle=8,
            real_ratio=0.25,
        ),
    )

    simulation = TransitionCollector(ToyDynamicsEnv(8, spec, gain=0.85, friction=0.22), spec)
    pipeline.pretrain_sac(simulation, control_steps=48, random_control_steps=8)
    with tempfile.TemporaryDirectory() as directory:
        replay_path = Path(directory) / "simulation_replay.pt"
        pipeline.save_simulation_replay(replay_path)
        restored = ReplayBuffer.load(replay_path)
        assert len(restored) == len(simulation_buffer)
        assert restored.state_dict()["data"]["policy_obs"].shape[0] == len(restored)
        checkpoint_path = Path(directory) / "pipeline.pt"
        pipeline.save_checkpoint(checkpoint_path)
        pipeline.load_checkpoint(checkpoint_path)
    wm_metrics = pipeline.pretrain_world_model(updates=8)

    # Also verify the true LIFT path: residual generalized forces are passed
    # through a differentiable nominal step before next-state loss.
    force_residual = ResidualWorldModel(
        ResidualWorldModelCfg(
            state=spec,
            prediction_type="generalized_force",
            hidden_dims=(32, 32),
            ensemble_size=2,
            num_elites=1,
            probabilistic=True,
        )
    )
    force_world_model = PhysicsInformedWorldModel(
        force_residual,
        nominal,
        observation_fn,
        reward_fn,
        termination_fn,
    )
    force_trainer = WorldModelTrainer(
        force_world_model,
        WorldModelTrainerCfg(batch_size=16, normalizer_samples=64, rollout_horizon=1),
        device,
    )
    force_metrics = force_trainer.train(simulation_buffer, updates=2)
    force_seed = simulation_buffer.sample(4)
    force_prediction = force_residual(force_seed.wm_history, force_seed.action)
    assert force_prediction.mean.shape == (
        force_residual.cfg.ensemble_size,
        4,
        spec.generalized_force_dim,
    )

    # wbtest SAC keeps the deployed estimator contract: raw312 -> estimate8
    # -> actor320, while the critic input remains independently configurable.
    estimator_agent = SACAgent(
        SACCfg(
            obs_dim=312,
            critic_obs_dim=312,
            action_dim=spec.action_dim,
            actor_hidden_dims=(32, 16),
            critic_hidden_dims=(32, 16),
            estimator_output_dim=8,
            estimator_hidden_dims=(32, 16),
        ),
        device,
    )
    count = 8
    raw = torch.randn(count, 312)
    labels = torch.randn(count, 8)
    labels[:, 4:6] = torch.randint(0, 2, (count, 2)).float()
    estimator_batch = TransitionBatch(
        policy_obs=raw,
        critic_obs=raw.clone(),
        action=torch.randn(count, spec.action_dim).tanh(),
        reward=torch.randn(count, 1),
        next_policy_obs=torch.randn(count, 312),
        next_critic_obs=torch.randn(count, 312),
        terminated=torch.zeros(count, 1, dtype=torch.bool),
        truncated=torch.zeros(count, 1, dtype=torch.bool),
        wm_history=torch.zeros(count, spec.history_length, spec.frame_dim),
        next_wm_history=torch.zeros(count, spec.history_length, spec.frame_dim),
        state_est_label=labels,
        next_state_est_label=torch.zeros(count, 8),
        state_est_label_valid=torch.ones(count, 1, dtype=torch.bool),
        next_state_est_label_valid=torch.zeros(count, 1, dtype=torch.bool),
    )
    estimator_metrics = estimator_agent.update(estimator_batch)
    assert estimator_agent.augment_observation(raw).shape == (count, 320)
    assert estimator_agent.actor.network[0].in_features == 320
    assert estimator_metrics["state_est/valid_samples"] == count
    with tempfile.TemporaryDirectory() as directory:
        ppo_path = Path(directory) / "ppo_estimator.pt"
        actor_state = {
            **{
                f"estimator_input_normalizer.{key}": value
                for key, value in estimator_agent.estimator_input_normalizer.state_dict().items()
            },
            **{
                f"estimator.{key}": value
                for key, value in estimator_agent.estimator.state_dict().items()
            },
        }
        torch.save({"actor_state_dict": actor_state}, ppo_path)
        estimator_agent.load_estimator_from_ppo(str(ppo_path))

    target = TransitionCollector(ToyDynamicsEnv(8, spec, gain=0.72, friction=0.30), spec)
    metrics = pipeline.finetune_cycle(target)
    assert len(target_buffer) == 64
    assert 0 < len(synthetic_buffer) <= 16 * 20
    assert metrics["finetune/synthetic_transitions"] == len(synthetic_buffer)
    print(
        "LIFT-like smoke passed:",
        f"sim={len(simulation_buffer)}",
        f"target={len(target_buffer)}",
        f"synthetic={len(synthetic_buffer)}",
        f"wm_mse={wm_metrics['world_model/validation_mse']:.6f}",
        f"force_wm_mse={force_metrics['world_model/validation_mse']:.6f}",
    )


if __name__ == "__main__":
    main()
