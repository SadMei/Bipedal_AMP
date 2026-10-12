"""Contract tests for the wbtest -> Bipedal_AMP WM port."""

from __future__ import annotations

import os
import unittest

import torch

os.environ["LEGGED_LAB_SKIP_TASK_REGISTRATION"] = "1"

from legged_lab.world_model import (
    BSRL_12_FRAME_JOINT_NAMES,
    BSRL_12_JOINT_NAMES,
    FrameHistory,
    ResidualWorldModel,
    ResidualWorldModelCfg,
    SACAgent,
    SACCfg,
    WorldModelStateSpec,
    bsrl_r2_mjx_cfg,
    bsrl_estimator_actor_indices,
    bsrl_serial_mjx_cfg,
    bsrl_world_model_state_spec,
    model_input,
    pack_frame,
)


def frame_values(spec: WorldModelStateSpec, batch: int = 4):
    value = lambda width: torch.randn(batch, width)
    quaternion = torch.nn.functional.normalize(value(4), dim=-1)
    return {
        "projected_gravity": value(3),
        "base_ang_vel": value(3),
        "joint_pos": value(spec.num_joints),
        "joint_vel": value(spec.num_joints),
        "root_quat": quaternion,
        "base_lin_vel": value(3),
        "root_height": value(1),
        "foot_contact": torch.sigmoid(value(spec.num_feet)),
        "foot_height": value(spec.num_feet),
        "reference_joint_pos": value(spec.num_joints),
        "reference_joint_vel": value(spec.num_joints),
        "previous_action": value(spec.num_joints),
    }


class WorldModelPortTests(unittest.TestCase):
    def test_bsrl_dimensions_are_derived_from_spec(self):
        spec = bsrl_world_model_state_spec()
        self.assertEqual(
            (spec.dynamic_dim, spec.context_dim, spec.frame_dim), (42, 36, 78)
        )
        cfg = ResidualWorldModelCfg(state=spec)
        self.assertEqual(cfg.model_input_dim, 47)
        self.assertEqual(cfg.output_dim, 18)
        self.assertEqual(cfg.uncertainty_dim, 34)

    def test_pack_history_and_model_are_bsrl_sized(self):
        torch.manual_seed(7)
        spec = bsrl_world_model_state_spec()
        frame = pack_frame(spec, frame_values(spec))
        history = FrameHistory.create(spec, frame)
        action = torch.randn(4, spec.action_dim)
        value = model_input(ResidualWorldModelCfg(state=spec), history.tensor, action)
        self.assertEqual(value.shape, (4, 47))
        prediction = ResidualWorldModel(ResidualWorldModelCfg(state=spec))(
            history.tensor, action
        )
        self.assertEqual(prediction.mean.shape, (7, 4, 18))
        self.assertEqual(prediction.logvar.shape, (7, 4, 34))
        self.assertTrue(torch.isfinite(prediction.mean).all())

    def test_quaternion_sign_does_not_change_model_input(self):
        spec = bsrl_world_model_state_spec()
        values = frame_values(spec)
        first = pack_frame(spec, values)
        values["root_quat"] = -values["root_quat"]
        second = pack_frame(spec, values)
        action = torch.randn(first.shape[0], spec.action_dim)
        cfg = ResidualWorldModelCfg(state=spec)
        torch.testing.assert_close(
            model_input(cfg, FrameHistory.create(spec, first).tensor, action),
            model_input(cfg, FrameHistory.create(spec, second).tensor, action),
        )

    def test_history_reset_cannot_leak_previous_episode(self):
        spec = bsrl_world_model_state_spec()
        initial = pack_frame(spec, frame_values(spec))
        history = FrameHistory.create(spec, initial)
        following = pack_frame(spec, frame_values(spec))
        reset = torch.tensor([False, True, False, True])
        result = history.push(following, reset=reset)
        torch.testing.assert_close(
            result[reset],
            following[reset, None, :].expand(-1, spec.history_length, -1),
        )

    def test_pack_frame_fails_closed_on_wrong_width(self):
        spec = bsrl_world_model_state_spec()
        values = frame_values(spec)
        values["previous_action"] = torch.zeros(4, spec.action_dim + 1)
        with self.assertRaisesRegex(ValueError, "previous_action"):
            pack_frame(spec, values)

    def test_bsrl_serial_backend_contract_and_r2_guard(self):
        spec = bsrl_world_model_state_spec()
        cfg = bsrl_serial_mjx_cfg("/tmp/bsrl.xml")
        cfg.validate(spec)
        self.assertEqual(cfg.frame_joint_names, BSRL_12_FRAME_JOINT_NAMES)
        self.assertEqual(cfg.frame_joint_names, BSRL_12_JOINT_NAMES)
        self.assertEqual(cfg.control_mode, "qfrc_applied")
        self.assertEqual(cfg.physics_dt * cfg.control_substeps, 0.02)
        with self.assertRaisesRegex(NotImplementedError, "parallel-ankle"):
            bsrl_r2_mjx_cfg("/tmp/bsrl.xml")

    def test_bsrl_sac_uses_full_history_for_estimator_and_current_frame_for_actor(self):
        indices = bsrl_estimator_actor_indices()
        self.assertEqual((len(indices), max(indices)), (48, 143))
        cfg = SACCfg(
            obs_dim=144,
            action_dim=12,
            actor_input_indices=indices,
            estimator_output_dim=8,
        )
        agent = SACAgent(cfg)
        observation = torch.randn(5, 144)
        self.assertEqual(agent.estimate(observation).shape, (5, 8))
        self.assertEqual(agent.augment_observation(observation).shape, (5, 56))
        self.assertEqual(agent.act(observation, deterministic=True).shape, (5, 12))

    def test_original_t1_dimensions_remain_compatible(self):
        spec = WorldModelStateSpec()
        cfg = ResidualWorldModelCfg(state=spec)
        self.assertEqual((spec.dynamic_dim, spec.context_dim, spec.frame_dim), (68, 75, 143))
        self.assertEqual((cfg.model_input_dim, cfg.output_dim, cfg.uncertainty_dim), (86, 31, 60))


if __name__ == "__main__":
    unittest.main()
