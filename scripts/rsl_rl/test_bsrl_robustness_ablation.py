"""Fast unit tests for the BSRL robustness experiment contract."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

from isaaclab.app import AppLauncher

_APP = AppLauncher(headless=True).app

from isaaclab.utils.modifiers import ModifierCfg

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bsrl_robustness_ablation import (  # noqa: E402
    BASELINE_RUN, OBSERVATION_PERTURBATIONS, TRANSMISSION_RANGES, VARIANTS,
    expected_configuration,
)
from t1_heading_ablation import configuration_differences, load_saved_config  # noqa: E402
from legged_lab.tasks.locomotion.amp.mdp.observation_modifiers import (  # noqa: E402
    EpisodeBiasDelay, EpisodeBiasDelayCfg,
)


class RobustnessConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = SCRIPT_DIR.parents[1]
        baseline = cls.root / BASELINE_RUN
        cls.saved_env = load_saved_config(baseline / "params/env.yaml")
        cls.saved_agent = load_saved_config(baseline / "params/agent.yaml")

    def build(self, variant):
        return expected_configuration(
            self.saved_env, self.saved_agent, variant=variant, run_name=f"test_{variant}",
            num_envs=128, max_iterations=3, device="cuda:0",
        )

    def test_r1_contract(self):
        env, agent = self.build(VARIANTS[0])
        reset = env["events"]["reset_from_ref"]
        self.assertTrue(reset["func"].endswith(":reset_from_ref_with_default_mixture"))
        self.assertEqual(reset["params"]["default_start_probability"], 0.2)
        self.assertEqual(env["events"]["push_robot"]["interval_range_s"], (2.5, 5.0))
        self.assertEqual(env["events"]["push_robot"]["params"]["velocity_range"]["yaw"], (-0.2, 0.2))
        self.assertEqual(agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"], 0.6)
        self.assertIsNone(env["events"]["ankle_transmission"])
        self.assertFalse(agent["resume"])

    def test_r2_only_adds_transmission_to_r1_training_contract(self):
        r1_env, r1_agent = self.build(VARIANTS[0])
        r2_env, r2_agent = self.build(VARIANTS[1])
        r1_env["events"]["ankle_transmission"] = copy.deepcopy(r2_env["events"]["ankle_transmission"])
        r1_agent["run_name"] = r2_agent["run_name"]
        self.assertEqual(r2_env["events"]["ankle_transmission"]["params"], TRANSMISSION_RANGES)
        self.assertEqual(configuration_differences(r1_env, r2_env), [])
        self.assertEqual(configuration_differences(r1_agent, r2_agent), [])

    def test_r3_only_adds_policy_modifiers_to_r2_training_contract(self):
        r2_env, r2_agent = self.build(VARIANTS[1])
        r3_env, r3_agent = self.build(VARIANTS[2])
        for term in OBSERVATION_PERTURBATIONS:
            r2_env["observations"]["policy"][term]["modifiers"] = copy.deepcopy(
                r3_env["observations"]["policy"][term]["modifiers"]
            )
            self.assertIsNone(r3_env["observations"]["critic"][term]["modifiers"])
        r2_agent["run_name"] = r3_agent["run_name"]
        self.assertEqual(configuration_differences(r2_env, r3_env), [])
        self.assertEqual(configuration_differences(r2_agent, r3_agent), [])


class EpisodeBiasDelayTests(unittest.TestCase):
    def make(self, bias=(0.0, 0.0), delay=1):
        cfg = EpisodeBiasDelayCfg(bias_range=bias, max_delay_steps=delay)
        self.assertIsInstance(cfg, ModifierCfg)
        return EpisodeBiasDelay(cfg, (4, 2), "cpu")

    def test_first_sample_after_reset_never_uses_zero_padding(self):
        modifier = self.make(bias=(0.25, 0.25))
        modifier.delay_steps.fill_(1)
        data = torch.arange(8, dtype=torch.float).reshape(4, 2) + 1
        torch.testing.assert_close(modifier(data), data + 0.25)

    def test_zero_and_one_step_delay(self):
        modifier = self.make()
        modifier.delay_steps[:] = torch.tensor([0, 1, 0, 1])
        first = torch.arange(8, dtype=torch.float).reshape(4, 2)
        second = first + 10
        modifier(first)
        expected = second.clone()
        expected[[1, 3]] = first[[1, 3]]
        torch.testing.assert_close(modifier(second), expected)

    def test_partial_reset_resamples_only_selected_rows_and_primes_them(self):
        torch.manual_seed(5)
        modifier = self.make(bias=(-1.0, 1.0))
        modifier.delay_steps.fill_(1)
        first = torch.arange(8, dtype=torch.float).reshape(4, 2)
        modifier(first)
        old_bias = modifier.bias.clone()
        modifier.reset(torch.tensor([1, 3]))
        torch.testing.assert_close(modifier.bias[[0, 2]], old_bias[[0, 2]])
        current = first + 20
        result = modifier(current)
        torch.testing.assert_close(result[[1, 3]], current[[1, 3]] + modifier.bias[[1, 3]])
        torch.testing.assert_close(result[[0, 2]], first[[0, 2]] + modifier.bias[[0, 2]])


if __name__ == "__main__":
    try:
        unittest.main(exit=False)
    finally:
        _APP.close()
