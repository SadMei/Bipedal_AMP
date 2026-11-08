"""Fast configuration and dataset tests for the R2 LAFAN ablations."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bsrl_r2_lafan_ablation import (  # noqa: E402
    ACCAD_FORWARD_MOTIONS, BASELINE_R2_RUN, LAFAN_MOTIONS, TASK_STYLE_LERP, VARIANTS,
    expected_configuration, motion_weights, prepare_dataset,
)
from t1_heading_ablation import configuration_differences, load_saved_config  # noqa: E402


class R2LafanAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = SCRIPT_DIR.parents[1]
        baseline = cls.root / BASELINE_R2_RUN
        cls.saved_env = load_saved_config(baseline / "params/env.yaml")
        cls.saved_agent = load_saved_config(baseline / "params/agent.yaml")

    def build(self, variant):
        dataset, manifest = prepare_dataset(self.root, variant)
        env, agent = expected_configuration(
            self.saved_env, self.saved_agent, variant=variant, dataset_dir=dataset,
            run_name=f"test_{variant}", num_envs=128, max_iterations=3, device="cuda:0",
        )
        return dataset, manifest, env, agent

    def test_dataset_views_and_sampling_mass(self):
        expected_mass = {
            VARIANTS[0]: (1.0, 0.0), VARIANTS[1]: (0.8, 0.2), VARIANTS[2]: (0.0, 1.0),
        }
        for variant in VARIANTS:
            dataset, manifest, _, _ = self.build(variant)
            self.assertAlmostEqual(sum(motion_weights(variant).values()), 1.0)
            self.assertAlmostEqual(manifest["sampling_mass"]["lafan"], expected_mass[variant][0])
            self.assertAlmostEqual(manifest["sampling_mass"]["accad_forward_stand"], expected_mass[variant][1])
            self.assertEqual({path.stem for path in dataset.glob("*.pkl")}, set(motion_weights(variant)))
            self.assertTrue(all(path.is_symlink() for path in dataset.glob("*.pkl")))

    def test_all_variants_preserve_r2_except_requested_style_changes(self):
        for variant in VARIANTS:
            dataset, _, env, agent = self.build(variant)
            self.assertEqual(env["motion_data"]["motion_dataset"]["motion_data_dir"], str(dataset.resolve()))
            self.assertEqual(env["motion_data"]["motion_dataset"]["motion_data_weights"], motion_weights(variant))
            self.assertIsNone(env["animation"]["animation"]["command_name"])
            self.assertIsNone(env["animation"]["animation"]["command_motion_patterns"])
            self.assertEqual(agent["algorithm"]["amp_cfg"]["amp_discriminator"]["task_style_lerp"],
                             TASK_STYLE_LERP[variant])
            self.assertFalse(agent["resume"])
            self.assertIsNotNone(env["events"]["ankle_transmission"])
            expected_env = copy.deepcopy(self.saved_env)
            expected_env["motion_data"] = copy.deepcopy(env["motion_data"])
            expected_env["animation"] = copy.deepcopy(env["animation"])
            expected_env["scene"]["num_envs"] = 128
            expected_env["scene"]["terrain"]["num_envs"] = 128
            expected_env["sim"]["device"] = "cuda:0"
            expected_env["log_dir"] = None
            self.assertEqual(configuration_differences(expected_env, env), [])

    def test_variant_contents(self):
        self.assertEqual(set(motion_weights(VARIANTS[0])), set(LAFAN_MOTIONS))
        self.assertEqual(set(motion_weights(VARIANTS[1])), set(LAFAN_MOTIONS + ACCAD_FORWARD_MOTIONS))
        self.assertEqual(set(motion_weights(VARIANTS[2])), set(ACCAD_FORWARD_MOTIONS))


if __name__ == "__main__":
    unittest.main()
