"""Pure tensor tests: reset must not masquerade as closure recovery."""
import importlib.util
from pathlib import Path

import torch


path = Path(__file__).resolve().parents[3] / "scripts/rsl_rl/closed_ankle_diagnostics.py"
spec = importlib.util.spec_from_file_location("closure_diagnostic_stats", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recovery_is_distinguished_from_reset():
    stats = module.ClosureStatistics(3, "cpu")
    stats.observe(torch.tensor([.02, .005, .02]))
    stats.observe(torch.tensor([.02, .02, .005]))
    stats.reset(torch.tensor([0]))
    stats.observe(torch.tensor([.005, .02, .005]))
    result = stats.summary(.001)
    assert result["recovered_without_reset_streaks"] == 1
    assert result["reset_censored_streaks"] == 1
    assert result["still_open_over10_streaks"] == 1
    assert result["longest_over10_ms"] == 2
    assert abs(result["env_time_over10_percent"] - 500 / 9) < 1e-6


def test_threshold_and_all_env_reset():
    stats = module.ClosureStatistics(2, "cpu")
    stats.observe(torch.tensor([.01, .033]))
    stats.reset(slice(None))
    stats.observe(torch.zeros(2))
    result = stats.summary(.005)
    assert result["recovered_without_reset_streaks"] == 0
    assert result["reset_censored_streaks"] == 1
    assert result["ever_over10_env_percent"] == 50
    assert result["longest_over10_ms"] == 5
