"""The local AMP runner supports Isaac Lab with and without config migration."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/rsl_rl"))
from local_source import prepare_agent_cfg_for_local_rsl_rl


@pytest.mark.parametrize("has_migration", [False, True])
@pytest.mark.parametrize("rsl_version", ["3.2.0", "4.0.0"])
def test_isaaclab_migration_is_optional(monkeypatch, has_migration, rsl_version):
    package = ModuleType("isaaclab_rl")
    adapter = ModuleType("isaaclab_rl.rsl_rl")
    package.rsl_rl = adapter
    calls = []
    if has_migration:
        def migrate(cfg, version):
            calls.append(version)
            return cfg
        adapter.handle_deprecated_rsl_rl_cfg = migrate
    monkeypatch.setitem(sys.modules, "isaaclab_rl", package)
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", adapter)
    cfg = SimpleNamespace(algorithm=SimpleNamespace(share_cnn_encoders=True))
    assert prepare_agent_cfg_for_local_rsl_rl(cfg, rsl_version) is cfg
    assert hasattr(cfg.algorithm, "share_cnn_encoders") == (rsl_version == "4.0.0")
    assert calls == ([rsl_version] if has_migration else [])
