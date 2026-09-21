"""CPU smoke tests for relocated tools; no Isaac app or original solver needed."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "scripts/tools"
MODULES = [
    f"scripts.tools.{group}.{path.stem}"
    for group in ("bsrl_rosbags", "parallel_ankle")
    for path in sorted((TOOLS_ROOT / group).glob("*.py"))
    if not path.name.startswith(("test_", "__"))
]
ENTRYPOINTS = [
    "bsrl_rosbags/export_bags.py",
    "bsrl_rosbags/analyze_bags.py",
    "bsrl_rosbags/detail_checks.py",
    "parallel_ankle/build_models.py",
    "parallel_ankle/build_preview_urdf.py",
    "parallel_ankle/check_mounting.py",
    "parallel_ankle/parallel_ankle.py",
    "parallel_ankle/parallel_ankle_approximate.py",
    "parallel_ankle/replay_policy.py",
    "parallel_ankle/render_replay.py",
    "parallel_ankle/restore_models.py",
]


@pytest.fixture
def importer(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    return importlib.import_module


def subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    return env


@pytest.mark.parametrize("name", MODULES)
def test_package_import(name, importer):
    module = importer(name)
    assert Path(module.__file__).resolve().is_relative_to(TOOLS_ROOT)


@pytest.mark.parametrize("entry", ENTRYPOINTS)
@pytest.mark.parametrize("package_mode", [False, True], ids=["direct", "module"])
def test_help_from_unrelated_directory(entry, package_mode, tmp_path):
    target = ["-m", "scripts.tools." + entry.removesuffix(".py").replace("/", ".")]
    if not package_mode:
        target = [str(TOOLS_ROOT / entry)]
    result = subprocess.run(
        [sys.executable, *target, "--help"], cwd=tmp_path, env=subprocess_env(),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert not list(tmp_path.iterdir()), "--help must not generate models or results"


def test_external_paths_and_cpp_interfaces(importer):
    bags = importer("scripts.tools.bsrl_rosbags._paths")
    ankle = importer("scripts.tools.parallel_ankle._paths")
    real = importer("scripts.tools.parallel_ankle.real_ankle")
    cad = importer("scripts.tools.parallel_ankle.cad_assets")
    models = importer("scripts.tools.parallel_ankle.build_models")
    approximate = importer("scripts.tools.parallel_ankle.parallel_ankle_approximate")
    assert bags.REPO_ROOT == ankle.REPO_ROOT == REPO_ROOT
    assert bags.WORKSPACE_ROOT == ankle.WORKSPACE == REPO_ROOT.parent
    assert bags.RESULTS_ROOT == bags.BAG_ROOT / "analysis/results_20260914"
    robot_data = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots"
    assert ankle.DATA_DIR == real.HERE == cad.HERE == robot_data / "BSRL_closed_ankle"
    assert cad.URDF == robot_data / "BSRL_urdf/urdf/export.urdf"
    assert models.SERIAL_XML == robot_data / "BSRL_urdf/xml/bsrl_scene.xml"
    assert approximate.MODEL == ankle.DATA_DIR / "parallel_ankle_approximate.xml"
    assert approximate.OUTPUT_DIR == ankle.DATA_DIR / "results/approximate"
    assert (TOOLS_ROOT / "bsrl_rosbags/ankle_batch.cpp").is_file()
    assert (ankle.SCRIPT_DIR / "core_bridge.cpp").is_file()


@pytest.mark.parametrize("group", ["bsrl_rosbags", "parallel_ankle"])
def test_external_dependency_overrides(group, tmp_path):
    env = subprocess_env()
    env.update(BSRL_ANKLE_CORE_DIR=str(tmp_path / "original-core"),
               BSRL_ROSBAG_ROOT=str(tmp_path / "bags"))
    code = f"from scripts.tools.{group} import _paths as p; "
    if group == "bsrl_rosbags":
        code += "print(p.CORE_ROOT); print(p.BAG_ROOT); print(p.RESULTS_ROOT)"
        expected = [tmp_path / "original-core", tmp_path / "bags", tmp_path / "bags/analysis/results_20260914"]
    else:
        code += "print(p.CORE); print(p.CALIBRATION)"
        expected = [tmp_path / "original-core", tmp_path / "original-core/bsrl_bridge_bringup.yaml"]
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(map(str, expected))


def test_missing_original_solver_is_explicit(importer, monkeypatch, tmp_path):
    bags = importer("scripts.tools.bsrl_rosbags.analyze_bags")
    real = importer("scripts.tools.parallel_ankle.real_ankle")
    missing = tmp_path / "missing-original-core"
    with pytest.raises(FileNotFoundError, match="BSRL_ANKLE_CORE_DIR"):
        bags.compile_core(missing, tmp_path / "build")
    monkeypatch.setattr(real, "CORE", missing)
    with pytest.raises(FileNotFoundError, match="BSRL_ANKLE_CORE_DIR"):
        real.RobotCore()
    assert not list(tmp_path.iterdir())


def test_torque_audit_uses_relocated_analysis(importer, monkeypatch, tmp_path):
    bags = importer("scripts.tools.bsrl_rosbags.analyze_bags")
    audit = importer("scripts.rsl_rl.summarize_torque_audit")
    (tmp_path / "manifest.json").write_text(json.dumps({"joint_names": [], "physics_dt": .005}))

    class ImportVerified(Exception):
        pass

    def stop_before_compilation(core, build):
        assert core == bags.CORE_ROOT
        assert build.is_dir()
        raise ImportVerified

    monkeypatch.setattr(bags, "compile_core", stop_before_compilation)
    monkeypatch.setattr(sys, "argv", [audit.__file__, str(tmp_path)])
    with pytest.raises(ImportVerified):
        audit.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_cad_read_keeps_original_assets(importer, side):
    cad = importer("scripts.tools.parallel_ankle.cad_assets")
    if not cad.URDF.is_file():
        pytest.skip("External BSRL_urdf is not installed")
    geometry = cad.load_cad(side)
    assert geometry["mounts"].shape == geometry["pivots"].shape == (2, 3)
    assert sum(len(mesh.faces) for mesh in geometry["groups"].values()) == len(geometry["mesh"].faces)
    assert cad.asset_directory().parent == cad.HERE / "assets"


def test_approximate_model_still_loads(importer):
    demo = importer("scripts.tools.parallel_ankle.parallel_ankle_approximate")
    if not demo.MODEL.is_file():
        pytest.skip("External approximate XML is not installed")
    model = demo.mujoco.MjModel.from_xml_path(str(demo.MODEL))
    assert model.nu == 2
    assert model.joint("ankle_pitch").id != model.joint("ankle_roll").id
