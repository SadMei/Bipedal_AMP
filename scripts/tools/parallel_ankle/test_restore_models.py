"""Recovery checks without substituting or requiring the lost hardware solver."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree
import trimesh

if __package__:
    from . import real_ankle
    from .cad_assets import load_cad
    from .real_ankle import Geometry, point_z
    from .restore_models import MODEL_NAMES, recovery_inputs, restore, sha
else:
    import real_ankle
    from cad_assets import load_cad
    from real_ankle import Geometry, point_z
    from restore_models import MODEL_NAMES, recovery_inputs, restore, sha


@pytest.fixture(scope="module")
def recovered(tmp_path_factory):
    root = tmp_path_factory.mktemp("recovered-mjcf") / "BSRL_urdf"
    report = restore(root)
    return root, report


def test_recovery_is_explicit_and_topology_preserved(recovered):
    _, report = recovered
    assert report["bitwise_original_xml_recovery_verified"] is False
    assert report["hardware_cpp_calibration_verified"] is False
    assert report["historical_trajectories_recovered"] is False
    checks = report["validation"]
    assert checks["actuators"] == 12
    assert checks["closure_constraints"] == 4
    assert checks["passive_ankle_direct_actuators"] == 0
    assert checks["workspace_poses"] == 882
    assert checks["workspace_max_gap_m"] < 1e-12


def test_all_restored_meshes_and_source_files_are_unchanged(recovered):
    root, report = recovered
    for relative, provenance in report["restored_assets"].items():
        assert sha(root / relative) == sha(provenance["source"]) == provenance["sha256"]
    for key in ("unchanged_serial_xml", "original_urdf"):
        assert sha(report[key]["path"]) == report[key]["sha256"]


def test_bundle_relocates_without_old_workspace(recovered, tmp_path):
    root, _ = recovered
    moved = tmp_path / "another-location"
    shutil.copytree(root, moved)
    for name in MODEL_NAMES:
        path = moved / "xml" / name
        doc = ET.parse(path)
        meshdir = doc.find("compiler").get("meshdir", "")
        assert not Path(meshdir).is_absolute()
        for mesh in doc.findall("./asset/mesh"):
            assert not Path(mesh.get("file")).is_absolute()
            resolved = (path.parent / meshdir / mesh.get("file")).resolve()
            assert resolved.is_relative_to(moved) and resolved.is_file()
        model = mujoco.MjModel.from_xml_path(str(path))
        mujoco.mj_forward(model, mujoco.MjData(model))


def test_existing_model_is_not_overwritten(recovered):
    root, report = recovered
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        restore(root)
    for name, expected in report["xml_sha256"].items():
        assert sha(root / "xml" / name) == expected


def test_conflicting_mesh_is_not_overwritten(tmp_path):
    existing = tmp_path / "meshes/base_waist.obj"
    existing.parent.mkdir()
    existing.write_text("user-owned mesh\n")
    with pytest.raises(FileExistsError, match="Existing asset differs"):
        restore(tmp_path)
    assert existing.read_text() == "user-owned mesh\n"
    assert not (tmp_path / "xml").exists()


def test_default_core_checks_are_not_bypassed(monkeypatch, tmp_path):
    monkeypatch.setattr(real_ankle, "CORE", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="BSRL_ANKLE_CORE_DIR"):
        real_ankle.RobotCore()
    cad = Geometry.from_cad("right")
    assert cad.core is None and cad.calibration is None
    with pytest.raises(ValueError):
        Geometry.from_cad("invalid")


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("q", [(0., 0.), (-.2, 0.), (.4, .3), (-.4, -.3)])
def test_cad_jacobian_matches_finite_difference_and_mujoco(recovered, side, q):
    root, _ = recovered
    geometry = Geometry.from_cad(side)
    q = np.array(q)
    motor, jacobian = geometry.forward(q)
    numerical = np.column_stack([(geometry.forward(q + delta)[0] - geometry.forward(q - delta)[0]) / 2e-6
                                 for delta in np.eye(2) * 1e-6])
    np.testing.assert_allclose(jacobian, numerical, atol=2e-8, rtol=0)
    np.testing.assert_allclose(geometry.inverse(motor, np.zeros(2)), q, atol=1e-8, rtol=0)
    name = MODEL_NAMES[1 if side == "right" else 2]
    model = mujoco.MjModel.from_xml_path(str(root / "xml" / name))
    data = mujoco.MjData(model)
    data.joint("ankle_pitch").qpos[0], data.joint("ankle_roll").qpos[0] = q
    geometry.initialize(model, data, q, "")
    mujoco.mj_forward(model, data)
    constraints = []
    for branch in (1, 2):
        rod, foot, unused = (np.zeros((3, model.nv)) for _ in range(3))
        mujoco.mj_jacSite(model, data, rod, unused, model.site(f"rod_end_{branch}").id)
        mujoco.mj_jacSite(model, data, foot, unused, model.site(f"foot_end_{branch}").id)
        constraints.append(rod - foot)
    constraint = np.vstack(constraints)
    ankles = [model.joint(name).dofadr[0] for name in ("ankle_pitch", "ankle_roll")]
    other = [index for index in range(model.nv) if index not in ankles]
    implied = np.linalg.lstsq(constraint[:, other], -constraint[:, ankles], rcond=None)[0]
    motors = [other.index(model.joint(f"motor_{branch}").dofadr[0]) for branch in (1, 2)]
    np.testing.assert_allclose(implied[motors], jacobian, atol=2e-8, rtol=0)


@pytest.mark.parametrize("side", ["right", "left"])
def test_archived_visuals_match_current_cad_without_deformation(side):
    _, geometries, visuals = recovery_inputs()
    geometry = geometries[side]
    cad = load_cad(side)
    frames = {"shank": (np.zeros(3), np.eye(3))}
    for index, branch in enumerate(("long", "short")):
        frames[f"{branch}_crank"] = geometry.pivots[index], geometry.crank_rotation(index, 0.)
        frames[f"{branch}_rod"] = (geometry.crank_end(index, 0.),
                                   point_z(geometry.mounts[index] - geometry.crank_end(index, 0.)))
        frames[f"{branch}_foot"] = geometry.mounts[index], np.eye(3)
    assert sum(len(mesh.faces) for mesh in cad["groups"].values()) == len(cad["mesh"].faces)
    for group, source in cad["groups"].items():
        archived = trimesh.load(visuals[side][group], force="mesh", process=False)
        assert len(archived.faces) == len(source.faces)
        position, rotation = frames[group]
        reconstructed = archived.vertices @ rotation.T + position
        assert cKDTree(source.vertices).query(reconstructed)[0].max() < 1e-8
        assert cKDTree(reconstructed).query(source.vertices)[0].max() < 1e-8
