"""Reconstruct lost CAD-zero MJCF from surviving code and archived CAD meshes.

This is generated-model recovery, not bitwise undelete or hardware validation.
No replacement C++ solver is fabricated. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

if __package__:
    from ._paths import CLOSED_ANKLE_ROOT, REPO_ROOT, SCRIPT_DIR, URDF
    from .build_models import SERIAL_XML, full_robot_xml
    from .cad_assets import asset_directory, load_cad
    from .real_ankle import Geometry, closure_gaps, standalone_xml, xml_string
else:
    from _paths import CLOSED_ANKLE_ROOT, REPO_ROOT, SCRIPT_DIR, URDF
    from build_models import SERIAL_XML, full_robot_xml
    from cad_assets import asset_directory, load_cad
    from real_ankle import Geometry, closure_gaps, standalone_xml, xml_string


GROUPS = ("shank", "long_crank", "long_rod", "long_foot", "short_crank", "short_rod", "short_foot")
MODEL_NAMES = ("bsrl_closed_ankles.xml", "parallel_ankle.xml", "parallel_ankle_left.xml")
MANIFEST_NAME = "parallel_ankle_recovery.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_inputs():
    """Require the archived partition to agree with the surviving original CAD."""
    snapshot = REPO_ROOT / "source/legged_lab/legged_lab/data/Robots/BSRL_closed_ankle" / asset_directory().name
    geometries, visuals = {}, {}
    for side in ("right", "left"):
        report = json.loads((snapshot / f"{side}_partition.json").read_text())
        geometry = Geometry.from_cad(side)
        cad = load_cad(side)
        if report["side"] != side or report["source_faces"] != len(cad["mesh"].faces):
            raise ValueError(f"Archived CAD partition differs: {side}")
        for key, points in (("mounts_m", geometry.mounts), ("pivots_m", geometry.pivots),
                            ("crank_ends_m", geometry.zero_crank_ends)):
            np.testing.assert_allclose(report[key], points, atol=1e-12, rtol=0,
                                       err_msg=f"Archived CAD points differ: {side}/{key}")
        if report["group_faces"] != {key: len(mesh.faces) for key, mesh in cad["groups"].items()}:
            raise ValueError(f"Archived CAD face partition differs: {side}")
        paths = {group: snapshot / f"{side}_{group}.obj" for group in GROUPS}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        geometries[side], visuals[side] = geometry, paths
    return snapshot, geometries, visuals


def prepare_models():
    snapshot, geometries, visuals = recovery_inputs()
    root = ET.fromstring(full_robot_xml(None, geometries=geometries, visual_assets=visuals))
    source_meshdir = Path(root.find("compiler").get("meshdir"))
    root.find("compiler").set("meshdir", "../meshes/")
    copies = {}
    for mesh in root.findall("./asset/mesh"):
        source = source_meshdir / mesh.get("file")
        # CAD files are archived byte-for-byte, not re-exported or reshaped.
        relative = Path("parallel_ankle") / snapshot.name / source.name if source.parent == snapshot else Path(source.name)
        mesh.set("file", relative.as_posix())
        destination = Path("meshes") / relative
        if destination in copies and copies[destination] != source:
            raise ValueError(f"Ambiguous mesh destination: {destination}")
        copies[destination] = source
    for side in ("right", "left"):
        name = f"{side}_partition.json"
        copies[Path("meshes/parallel_ankle") / snapshot.name / name] = snapshot / name
    texts = {
        MODEL_NAMES[0]: xml_string(root) + "\n",
        MODEL_NAMES[1]: standalone_xml(geometries["right"]) + "\n",
        MODEL_NAMES[2]: standalone_xml(geometries["left"]) + "\n",
    }
    return texts, copies, geometries


def validate_models(robot_root, geometries):
    """Check portable loading, passive ankles, original inertias and 882 poses."""
    full = mujoco.MjModel.from_xml_path(str(robot_root / "xml" / MODEL_NAMES[0]))
    serial = mujoco.MjModel.from_xml_path(str(SERIAL_XML))
    if full.nu != 12 or full.neq != 4:
        raise ValueError(f"Unexpected closed-chain topology: nu={full.nu}, neq={full.neq}")
    for body_id in range(1, serial.nbody):
        name = serial.body(body_id).name
        for field in ("mass", "inertia", "ipos", "iquat"):
            np.testing.assert_allclose(getattr(full.body(name), field), getattr(serial.body(name), field),
                                       atol=1e-12, rtol=0, err_msg=f"Original inertia changed: {name}/{field}")
    np.testing.assert_allclose(full.body_mass.sum() - serial.body_mass.sum(), .008, atol=1e-12, rtol=0)
    for side in ("right", "left"):
        for axis in ("pitch", "roll"):
            jid = full.joint(f"joint_{side}_ankle_{axis}").id
            if np.any(full.actuator_trnid[:, 0] == jid):
                raise ValueError(f"Passive ankle has a direct actuator: {side}/{axis}")
    data = mujoco.MjData(full)
    mujoco.mj_forward(full, data)
    zero_gap = float(closure_gaps(full, data, ["right_", "left_"]).max())
    if zero_gap >= 1e-12:
        raise ValueError(f"Full-model zero pose is not closed: {zero_gap}")
    worst = 0.
    poses = 0
    for side, name in zip(("right", "left"), MODEL_NAMES[1:]):
        geometry = geometries[side]
        model = mujoco.MjModel.from_xml_path(str(robot_root / "xml" / name))
        state = mujoco.MjData(model)
        for pitch in np.linspace(-.6, .6, 21):
            for roll in np.linspace(-.52, .52, 21):
                q = np.array([pitch, roll])
                mujoco.mj_resetData(model, state)
                state.joint("ankle_pitch").qpos[0], state.joint("ankle_roll").qpos[0] = q
                geometry.initialize(model, state, q, "")
                worst = max(worst, float(closure_gaps(model, state, [""]).max()))
                poses += 1
    if worst >= 1e-12:
        raise ValueError(f"Standalone workspace does not close: {worst}")
    return dict(actuators=full.nu, closure_constraints=full.neq, passive_ankle_direct_actuators=0,
                total_mass_kg=float(full.body_mass.sum()), auxiliary_mass_kg=.008,
                full_zero_gap_m=zero_gap, workspace_poses=poses, workspace_max_gap_m=worst)


def restore(robot_root):
    robot_root = Path(robot_root).resolve()
    output_names = (*MODEL_NAMES, MANIFEST_NAME)
    for name in output_names:
        destination = robot_root / "xml" / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    texts, copies, geometries = prepare_models()
    # Check every existing asset before publishing any new file.
    for relative, source in copies.items():
        destination = robot_root / relative
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or sha(destination) != sha(source):
                raise FileExistsError(f"Existing asset differs; refusing to overwrite {destination}")
    with tempfile.TemporaryDirectory(prefix="bsrl-mjcf-recovery-") as temporary:
        staged = Path(temporary) / "BSRL_urdf"
        (staged / "xml").mkdir(parents=True)
        for relative, source in copies.items():
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name, value in texts.items():
            (staged / "xml" / name).write_text(value)
        validation = validate_models(staged, geometries)
        manifest = dict(
            recovery_kind="reconstructed_from_surviving_generator_and_archived_CAD",
            bitwise_original_xml_recovery_verified=False,
            hardware_cpp_calibration_verified=False,
            historical_trajectories_recovered=False,
            geometry_revision=Geometry.revision,
            unchanged_serial_xml=dict(path=str(SERIAL_XML), sha256=sha(SERIAL_XML)),
            original_urdf=dict(path=str(URDF), sha256=sha(URDF)),
            source_code_sha256={name: sha(SCRIPT_DIR / name) for name in
                                ("real_ankle.py", "build_models.py", "cad_assets.py", "restore_models.py")},
            restored_assets={str(relative): dict(source=str(source), sha256=sha(source))
                             for relative, source in copies.items()},
            xml_sha256={name: sha(staged / "xml" / name) for name in MODEL_NAMES},
            validation=validation,
        )
        (staged / "xml" / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        for relative in (*copies, *(Path("xml") / name for name in output_names)):
            destination = robot_root / relative
            if destination.is_file() and sha(destination) == sha(staged / relative):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output, (staged / relative).open("rb") as source:
                shutil.copyfileobj(source, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-root", type=Path, default=CLOSED_ANKLE_ROOT,
                        help="Bundle root containing xml/ and meshes/; existing files are preserved")
    args = parser.parse_args()
    result = restore(args.robot_root)
    print(json.dumps(dict(xml=str(args.robot_root.resolve() / "xml" / MODEL_NAMES[0]),
                          recovery_kind=result["recovery_kind"], validation=result["validation"]), indent=2))


if __name__ == "__main__":
    main()
