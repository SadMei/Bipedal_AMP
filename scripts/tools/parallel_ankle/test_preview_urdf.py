"""Independent URDF parser/FK checks for the viewer-only ankle overlay."""
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

if __package__:
    from .build_preview_urdf import OUTPUT, SOURCE, build_preview
    from .real_ankle import Geometry, RobotCore
else:
    from build_preview_urdf import OUTPUT, SOURCE, build_preview
    from real_ankle import Geometry, RobotCore


@pytest.fixture(scope="module")
def preview():
    core = RobotCore()
    text = build_preview(core)
    return core, text, pin.buildModelFromXML(text)


def structural_key(element):
    return element.tag, element.attrib, (element.text or "").strip(), [structural_key(e) for e in element]


def test_source_unchanged_and_no_extra_dynamic_joints(preview):
    core, text, model = preview
    original = ET.parse(SOURCE).getroot()
    root = ET.fromstring(text)
    for tag in ("link", "joint"):
        for element in original.findall(tag):
            assert structural_key(element) == structural_key(root.find(f"{tag}[@name='{element.get('name')}']"))
    added_links = [link for link in root.findall("link") if link.get("name").startswith("preview_")]
    assert len(added_links) == 12
    assert all(link.find("collision") is None and link.find("inertial") is None for link in added_links)
    joints = [joint for joint in root.findall("joint") if joint.get("name").startswith("preview_")]
    assert len(joints) == 12 and all(j.get("type") == "fixed" for j in joints)
    assert model.nq == model.nv == 12
    for mesh in root.findall(".//mesh"):
        assert (OUTPUT.parent / mesh.get("filename")).is_file()
        assert not mesh.get("filename").startswith("/")
    with pytest.raises(ValueError, match="overwrite"):
        build_preview(core, output=SOURCE)


@pytest.mark.parametrize("side", ["right", "left"])
def test_urdf_zero_pose_matches_mujoco_geometry(preview, side):
    core, _, model = preview
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    frames = lambda name: data.oMf[model.getFrameId(name)]
    shank, foot = frames(f"link_{side}_shank"), frames(f"link_{side}_ankle_roll")
    geometry = Geometry(side, core)
    ankle_offset = shank.actInv(foot.translation)
    for i, branch in enumerate(("long", "short")):
        prefix = f"preview_{side}_{branch}"
        crank, rod, mount = (frames(f"{prefix}_{part}") for part in ("crank", "rod", "foot_mount"))
        np.testing.assert_allclose(shank.actInv(crank.translation), ankle_offset + geometry.pivots[i], atol=1e-12)
        np.testing.assert_allclose(shank.rotation.T @ crank.rotation, geometry.crank_rotation(i, 0.), atol=1e-12)
        np.testing.assert_allclose(shank.actInv(rod.translation), ankle_offset + geometry.crank_end(i, 0.), atol=1e-12)
        np.testing.assert_allclose(foot.actInv(mount.translation), geometry.mounts[i], atol=1e-12)
        np.testing.assert_allclose(rod.act(np.array([0., 0., geometry.l4[i]])), mount.translation, atol=1e-12)
    # Also load all OBJ/primitives through a real URDF geometry parser.
    geometries = pin.buildGeomFromUrdf(model, str(OUTPUT), pin.GeometryType.VISUAL,
                                      package_dirs=[str(OUTPUT.parent)])
    assert len(geometries.geometryObjects) == len(ET.parse(OUTPUT).findall(".//visual"))
