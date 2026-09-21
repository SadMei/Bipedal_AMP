"""Regenerate calibrated standalone/full BSRL MJCF without changing serial assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from ._paths import SERIAL_XML
    from .real_ankle import HERE, WORKSPACE, Geometry, RobotCore, standalone_xml, xml_string
    from .cad_assets import export_visuals
else:
    from _paths import SERIAL_XML
    from real_ankle import HERE, WORKSPACE, Geometry, RobotCore, standalone_xml, xml_string
    from cad_assets import export_visuals


def full_robot_xml(core, serial_xml=SERIAL_XML, auxiliary_mass=.001, equality_timeconst=.002,
                   *, geometries=None, visual_assets=None):
    """Build the same topology; explicit inputs also support offline recovery."""
    root = ET.parse(serial_xml).getroot()
    root.set("model", "BSRL CAD-zero connected ankles")
    compiler = root.find("compiler")
    meshdir = (Path(serial_xml).parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", str(meshdir))
    if compiler.get("angle") != "radian":
        raise ValueError("Expected radian BSRL MJCF")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for entry in list(actuator):
        if "ankle" in entry.get("joint", ""):
            actuator.remove(entry)
    for joint in root.findall(".//joint"):
        name = joint.get("name", "")
        if name.endswith(("ankle_pitch", "ankle_roll")):
            joint.attrib.pop("actuatorfrcrange", None)
            joint.set("actuatorfrclimited", "false")
        elif name.startswith("joint_"):
            if not any(a.get("joint") == name for a in actuator):
                ET.SubElement(actuator, "motor", name=f"drive_{name}", joint=name, gear="1")
    for side in ("right", "left"):
        geom = Geometry(side, core) if geometries is None else geometries[side]
        if geom.side != side:
            raise ValueError(f"Geometry side mismatch: expected {side}, got {geom.side}")
        shank = root.find(f".//body[@name='link_{side}_shank']")
        pitch = shank.find(f"body[@name='link_{side}_ankle_pitch']")
        foot = pitch.find(f"body[@name='link_{side}_ankle_roll']")
        np.testing.assert_allclose(np.fromstring(pitch.find("joint").get("axis"), sep=" "), [0,1,0])
        np.testing.assert_allclose(np.fromstring(foot.find("joint").get("axis"), sep=" "), [1,0,0])
        # URDF axes and zero poses are authoritative: never rotate/re-zero them.
        origin = np.fromstring(pitch.get("pos"), sep=" ")
        geom.attach(root, shank, foot, origin, f"{side}_", auxiliary_mass, equality_timeconst)
        paths = export_visuals(geom) if visual_assets is None else visual_assets[side]
        assets = root.find("asset")
        for key, path in paths.items():
            ET.SubElement(assets, "mesh", name=f"cad_{side}_{key}", file=str(path))
        # Remove the old baked rods only from this generated simulation's visual
        # reference. All source meshes, collisions and explicit inertias remain.
        old_visual = shank.find(f"geom[@mesh='link_{side}_shank']")
        old_visual.set("mesh", f"cad_{side}_shank")
        old_visual.set("pos", " ".join(map(str, origin)))
        old_visual.set("quat", "1 0 0 0")
        for i, branch in enumerate(("long", "short"), start=1):
            for kind in ("crank", "rod"):
                body = shank.find(f".//body[@name='{kind}_{side}_{i}']")
                # Keep the existing 1g auxiliary inertia primitive; hide its
                # graphics so there are no duplicate rods or hidden mass changes.
                for primitive in body.findall("geom"):
                    primitive.set("rgba", "0 0 0 0")
                    primitive.set("group", "5")
                ET.SubElement(body, "geom", name=f"cad_visual_{side}_{branch}_{kind}",
                              type="mesh", mesh=f"cad_{side}_{branch}_{kind}",
                              density="0", contype="0", conaffinity="0", group="2",
                              rgba=".15 .55 .95 1" if i == 1 else ".98 .48 .12 1")
            ET.SubElement(foot, "geom", name=f"cad_visual_{side}_{branch}_foot",
                          type="mesh", mesh=f"cad_{side}_{branch}_foot",
                          pos=" ".join(map(str, geom.mounts[i-1])), density="0",
                          contype="0", conaffinity="0", group="2", rgba=".85 .85 .85 1")
    return xml_string(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = RobotCore()
    for side in ("right", "left"):
        path = args.output_dir / ("parallel_ankle.xml" if side=="right" else "parallel_ankle_left.xml")
        path.write_text(standalone_xml(Geometry(side, core))+"\n")
        print(path)
    path = args.output_dir / "bsrl_closed_ankles.xml"
    path.write_text(full_robot_xml(core)+"\n")
    print(path)


if __name__ == "__main__":
    main()
