"""Check linkage installation against the independent, unchanged URDF CAD.

The CAD is an orientation reference, not a substitute for the hardware YAML's
dimensions. In particular, it has different crank lengths and lateral spacing.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

if __package__:
    from ._paths import URDF
    from .real_ankle import Geometry, RobotCore
else:
    from _paths import URDF
    from real_ankle import Geometry, RobotCore


def cad_mesh(side, part):
    """Return one visual mesh in the standard, q=0 ankle coordinate frame."""
    doc = ET.parse(URDF)
    visual = doc.find(f"link[@name='link_{side}_{part}']/visual")
    origin = visual.find("origin")
    mesh_spec = visual.find("geometry/mesh")
    mesh = trimesh.load(URDF.parent / mesh_spec.get("filename"), force="mesh")
    scale = np.fromstring(mesh_spec.get("scale", "1 1 1"), sep=" ")
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    mesh.vertices = (mesh.vertices * scale) @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    if part == "shank":
        joint = doc.find(f"joint[@name='joint_{side}_ankle_pitch']/origin")
        np.testing.assert_allclose(np.fromstring(joint.get("rpy", "0 0 0"), sep=" "), 0)
        mesh.vertices -= np.fromstring(joint.get("xyz"), sep=" ")
    return mesh


def cad_rod_ends(mesh):
    """Identify the four spherical bearings by their independent CAD envelopes.

    Bearings are ~11 mm in X/Z and ~8 mm in Y. If future CAD changes these
    envelopes, fail explicitly instead of silently inferring a new installation.
    """
    centers = []
    for part in mesh.split(only_watertight=False):
        extent = part.extents
        center = part.bounds.mean(axis=0)
        if (center[0] < -.05 and .010 < extent[0] < .012
                and .007 < extent[1] < .009 and .010 < extent[2] < .012):
            centers.append(center)
    centers = np.asarray(centers)
    if centers.shape != (4, 3):
        raise ValueError(f"Expected four CAD rod bearings; got {centers.shape}")
    lower = centers[centers[:, 2] < .04]
    upper = centers[centers[:, 2] > .1]
    if len(lower) != 2 or len(upper) != 2:
        raise ValueError("CAD bearing heights do not identify two rods")
    upper = upper[np.argsort(-upper[:, 2])]  # long, short
    lower = np.array([lower[np.argmin(np.abs(lower[:, 1] - end[1]))] for end in upper])
    return lower, upper


def audit(geometry, mesh):
    lower, upper = cad_rod_ends(mesh)
    generated_upper = np.array([geometry.crank_end(i, 0.) for i in range(2)])
    # Current default geometry uses these user-confirmed CAD-zero points.
    assert np.all(lower[:, 0] < 0) and np.all(geometry.mounts[:, 0] < 0)
    np.testing.assert_array_equal(np.sign(geometry.mounts[:, 1]), np.sign(lower[:, 1]))
    np.testing.assert_array_equal(np.sign(generated_upper[:, 1]), np.sign(upper[:, 1]))
    lower_error = np.linalg.norm(geometry.mounts - lower, axis=1)
    upper_error = np.linalg.norm(generated_upper - upper, axis=1)
    assert lower_error.max() < 1e-9
    assert upper_error.max() < 1e-9
    return dict(
        side=geometry.side, frame="URDF ankle zero: +X toe, +Y left, +Z up",
        order=["long", "short"], cad_foot_bearing_m=lower.tolist(),
        cad_crank_bearing_m=upper.tolist(), generated_foot_bearing_m=geometry.mounts.tolist(),
        generated_crank_bearing_m=generated_upper.tolist(),
        foot_endpoint_offset_mm=(lower_error * 1000).tolist(),
        crank_endpoint_offset_mm=(upper_error * 1000).tolist(),
        interpretation="generated connection points match user-confirmed CAD zero; hardware YAML is unchanged",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    core = RobotCore()
    reports = []
    if args.plot:
        if args.plot.exists():
            raise FileExistsError(args.plot)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, side in enumerate(("right", "left")):
        geometry = Geometry(side, core)
        shank, foot = cad_mesh(side, "shank"), cad_mesh(side, "ankle_roll")
        reports.append(audit(geometry, shank))
        if not args.plot:
            continue
        for col, (horizontal, vertical) in enumerate(((0, 2), (0, 1))):
            ax = axes[row, col]
            for mesh in (shank, foot):
                vertices = mesh.vertices
                vertices = vertices[vertices[:, 2] < .29][::4] * 1000
                ax.scatter(vertices[:, horizontal], vertices[:, vertical], s=.15, color=".55", alpha=.35)
            for i, color in enumerate(("tab:blue", "tab:orange")):
                points = np.array([geometry.pivots[i], geometry.crank_end(i, 0.), geometry.mounts[i]])
                ax.plot(points[:, horizontal] * 1000, points[:, vertical] * 1000, "o-", color=color,
                        linewidth=2, markersize=4, label=("long" if i == 0 else "short") + " corrected")
                old = points @ geometry.installation_rotation
                ax.plot(old[:, horizontal] * 1000, old[:, vertical] * 1000, "--", color=color,
                        alpha=.5, label=("long" if i == 0 else "short") + " previous (wrong)")
            ax.set_title(f"{side} ankle | {'side' if col == 0 else 'top'} view | q=0")
            ax.set_xlabel("X / mm (toe ->)")
            ax.set_ylabel(("Z" if col == 0 else "Y") + " / mm")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    print(json.dumps(reports, indent=2))
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle("Unchanged URDF CAD (gray): linkage installation Rz(180 deg) correction")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
