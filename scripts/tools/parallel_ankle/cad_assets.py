"""Extract user-confirmed zero-pose joints and split CAD visuals without loss.

Source URDF/meshes are read-only. Exported parts are visual-only, do not provide
new mass/inertia, and reconstruct the original mesh exactly in its zero pose.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

if __package__:
    from ._paths import DATA_DIR as HERE, URDF
else:
    from _paths import DATA_DIR as HERE, URDF


def cad_mesh(side, part):
    doc = ET.parse(URDF)
    visual = doc.find(f"link[@name='link_{side}_{part}']/visual")
    origin = visual.find("origin")
    spec = visual.find("geometry/mesh")
    mesh = trimesh.load(URDF.parent / spec.get("filename"), force="mesh")
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    mesh.vertices = (mesh.vertices * np.fromstring(spec.get("scale", "1 1 1"), sep=" ")) @ rotation.T
    mesh.vertices += np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    if part == "shank":
        joint = doc.find(f"joint[@name='joint_{side}_ankle_pitch']/origin")
        np.testing.assert_allclose(np.fromstring(joint.get("rpy", "0 0 0"), sep=" "), 0)
        mesh.vertices -= np.fromstring(joint.get("xyz"), sep=" ")
    return mesh


def is_inner_bearing(part):
    e = part.extents
    return part.bounds.mean(axis=0)[0] < -.05 and .010 < e[0] < .012 and .007 < e[1] < .009 and .010 < e[2] < .012


def cad_rod_ends(mesh):
    centers = np.asarray([p.bounds.mean(axis=0) for p in mesh.split(only_watertight=False, repair=False) if is_inner_bearing(p)])
    if centers.shape != (4, 3):
        raise ValueError(f"Expected four CAD rod bearings; got {centers.shape}")
    lower, upper = centers[centers[:, 2] < .04], centers[centers[:, 2] > .1]
    if len(lower) != 2 or len(upper) != 2:
        raise ValueError("CAD bearing heights do not identify two rods")
    upper = upper[np.argsort(-upper[:, 2])]
    lower = np.array([lower[np.argmin(np.abs(lower[:, 1] - end[1]))] for end in upper])
    return lower, upper


@lru_cache(maxsize=2)
def load_cad(side):
    mesh = cad_mesh(side, "shank")
    parts = mesh.split(only_watertight=False, repair=False)
    lower, upper = cad_rod_ends(mesh)
    centers = np.asarray([p.bounds.mean(axis=0) for p in parts
                          if .077 < p.extents[0] < .078 and .077 < p.extents[2] < .078
                          and .015 < p.extents[1] < .016])
    if centers.shape != (2, 3):
        raise ValueError("Cannot uniquely identify the two CAD motor cylinders")
    centers = centers[np.argsort(-centers[:, 2])]
    # A motor axis is a line along Y. Choose its intersection with the crank
    # bearing's Y plane, rather than the middle of the stationary motor housing.
    centers[:, 1] = upper[:, 1]
    buckets = {"shank": []}
    for branch in ("long", "short"):
        buckets.update({f"{branch}_{kind}": [] for kind in ("crank", "rod", "foot")})
    assignments = []
    for i, part in enumerate(parts):
        c, extent = part.bounds.mean(axis=0), part.extents
        if part.bounds[1, 0] < -.055:
            branch = ("long", "short")[np.argmin(np.abs(lower[:, 1] - c[1]))]
            bolt = extent[0] < .009 and extent[2] < .009
            kind = ("foot" if c[2] < .04 else "crank") if (is_inner_bearing(part) or bolt) else "rod"
            name = f"{branch}_{kind}"
        elif part.bounds[0, 0] < -.08 and part.bounds[1, 0] > 0 and .03 < extent[2] < .04:
            branch = ("long", "short")[np.argmin(np.abs(centers[:, 2] - c[2]))]
            name = f"{branch}_crank"
        else:
            name = "shank"
        buckets[name].append(part)
        assignments.append(dict(component=i, owner=name, faces=len(part.faces)))
    if any(not v for v in buckets.values()):
        raise ValueError("Incomplete CAD component classification")
    assert sum(len(p.faces) for p in parts) == len(mesh.faces)
    groups = {name: trimesh.util.concatenate(values) for name, values in buckets.items()}
    return dict(mesh=mesh, groups=groups, assignments=assignments, mounts=lower,
                crank_ends=upper, pivots=centers)


@lru_cache(maxsize=1)
def asset_directory():
    files = [URDF] + [URDF.parent.parent / f"meshes/link_{s}_shank.obj" for s in ("right", "left")]
    digest = hashlib.sha256(b"cad-zero-split-v1" + b"".join(p.read_bytes() for p in files)).hexdigest()[:16]
    return HERE / "assets" / f"cad_zero_{digest}"


def export_visuals(geometry):
    """Derive immutable, content-addressed OBJ parts for this zero-pose geometry."""
    out = asset_directory()
    out.mkdir(parents=True, exist_ok=True)
    cad = load_cad(geometry.side)
    frames = {"shank": (np.zeros(3), np.eye(3))}
    if __package__:
        from .real_ankle import point_z
    else:
        from real_ankle import point_z
    for i, branch in enumerate(("long", "short")):
        frames[f"{branch}_crank"] = geometry.pivots[i], geometry.crank_rotation(i, 0.)
        frames[f"{branch}_rod"] = geometry.crank_end(i, 0.), point_z(geometry.mounts[i] - geometry.crank_end(i, 0.))
        frames[f"{branch}_foot"] = geometry.mounts[i], np.eye(3)
    paths = {}
    for name, mesh in cad["groups"].items():
        path = out / f"{geometry.side}_{name}.obj"
        paths[name] = path
        if not path.exists():
            local = mesh.copy()
            p, rotation = frames[name]
            local.vertices = (local.vertices - p) @ rotation
            local.export(path, file_type="obj", include_normals=False, include_texture=False)
    report = out / f"{geometry.side}_partition.json"
    if not report.exists():
        report.write_text(json.dumps(dict(source=str(URDF), side=geometry.side,
                         source_faces=len(cad["mesh"].faces),
                         group_faces={k:len(v.faces) for k,v in cad["groups"].items()},
                         assignments=cad["assignments"],
                         mounts_m=geometry.mounts.tolist(), pivots_m=geometry.pivots.tolist(),
                         crank_ends_m=cad["crank_ends"].tolist()), indent=2) + "\n")
    return paths
