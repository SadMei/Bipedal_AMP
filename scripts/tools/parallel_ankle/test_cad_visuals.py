"""Verify CAD splitting conserves source faces and their exact zero geometry."""
import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

if __package__:
    from .cad_assets import load_cad, export_visuals
    from .real_ankle import Geometry, RobotCore, point_z
else:
    from cad_assets import load_cad, export_visuals
    from real_ankle import Geometry, RobotCore, point_z


@pytest.mark.parametrize("side", ["right", "left"])
def test_visual_parts_are_lossless_and_attached_to_correct_frames(side):
    g = Geometry(side, RobotCore())
    cad = load_cad(side)
    paths = export_visuals(g)
    assert sum(len(m.faces) for m in cad["groups"].values()) == len(cad["mesh"].faces)
    assert len({a["component"] for a in cad["assignments"]}) == len(cad["assignments"])
    # No rear rods/crank plates remain baked into the static shank visual.
    assert cad["groups"]["shank"].bounds[0, 0] > -.055
    frames = {"shank": (np.zeros(3), np.eye(3))}
    for i, branch in enumerate(("long", "short")):
        frames[f"{branch}_crank"] = g.pivots[i], g.crank_rotation(i, 0.)
        frames[f"{branch}_rod"] = g.crank_end(i, 0.), point_z(g.mounts[i]-g.crank_end(i, 0.))
        frames[f"{branch}_foot"] = g.mounts[i], np.eye(3)
    for name, source in cad["groups"].items():
        exported = trimesh.load(paths[name], force="mesh", process=False)
        assert len(exported.faces) == len(source.faces)
        pos, rot = frames[name]
        reconstructed = exported.vertices @ rot.T + pos
        # OBJ numeric serialization is 8 decimals; no visual deformation/scaling.
        assert cKDTree(source.vertices).query(reconstructed)[0].max() < 1e-8
        assert cKDTree(reconstructed).query(source.vertices)[0].max() < 1e-8
