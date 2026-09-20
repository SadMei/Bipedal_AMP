"""Generate physical BSRL closed ankle loops after URDF import, before cloning.

There is no separately maintained USD. Four motor hinges and four rod spherical
joints belong to the articulation tree; four distal spherical joints close the
loops and are explicitly excluded from that tree. Auxiliary collision is off.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import tempfile
import torch
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
from isaaclab.sim.utils import clone, get_current_stage
from legged_lab.assets.bsrl_closed_ankle_geometry import CADAnkleGeometry, point_z


def _quat(matrix):
    xyzw = Rotation.from_matrix(np.asarray(matrix)).as_quat()
    return Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*map(float, xyzw[:3])))


def _vec(v):
    return Gf.Vec3f(*map(float, v))


def _body(stage, path, matrix, length, radius, axis, mass):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(matrix))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    inertia = UsdPhysics.MassAPI.Apply(prim)
    inertia.CreateMassAttr(mass)
    center = np.zeros(3)
    center[axis] = length/2
    diag = np.full(3, mass*(length*length/12 + radius*radius/4))
    diag[axis] = mass*radius*radius/2
    inertia.CreateCenterOfMassAttr(_vec(center))
    inertia.CreateDiagonalInertiaAttr(_vec(diag))
    inertia.CreatePrincipalAxesAttr(Gf.Quatf(1))
    capsule = UsdGeom.Capsule.Define(stage, path + "/link_visual")
    capsule.CreateAxisAttr("XYZ"[axis])
    capsule.CreateHeightAttr(length)
    capsule.CreateRadiusAttr(radius)
    capsule.CreateDisplayColorAttr([Gf.Vec3f(.2, .55, .85)])
    UsdGeom.Xformable(capsule).AddTranslateOp().Set(Gf.Vec3d(*center))
    # No CollisionAPI: these are idealized rods, not new contact surfaces.
    return prim


def _joint(stage, path, schema, body0, body1, pos0, pos1, rot0=None, excluded=False):
    joint = schema.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(_vec(pos0))
    joint.CreateLocalPos1Attr(_vec(pos1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1) if rot0 is None else _quat(rot0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(excluded)
    joint.CreateCollisionEnabledAttr(False)
    return joint


def _transform(rotation, translation):
    # USD/Gf uses row-vector transforms, NumPy geometry uses column vectors.
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).T
    result[3, :3] = np.asarray(translation)
    return result


def _cad_visual_cache():
    """Generate instanceable CAD visuals; copied OBJ parts are the only inputs."""
    directory = Path(__file__).resolve().parents[1] / "data/Robots/BSRL_closed_ankle/cad_zero_c18b94eb33e281e6"
    out = Path(tempfile.mkdtemp(prefix="bsrl-closed-ankle-visuals-")) / "cad.usdc"
    stage = Usd.Stage.CreateNew(str(out))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/CAD").GetPrim())
    for side in ("left", "right"):
        for part in ("shank", "long_crank", "short_crank", "long_rod", "short_rod", "long_foot", "short_foot"):
            path = f"/CAD/{side}_{part}"
            UsdGeom.Xform.Define(stage, path)
            mesh = trimesh.load(directory / f"{side}_{part}.obj", force="mesh", process=False)
            usd_mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
            usd_mesh.CreatePointsAttr(mesh.vertices.astype(np.float32))
            usd_mesh.CreateFaceVertexCountsAttr(np.full(len(mesh.faces), 3, dtype=np.int32))
            usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.astype(np.int32).reshape(-1))
            usd_mesh.CreateSubdivisionSchemeAttr("none")
            usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(.7, .7, .72)])
    stage.GetRootLayer().Save()
    return str(out)


def _cad_instance(stage, path, cache_file, name, translation=(0, 0, 0)):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    prim.GetReferences().AddReference(cache_file, f"/CAD/{name}")
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    prim.SetInstanceable(True)


@clone
def spawn_bsrl_closed_ankles(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_from_urdf.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    cache = UsdGeom.XformCache()
    root_inverse = np.asarray(cache.GetLocalToWorldTransform(prim).GetInverse())
    bodies = {p.GetName(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    geometry = CADAnkleGeometry(dtype=torch.float64)
    cad_cache = _cad_visual_cache()
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    neutral_crank = geometry.crank_rotation(zeros).numpy()
    neutral_ends = geometry.crank_end(zeros).numpy()
    neutral_rod = point_z(geometry.mounts - geometry.crank_end(zeros)).numpy()
    UsdGeom.Scope.Define(stage, prim_path + "/closed_ankle_joints")
    for side_idx, side in enumerate(("left", "right")):
        shank = bodies[f"link_{side}_shank"]
        foot = bodies[f"link_{side}_ankle_roll"]
        # Read the pitch pivot from the imported URDF, never assume a new zero.
        pitch = bodies[f"link_{side}_ankle_pitch"]
        shank_world = cache.GetLocalToWorldTransform(shank)
        pitch_to_shank = cache.GetLocalToWorldTransform(pitch) * shank_world.GetInverse()
        origin = np.asarray(pitch_to_shank.ExtractTranslation())
        shank_to_root = np.asarray(shank_world) @ root_inverse
        # The original shank mesh contains stationary rods. Replace only its
        # visual with the CAD partition, keeping every original collision/inertia.
        visual = shank.GetChild("visuals")
        if not visual.IsValid():
            raise RuntimeError(f"Cannot locate imported shank visuals at {shank.GetPath()}")
        UsdGeom.Imageable(visual).MakeInvisible()
        _cad_instance(stage, str(shank.GetPath()) + "/closed_shank_visual", cad_cache, f"{side}_shank", origin)
        for i, branch in enumerate(("long", "short")):
            name = f"{side}_{branch}"
            crank_path = f"{prim_path}/crank_{name}"
            rod_path = f"{prim_path}/rod_{name}"
            pivot = geometry.pivots[side_idx, i].numpy()
            lc = float(geometry.crank_lengths[side_idx, i])
            lr = float(geometry.rod_lengths[side_idx, i])
            _body(stage, crank_path, _transform(neutral_crank[side_idx, i], origin+pivot) @ shank_to_root,
                  lc, .004, 0, .001)
            _body(stage, rod_path, _transform(neutral_rod[side_idx, i], origin+neutral_ends[side_idx, i]) @ shank_to_root,
                  lr, .003, 2, .001)
            for body_path, part in ((crank_path, "crank"), (rod_path, "rod")):
                UsdGeom.Imageable(stage.GetPrimAtPath(body_path + "/link_visual")).MakeInvisible()
                _cad_instance(stage, body_path + "/cad_visual", cad_cache, f"{side}_{branch}_{part}")
            _cad_instance(stage, str(foot.GetPath()) + f"/cad_{branch}_mount", cad_cache,
                          f"{side}_{branch}_foot", geometry.mounts[side_idx, i].numpy())
            motor = _joint(stage, f"{prim_path}/closed_ankle_joints/motor_{name}", UsdPhysics.RevoluteJoint,
                           str(shank.GetPath()), crank_path, origin+pivot, [0, 0, 0], neutral_crank[side_idx, i])
            motor.CreateAxisAttr("Y")
            _joint(stage, f"{prim_path}/closed_ankle_joints/rod_ball_{name}", UsdPhysics.SphericalJoint,
                   crank_path, rod_path, [lc, 0, 0], [0, 0, 0], geometry.ball_zero_rot[side_idx, i].numpy())
            _joint(stage, f"{prim_path}/closed_ankle_joints/closure_{name}", UsdPhysics.SphericalJoint,
                   rod_path, str(foot.GetPath()), [0, 0, lr], geometry.mounts[side_idx, i].numpy(), excluded=True)
    print("[ClosedAnkle] Added 4 motor hinges, 4 rod ball joints, 4 EXTERNAL spherical closures; auxiliary mass +8 g.", flush=True)
    return prim
