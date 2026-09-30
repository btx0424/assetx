"""Tests for ``assetx.conversion.usd.postprocess.flatten_articulation``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from assetx.conversion.usd.postprocess import (
    flatten_articulation,
    group_body_geometry,
    group_joint_prims,
)


def _body(stage: Usd.Stage, path: str, pos: tuple[float, float, float], deg: float = 0.0):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), deg).GetQuat()))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    return xf.GetPrim()


def _make_nested(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/robot")
    robot.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.Scope.Define(stage, "/robot/Geometry")
    base = _body(stage, "/robot/Geometry/base", (0.1, 0.0, 0.0))
    UsdPhysics.ArticulationRootAPI.Apply(base)
    _body(stage, "/robot/Geometry/base/thigh", (0.0, 0.2, 0.0), deg=90.0)
    _body(stage, "/robot/Geometry/base/thigh/shin", (0.3, 0.0, 0.0))
    cube = UsdGeom.Cube.Define(stage, "/robot/Geometry/base/thigh/shin/visual")
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.1))
    foot = UsdGeom.Sphere.Define(stage, "/robot/Geometry/base/thigh/shin/foot")
    foot.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.3))
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim())
    base.CreateRelationship("test:foot").SetTargets(["/robot/Geometry/base/thigh/shin/foot"])

    mat = UsdShade.Material.Define(stage, "/robot/Materials/red")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    knee = UsdPhysics.RevoluteJoint.Define(stage, "/robot/Geometry/base/thigh/shin/knee")
    knee.CreateBody0Rel().SetTargets(["/robot/Geometry/base/thigh"])
    knee.CreateBody1Rel().SetTargets(["/robot/Geometry/base/thigh/shin"])
    base.CreateRelationship("test:knee").SetTargets([knee.GetPath()])
    pairs = UsdPhysics.FilteredPairsAPI.Apply(base)
    pairs.CreateFilteredPairsRel().SetTargets(["/robot/Geometry/base/thigh/shin"])
    stage.GetRootLayer().Save()


def _world(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))


def test_flatten_preserves_poses_and_targets(tmp_path: Path) -> None:
    usd = tmp_path / "robot.usda"
    _make_nested(usd)
    before = Usd.Stage.Open(str(usd))
    old = {
        n: _world(before, p)
        for n, p in {
            "base": "/robot/Geometry/base",
            "thigh": "/robot/Geometry/base/thigh",
            "shin": "/robot/Geometry/base/thigh/shin",
            "visual": "/robot/Geometry/base/thigh/shin/visual",
            "foot": "/robot/Geometry/base/thigh/shin/foot",
        }.items()
    }
    del before

    moves = flatten_articulation(usd)
    assert moves["/robot/Geometry/base/thigh/shin"] == "/robot/shin"

    stage = Usd.Stage.Open(str(usd))
    root = stage.GetPrimAtPath("/robot")
    assert sorted(c.GetName() for c in root.GetChildren()) == [
        "Materials",
        "base",
        "joints",
        "shin",
        "thigh",
    ]
    new_paths = {
        "base": "/robot/base",
        "thigh": "/robot/thigh",
        "shin": "/robot/shin",
        "visual": "/robot/shin/visuals/visual",
        "foot": "/robot/shin/collisions/foot",
    }
    for name, path in new_paths.items():
        assert Gf.IsClose(_world(stage, path), old[name], 1e-6), name

    shin_children = sorted(c.GetName() for c in stage.GetPrimAtPath("/robot/shin").GetChildren())
    assert shin_children == ["collisions", "visuals"]
    knee = UsdPhysics.Joint(stage.GetPrimAtPath("/robot/joints/knee"))
    assert knee.GetBody0Rel().GetTargets() == [Sdf.Path("/robot/thigh")]
    assert knee.GetBody1Rel().GetTargets() == [Sdf.Path("/robot/shin")]
    knee_rel = stage.GetPrimAtPath("/robot/base").GetRelationship("test:knee")
    assert knee_rel.GetTargets() == [Sdf.Path("/robot/joints/knee")]
    pairs = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath("/robot/base"))
    assert pairs.GetFilteredPairsRel().GetTargets() == [Sdf.Path("/robot/shin")]
    foot_rel = stage.GetPrimAtPath("/robot/base").GetRelationship("test:foot")
    assert foot_rel.GetTargets() == [Sdf.Path("/robot/shin/collisions/foot")]
    bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/robot/shin/visuals/visual"))
    assert bound.ComputeBoundMaterial()[0].GetPath() == Sdf.Path("/robot/Materials/red")
    assert stage.GetPrimAtPath("/robot").HasAPI(UsdPhysics.ArticulationRootAPI) is False
    assert stage.GetPrimAtPath("/robot/base").HasAPI(UsdPhysics.ArticulationRootAPI)


def test_flatten_without_grouping_and_grouping_is_idempotent(tmp_path: Path) -> None:
    usd = tmp_path / "robot.usda"
    _make_nested(usd)
    flatten_articulation(usd, group_geometry=False, group_joints=False)
    stage = Usd.Stage.Open(str(usd))
    assert stage.GetPrimAtPath("/robot/shin/visual")
    assert stage.GetPrimAtPath("/robot/shin/foot")
    assert stage.GetPrimAtPath("/robot/shin/knee")
    del stage

    assert group_body_geometry(usd)
    assert group_body_geometry(usd) == {}
    assert group_joint_prims(usd) == {"/robot/shin/knee": "/robot/joints/knee"}
    assert group_joint_prims(usd) == {}
    stage = Usd.Stage.Open(str(usd))
    assert stage.GetPrimAtPath("/robot/shin/visuals/visual")
    assert stage.GetPrimAtPath("/robot/shin/collisions/foot")
    assert stage.GetPrimAtPath("/robot/joints/knee")


def test_group_joints_qualifies_duplicate_names(tmp_path: Path) -> None:
    usd = tmp_path / "robot.usda"
    _make_nested(usd)
    stage = Usd.Stage.Open(str(usd))
    for body in ("thigh", "thigh/shin"):
        UsdPhysics.FixedJoint.Define(stage, f"/robot/Geometry/base/{body}/PhysicsFixedJoint")
    stage.GetRootLayer().Save()
    del stage

    flatten_articulation(usd)
    stage = Usd.Stage.Open(str(usd))
    names = sorted(c.GetName() for c in stage.GetPrimAtPath("/robot/joints").GetChildren())
    assert names == ["knee", "shin_PhysicsFixedJoint", "thigh_PhysicsFixedJoint"]


def test_flatten_rejects_name_clash(tmp_path: Path) -> None:
    usd = tmp_path / "robot.usda"
    _make_nested(usd)
    stage = Usd.Stage.Open(str(usd))
    UsdGeom.Xform.Define(stage, "/robot/shin")
    stage.GetRootLayer().Save()
    del stage
    with pytest.raises(ValueError, match="already exists"):
        flatten_articulation(usd)
