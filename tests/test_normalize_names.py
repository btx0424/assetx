"""Tests for assigning names to unnamed MJCF geoms."""

from pathlib import Path

import mujoco

from assetx import MujocoAsset, NormalizeGeomNames


def test_normalize_names_assigns_role_names_without_collisions() -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base")
    body.add_geom(
        name="base_collision0", type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, 1)
    )
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, 1))
    visual0 = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, 1))
    visual0.contype = 0
    visual0.conaffinity = 0
    visual1 = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1, 1, 1))
    visual1.contype = 0
    visual1.conaffinity = 0
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    normalized = NormalizeGeomNames().transform(asset)

    assert [geom.name for geom in normalized.spec.geoms] == [
        "base_collision0",
        "base_collision1",
        "base_visual0",
        "base_visual1",
    ]
    assert [geom.name for geom in asset.spec.geoms] == [
        "base_collision0",
        "",
        "",
        "",
    ]
