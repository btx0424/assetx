"""ReplaceCylinderWithCapsule include/exclude filters."""

from __future__ import annotations

from pathlib import Path

import mujoco

from assetx import MujocoAsset, ReplaceCylinderWithCapsule


def _asset() -> MujocoAsset:
    spec = mujoco.MjSpec()
    base = spec.worldbody.add_body(name="base")
    base.add_geom(
        name="rail",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.07, 0.3, 0.0),
    )
    base.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
    )
    wheel = base.add_body(name="fl_wheel")
    wheel.add_geom(
        name="fl_wheel_collision0",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.09, 0.027, 0.0),
    )
    link = base.add_body(name="fl_hipx")
    link.add_geom(
        name="",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.048, 0.07, 0.0),
    )
    return MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))


def _types(asset: MujocoAsset) -> dict[str, int]:
    return {geom.name: int(geom.type) for geom in asset.spec.geoms}


def test_default_converts_every_cylinder() -> None:
    out = ReplaceCylinderWithCapsule().transform(_asset())
    types = _types(out)
    assert types["rail"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types["fl_wheel_collision0"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types[""] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types["box"] == int(mujoco.mjtGeom.mjGEOM_BOX)


def test_exclude_matches_body_or_geom_name() -> None:
    out = ReplaceCylinderWithCapsule(exclude=[".*_wheel"]).transform(_asset())
    types = _types(out)
    assert types["fl_wheel_collision0"] == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert types["rail"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types[""] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)


def test_include_limits_conversion() -> None:
    out = ReplaceCylinderWithCapsule(include=["rail"]).transform(_asset())
    types = _types(out)
    assert types["rail"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types["fl_wheel_collision0"] == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert types[""] == int(mujoco.mjtGeom.mjGEOM_CYLINDER)


def test_exclude_wins_over_include() -> None:
    out = ReplaceCylinderWithCapsule(
        include=[".*"],
        exclude=["fl_wheel"],
    ).transform(_asset())
    types = _types(out)
    assert types["fl_wheel_collision0"] == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert types["rail"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
