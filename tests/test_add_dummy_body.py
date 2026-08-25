"""Tests for AddDummyBody."""

from __future__ import annotations

from pathlib import Path

import mujoco

from assetx import AddDummyBody, MujocoAsset


def test_add_dummy_body_with_visual_marker() -> None:
    spec = mujoco.MjSpec()
    parent = spec.worldbody.add_body(name="gripper_base")
    parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = AddDummyBody(
        parent_path="gripper_base",
        name="grasp_point",
        pos=(0.05, 0.0, 0.0),
        marker_size=0.01,
        rgba=(1.0, 0.0, 0.0, 0.6),
    ).transform(asset)

    body = out.spec.body("grasp_point")
    assert body is not None
    assert body.parent.name == "gripper_base"
    assert tuple(float(x) for x in body.pos) == (0.05, 0.0, 0.0)
    assert float(body.mass) == 0.0
    assert len(body.joints) == 0
    assert len(body.geoms) == 1
    geom = body.geoms[0]
    assert geom.name == "grasp_point_visual"
    assert int(geom.contype) == 0
    assert int(geom.conaffinity) == 0


def test_add_dummy_body_rejects_duplicate_name() -> None:
    spec = mujoco.MjSpec()
    parent = spec.worldbody.add_body(name="gripper_base")
    parent.add_body(name="grasp_point")
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    try:
        AddDummyBody(parent_path="gripper_base", name="grasp_point").transform(asset)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate body name")
