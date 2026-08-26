"""Tests for AddDummyBody."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

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


def test_add_dummy_body_align_to_world() -> None:
    spec = mujoco.MjSpec()
    # Parent rotated 90 deg about Y: local +X -> world -Z.
    parent = spec.worldbody.add_body(
        name="gripper_base",
        quat=(0.70710678, 0.0, 0.70710678, 0.0),
    )
    parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = AddDummyBody(
        parent_path="gripper_base",
        name="grasp_point",
        pos=(0.05, 0.0, 0.0),  # world +X
        align_to="world",
        marker=False,
    ).transform(asset)

    model = out.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_point")
    xmat = np.asarray(data.xmat[gid], dtype=float).reshape(3, 3)
    assert np.allclose(xmat, np.eye(3), atol=1e-5)
    assert np.allclose(
        data.xpos[gid] - data.xpos[pid],
        (0.05, 0.0, 0.0),
        atol=1e-5,
    )


def test_add_dummy_body_align_to_body() -> None:
    spec = mujoco.MjSpec()
    base = spec.worldbody.add_body(name="base_link")
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    # 180 deg about X relative to base.
    parent = base.add_body(
        name="ee_link",
        quat=(0.0, 1.0, 0.0, 0.0),
    )
    parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.02, 0.02, 0.02))
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = AddDummyBody(
        parent_path="ee_link",
        name="grasp_point",
        pos=(0.0, 0.0, -0.05),  # base-frame offset
        align_to="base_link",
        marker=False,
    ).transform(asset)

    model = out.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    gp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_point")
    R_base = np.asarray(data.xmat[base_id], dtype=float).reshape(3, 3)
    R_gp = np.asarray(data.xmat[gp_id], dtype=float).reshape(3, 3)
    assert np.allclose(R_base.T @ R_gp, np.eye(3), atol=1e-5)
    assert np.allclose(
        data.xpos[gp_id] - data.xpos[pid],
        R_base @ np.array([0.0, 0.0, -0.05]),
        atol=1e-5,
    )


def test_add_dummy_body_align_to_none_keeps_parent_pose() -> None:
    spec = mujoco.MjSpec()
    parent = spec.worldbody.add_body(
        name="gripper_base",
        quat=(0.70710678, 0.0, 0.70710678, 0.0),
    )
    parent.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = AddDummyBody(
        parent_path="gripper_base",
        name="grasp_point",
        pos=(0.05, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        align_to=None,
        marker=False,
    ).transform(asset)

    body = out.spec.body("grasp_point")
    assert tuple(float(x) for x in body.pos) == (0.05, 0.0, 0.0)
    assert np.allclose(body.quat, (1.0, 0.0, 0.0, 0.0), atol=1e-6)
