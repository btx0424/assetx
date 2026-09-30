"""Tests for GeomsToBody."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from assetx import GeomsToBody, MujocoAsset, NormalizeGeomNames


def test_geoms_to_body_moves_and_renames() -> None:
    spec = mujoco.MjSpec()
    calf = spec.worldbody.add_body(name="FL_calf")
    calf.add_geom(
        name="keep_me",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.01, 0.01, 0.01),
        pos=(0.0, 0.0, -0.1),
    )
    calf.add_geom(
        name="FL_calf_collision4",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.032, 0.0, 0.0),
        pos=(0.0, 0.0, -0.275),
        rgba=(0.7, 0.7, 0.7, 1.0),
    )
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = GeomsToBody(["FL_calf_collision4"], "FL_foot").transform(asset)

    foot = out.spec.body("FL_foot")
    assert foot is not None
    assert foot.parent.name == "FL_calf"
    assert np.allclose(foot.pos, (0.0, 0.0, -0.275), atol=1e-8)
    assert len(foot.geoms) == 1
    geom = foot.geoms[0]
    assert geom.name == "FL_foot_collision0"
    assert np.allclose(geom.pos, (0.0, 0.0, 0.0), atol=1e-8)
    assert float(geom.size[0]) == 0.032
    assert out.spec.geom("FL_calf_collision4") is None
    assert out.spec.body("FL_calf").geoms[0].name == "keep_me"


def test_geoms_to_body_preserves_world_pose() -> None:
    spec = mujoco.MjSpec()
    calf = spec.worldbody.add_body(name="FL_calf")
    calf.add_geom(
        name="foot_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.03, 0.0, 0.0),
        pos=(0.0, 0.0, -0.275),
    )
    before = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    model0 = before.spec.compile()
    data0 = mujoco.MjData(model0)
    mujoco.mj_forward(model0, data0)
    gid0 = mujoco.mj_name2id(model0, mujoco.mjtObj.mjOBJ_GEOM, "foot_sphere")
    xpos0 = np.array(data0.geom_xpos[gid0], dtype=float)

    out = GeomsToBody(["foot_sphere"], "FL_foot").transform(before)
    model1 = out.spec.compile()
    data1 = mujoco.MjData(model1)
    mujoco.mj_forward(model1, data1)
    gid1 = mujoco.mj_name2id(model1, mujoco.mjtObj.mjOBJ_GEOM, "FL_foot_collision0")
    assert np.allclose(data1.geom_xpos[gid1], xpos0, atol=1e-8)
    bid = mujoco.mj_name2id(model1, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
    assert np.allclose(data1.xpos[bid], xpos0, atol=1e-8)


def test_geoms_to_body_with_normalize_pipeline() -> None:
    spec = mujoco.MjSpec()
    calf = spec.worldbody.add_body(name="FL_calf")
    # Unnamed foot sphere — NormalizeGeomNames -> FL_calf_collision0
    calf.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.032, 0.0, 0.0),
        pos=(0.0, 0.0, -0.275),
    )
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    out = NormalizeGeomNames().transform(asset)
    out = GeomsToBody(["FL_calf_collision0"], "FL_foot").transform(out)
    assert out.spec.body("FL_foot") is not None
    assert out.spec.geom("FL_foot_collision0") is not None


def test_geoms_to_body_explicit_mass() -> None:
    spec = mujoco.MjSpec()
    calf = spec.worldbody.add_body(name="FL_calf")
    calf.add_geom(
        name="foot_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.032, 0.0, 0.0),
        pos=(0.0, 0.0, -0.275),
        density=1000.0,
    )
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = GeomsToBody(
        ["foot_sphere"],
        "FL_foot",
        mass=0.05,
        inertia=(1e-4, 1e-4, 1e-4),
    ).transform(asset)

    foot = out.spec.body("FL_foot")
    assert float(foot.mass) == 0.05
    assert np.allclose(foot.inertia, (1e-4, 1e-4, 1e-4))
    assert int(foot.explicitinertial) == 1
    assert float(foot.geoms[0].density) == 0.0

    model = out.spec.compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_foot")
    assert np.isclose(model.body_mass[bid], 0.05)
