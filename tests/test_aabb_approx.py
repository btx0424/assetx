"""Tests for AABB fitting and ApproximateWithAABB."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from assetx import ApproximateWithAABB, MujocoAsset, fit_aabb


def test_fit_aabb_axis_aligned_cloud() -> None:
    points = np.array(
        [
            [-1.0, -2.0, -3.0],
            [1.0, -2.0, -3.0],
            [-1.0, 2.0, -3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    fit = fit_aabb(points)
    assert fit.pos == (0.0, 0.0, 0.0)
    assert fit.half_sizes == (1.0, 2.0, 3.0)


def test_approximate_with_aabb_replaces_named_geom() -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base")
    geom = body.add_geom(
        name="base_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.05, 0.02),
        pos=(0.2, -0.1, 0.3),
    )
    geom.rgba = (0.1, 0.2, 0.3, 1.0)
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = ApproximateWithAABB(
        [["base_collision"]],
        names=["base_box"],
        replace=True,
    ).transform(asset)

    names = [g.name for g in out.spec.geoms]
    assert names == ["base_box"]
    box = out.spec.geom("base_box")
    assert box is not None
    assert int(box.type) == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert tuple(float(x) for x in box.quat) == (1.0, 0.0, 0.0, 0.0)
    assert np.allclose(box.pos, (0.2, -0.1, 0.3), atol=1e-9)
    assert np.allclose(box.size, (0.1, 0.05, 0.02), atol=1e-9)
