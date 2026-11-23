"""Sphere fits for disk-shaped wheel geoms."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from assetx import ApproximateWithSphere, MujocoAsset, fit_sphere


def test_fit_sphere_uses_rolling_radius_not_bounding_sphere() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ring = np.stack((0.1 * np.cos(angles), 0.1 * np.sin(angles), np.zeros(64)), axis=1)
    disk = np.vstack((ring + (0.0, 0.0, 0.02), ring + (0.0, 0.0, -0.02)))
    disk += np.array([1.0, 2.0, 3.0])

    fit = fit_sphere(disk)

    assert fit.pos == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)
    assert fit.radius == pytest.approx(0.1, abs=1e-6)
    assert fit.radius < np.hypot(0.1, 0.02)


def test_fit_sphere_keeps_a_round_cloud() -> None:
    points = 0.2 * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )

    fit = fit_sphere(points)

    assert fit.radius == pytest.approx(0.2, abs=1e-6)


def test_approximate_wheel_cylinder_with_sphere() -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="fl_wheel")
    body.add_geom(
        name="fl_wheel_collision0",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.09, 0.027, 0.0),
        pos=(0.0, 0.05, 0.0),
        quat=(0.707107, 0.707107, 0.0, 0.0),
    )
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = ApproximateWithSphere(
        [["fl_wheel_collision0"]],
        names=["fl_wheel_collision0"],
        replace=True,
    ).transform(asset)

    sphere = out.spec.geom("fl_wheel_collision0")
    assert sphere is not None
    assert sphere.type == mujoco.mjtGeom.mjGEOM_SPHERE
    assert float(sphere.size[0]) == pytest.approx(0.09, abs=1e-3)
    assert tuple(float(x) for x in sphere.pos) == pytest.approx((0.0, 0.05, 0.0), abs=1e-3)


def test_approximate_with_sphere_requires_a_group() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ApproximateWithSphere([])
