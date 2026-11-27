"""Rounded-wheel fits recorded beside the MJCF, not as a fake primitive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mujoco

from assetx import FitRoundedWheels, MujocoAsset, fit_rounded_wheel


def test_fit_rounded_wheel_offsets_a_disk() -> None:
    axis = np.array([0.0, 1.0, 0.0])
    center = np.array([0.0, 0.05, 0.0])
    radius = 0.08
    half_width = 0.02
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    rim = np.stack((radius * np.cos(angles), np.zeros(16), radius * np.sin(angles)), axis=1)
    points = np.vstack(
        (
            center + rim,
            center + rim + half_width * axis,
            center + rim - half_width * axis,
            center + np.array([half_width, 0.0, 0.0]),
        )
    )
    fitted_center, fitted_axis, fitted_radius, fitted_half_width = fit_rounded_wheel(
        points, axis
    )
    assert fitted_axis == (0.0, 1.0, 0.0)
    assert fitted_center == pytest.approx((0.0, 0.05, 0.0), abs=1e-6)
    assert fitted_radius == pytest.approx(radius, abs=1e-6)
    assert fitted_half_width == pytest.approx(half_width, abs=1e-6)


def test_cylinder_fit_is_exact_and_saved(tmp_path: Path) -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wheel")
    geom = body.add_geom(
        name="wheel_collision0",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.09, 0.027, 0.0),
        quat=(0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    )
    geom.contype = 1
    geom.conaffinity = 1
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    out = FitRoundedWheels([["wheel_collision0"]]).transform(asset)

    assert out.spec.geom("wheel_collision0").type == mujoco.mjtGeom.mjGEOM_CYLINDER
    (fit,) = out.metadata["rounded_wheels"]
    assert fit.body == "wheel"
    assert fit.geoms == ("wheel_collision0",)
    assert fit.radius == pytest.approx(0.09)
    assert fit.half_width == pytest.approx(0.027)
    assert fit.center == pytest.approx((0.0, 0.0, 0.0))
    assert fit.axis == pytest.approx((0.0, 1.0, 0.0))

    saved = out.save(tmp_path / "wheel", save_urdf=False)
    payload = json.loads((tmp_path / "wheel" / "rounded_wheels.json").read_text())
    assert payload["wheels"][0]["radius"] == pytest.approx(0.09)
    assert saved.metadata["rounded_wheels"][0].half_width == pytest.approx(0.027)
