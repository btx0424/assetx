"""Tests for capsule PCA fitting and chained approximation."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from assetx import (
    ApproximateWithCapsule,
    CapsuleFit,
    MujocoAsset,
    fit_capsule_chain_pca,
    fit_capsule_pca,
    split_capsule_fit,
)


def _point_capsule_distance(
    points: np.ndarray,
    pos: np.ndarray,
    axis: np.ndarray,
    radius: float,
    half_height: float,
) -> np.ndarray:
    """Signed distance to capsule surface (<= 0 means inside / on surface)."""
    rel = points - pos
    t = rel @ axis
    t_clamped = np.clip(t, -half_height, half_height)
    closest = pos + np.outer(t_clamped, axis)
    return np.linalg.norm(points - closest, axis=1) - radius


def _capsule_centerline_from_fit(fit: CapsuleFit) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(fit.pos, dtype=float)
    axis = sRot.from_quat(fit.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    return pos - fit.half_height * axis, pos + fit.half_height * axis


def _capsule_tips_from_fit(fit: CapsuleFit) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(fit.pos, dtype=float)
    axis = sRot.from_quat(fit.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    extent = fit.half_height + fit.radius
    return pos - extent * axis, pos + extent * axis


def _points_covered_by_chain(points: np.ndarray, segments: list[CapsuleFit]) -> bool:
    covered = np.zeros(points.shape[0], dtype=bool)
    for fit in segments:
        axis = sRot.from_quat(fit.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
        dist = _point_capsule_distance(
            points,
            np.asarray(fit.pos),
            axis,
            fit.radius,
            fit.half_height,
        )
        covered |= dist <= 1e-6
    return bool(np.all(covered))


def test_fit_capsule_pca_encapsulates_cylinder_cloud() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(-0.05, 0.05, size=800)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=800)
    r = 0.02 * np.sqrt(rng.uniform(0.0, 1.0, size=800))
    points = np.stack((r * np.cos(angles), r * np.sin(angles), z), axis=1)

    fit = fit_capsule_pca(points)
    assert fit.radius > 0.0
    assert fit.half_height >= 0.0

    axis = sRot.from_quat(fit.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    dist = _point_capsule_distance(
        points,
        np.asarray(fit.pos),
        axis,
        fit.radius,
        fit.half_height,
    )
    assert float(dist.max()) <= 1e-9


def test_fit_capsule_pca_principal_axis_on_elongated_cloud() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(400, 3)) * np.array([0.01, 0.01, 0.08])
    fit = fit_capsule_pca(points)
    axis = sRot.from_quat(fit.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    assert abs(abs(float(axis[2])) - 1.0) < 0.05
    assert fit.half_height > fit.radius


def test_fit_capsule_pca_single_point() -> None:
    fit = fit_capsule_pca(np.array([[1.0, 2.0, 3.0]]))
    assert fit.pos == (1.0, 2.0, 3.0)
    assert fit.radius == 0.0
    assert fit.half_height == 0.0


def test_split_capsule_fit_chains_endpoints() -> None:
    fit = CapsuleFit(
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        radius=0.02,
        half_height=0.10,
    )
    segments = split_capsule_fit(fit, 3)
    assert len(segments) == 3

    orig_start, orig_end = _capsule_tips_from_fit(fit)

    prev_end = None
    for seg in segments:
        start, end = _capsule_tips_from_fit(seg)
        if prev_end is not None:
            assert np.allclose(prev_end, start, atol=1e-9)
        prev_end = end
    assert np.allclose(_capsule_tips_from_fit(segments[0])[0], orig_start, atol=1e-9)
    assert np.allclose(prev_end, orig_end, atol=1e-9)


def test_fit_capsule_chain_pca_bent_geometry() -> None:
    rng = np.random.default_rng(2)
    leg_a = rng.normal(size=(200, 3)) * np.array([0.01, 0.01, 0.08]) + np.array(
        [0.0, 0.0, -0.10]
    )
    leg_b = rng.normal(size=(200, 3)) * np.array([0.08, 0.01, 0.01]) + np.array(
        [0.10, 0.0, 0.05]
    )
    points = np.vstack([leg_a, leg_b])

    segments = fit_capsule_chain_pca(points, 2, coverage_quantile=1.0)
    assert len(segments) == 2

    axis0 = sRot.from_quat(segments[0].quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    axis1 = sRot.from_quat(segments[1].quat, scalar_first=True).apply([0.0, 0.0, 1.0])
    split_axes = [
        sRot.from_quat(s.quat, scalar_first=True).apply([0.0, 0.0, 1.0])
        for s in split_capsule_fit(fit_capsule_pca(points), 2)
    ]
    chain_dot = abs(float(np.dot(axis0, axis1)))
    split_dot = abs(float(np.dot(split_axes[0], split_axes[1])))
    assert chain_dot < split_dot - 1e-3
    assert _points_covered_by_chain(points, segments)


def test_approximate_with_capsule_num_capsules() -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="link")
    body.add_geom(
        name="link_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.01, 0.01, 0.05),
        pos=(0.0, 0.0, 0.0),
    )
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = ApproximateWithCapsule(
        [["link_collision"]],
        names=["link_capsule"],
        num_capsules=2,
        replace=True,
    ).transform(asset)

    assert out.spec.geom("link_collision") is None
    assert out.spec.geom("link_capsule_0") is not None
    assert out.spec.geom("link_capsule_1") is not None
