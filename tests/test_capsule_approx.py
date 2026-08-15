"""Tests for capsule PCA fitting."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from assetx import fit_capsule_pca


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
