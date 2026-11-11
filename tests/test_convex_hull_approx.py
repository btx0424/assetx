"""Tests for ApproximateWithConvexHull."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from assetx import ApproximateWithConvexHull, MujocoAsset


def _box_with_dent_spec(*, collision: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(name="dent")
    mesh.uservert = np.array(
        [
            0, 0, 0,
            1, 0, 0,
            1, 1, 0,
            0, 1, 0,
            0, 0, 1,
            1, 0, 1,
            1, 1, 1,
            0, 1, 1,
            0.5, 0.5, 0.5,
        ],
        dtype=np.float64,
    )
    mesh.userface = np.array(
        [
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ],
        dtype=np.int32,
    )
    body = spec.worldbody.add_body(name="link")
    geom = body.add_geom(
        name="link_src",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="dent",
    )
    if collision:
        geom.contype = 1
        geom.conaffinity = 1
    else:
        geom.contype = 0
        geom.conaffinity = 0
    return spec


def test_convex_hull_keeps_visual_and_adds_collision() -> None:
    spec = _box_with_dent_spec(collision=False)
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    out = ApproximateWithConvexHull(
        [["link_src"]],
        names=["link_collision0"],
        body_paths=["link"],
    ).transform(asset)

    names = [g.name for g in out.spec.geoms]
    assert "link_src" in names
    assert "link_collision0" in names
    hull = out.spec.geom("link_collision0")
    assert hull is not None
    assert int(hull.type) == int(mujoco.mjtGeom.mjGEOM_MESH)
    assert int(hull.contype) == 1
    assert int(hull.conaffinity) == 1
    assert float(hull.density) == 0.0
    mesh = out.spec.mesh(hull.meshname)
    assert mesh is not None
    verts = np.asarray(mesh.uservert, dtype=float).reshape(-1, 3)
    # Interior dent vertex is not a hull vertex.
    interior = np.array([0.5, 0.5, 0.5])
    assert not np.any(np.all(np.isclose(verts, interior, atol=1e-6), axis=1))


def test_convex_hull_replaces_collision_source() -> None:
    spec = _box_with_dent_spec(collision=True)
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    out = ApproximateWithConvexHull(
        [["link_src"]],
        names=["link_collision0"],
    ).transform(asset)

    names = [g.name for g in out.spec.geoms]
    assert "link_src" not in names
    assert names == ["link_collision0"]


def test_convex_hull_scale_shrinks_about_centroid() -> None:
    spec = _box_with_dent_spec(collision=False)
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))
    full = ApproximateWithConvexHull(
        [["link_src"]],
        names=["link_collision0"],
        scale=1.0,
    ).transform(asset)
    scaled = ApproximateWithConvexHull(
        [["link_src"]],
        names=["link_collision0"],
        scale=0.5,
    ).transform(asset)

    def _extent(asset: MujocoAsset) -> np.ndarray:
        geom = asset.spec.geom("link_collision0")
        assert geom is not None
        mesh = asset.spec.mesh(geom.meshname)
        assert mesh is not None
        verts = np.asarray(mesh.uservert, dtype=float).reshape(-1, 3)
        return verts.max(axis=0) - verts.min(axis=0)

    assert np.allclose(_extent(scaled), 0.5 * _extent(full), atol=1e-6)
