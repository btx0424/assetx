from __future__ import annotations

import os
from itertools import product

import mujoco
import numpy as np

from assetx.core.asset import MujocoAsset


def compile_asset_spec(asset: MujocoAsset, spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile ``spec`` with meshes resolved relative to the asset model dir."""
    prev_cwd = os.getcwd()
    model_dir = str(asset.model_dir)
    try:
        os.chdir(model_dir)
        return spec.compile()
    finally:
        os.chdir(prev_cwd)


def mesh_vertices_local(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=float)
    scale = np.asarray(model.mesh_scale[mesh_id], dtype=float)
    return verts * scale


def sample_sphere(radius: float, n: int = 64) -> np.ndarray:
    """Fibonacci-sphere samples on a sphere of the given radius (plus origin)."""
    if radius <= 0.0:
        return np.zeros((1, 3), dtype=float)
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    pts = np.stack((np.cos(theta) * r, y, np.sin(theta) * r), axis=1) * radius
    return np.vstack((pts, np.zeros((1, 3))))


def geom_points_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    """Sample geom surface/vertex points in world frame."""
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)

    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError(f"geom {geom_id} is MESH but has no mesh id")
        local = mesh_vertices_local(model, mesh_id)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        sx, sy, sz = size
        local = np.asarray(list(product((-sx, sx), (-sy, sy), (-sz, sz))), dtype=float)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local = sample_sphere(float(size[0]))
    elif gtype in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        radius = float(size[0])
        half = float(size[1])
        rings = np.linspace(-half, half, num=7)
        angles = np.linspace(0.0, 2.0 * np.pi, num=16, endpoint=False)
        circle = np.stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.zeros(len(angles))),
            axis=1,
        )
        local_parts = [circle + np.array([0.0, 0.0, z]) for z in rings]
        if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            local_parts.append(sample_sphere(radius) + np.array([0.0, 0.0, half]))
            local_parts.append(sample_sphere(radius) + np.array([0.0, 0.0, -half]))
        local = np.vstack(local_parts)
    else:
        raise ValueError(f"Unsupported geom type for approximation: {gtype}")

    return (xmat @ local.T).T + xpos


def points_in_body_frame(
    points_world: np.ndarray,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
) -> np.ndarray:
    return (points_world - body_xpos) @ body_xmat
