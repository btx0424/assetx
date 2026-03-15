"""Utilities for extracting and exporting mesh data from USD stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

try:
    from pxr import Usd, UsdGeom
except ImportError:
    raise ImportError("Use `pip install usd-core` to install the USD library.")


def _triangulate_face_indices(
    face_vertex_indices: np.ndarray,
    face_vertex_counts: np.ndarray,
) -> np.ndarray:
    """Convert polygon face indices to triangles using a fan from the first vertex.

    USD meshes can have quads or n-gons; trimesh requires triangles.
    """
    triangles = []
    offset = 0
    for count in face_vertex_counts:
        if count < 3:
            offset += count
            continue
        face = face_vertex_indices[offset : offset + count]
        for i in range(1, count - 1):
            triangles.append([face[0], face[i], face[i + 1]])
        offset += count
    return np.array(triangles, dtype=np.int32) if triangles else np.empty((0, 3), dtype=np.int32)


def _usd_mesh_to_trimesh(prim: Usd.Prim, time: Usd.TimeCode) -> trimesh.Trimesh | None:
    """Convert a single USD Mesh or Cube prim to trimesh.Trimesh. Returns None if not a mesh type."""
    type_name = prim.GetTypeName()
    if type_name == "Cube":
        cube = UsdGeom.Cube(prim)
        size_attr = cube.GetSizeAttr()
        if size_attr:
            size = np.asarray(size_attr.Get(time), dtype=np.float64)
            if size.ndim == 0:
                size = np.full(3, float(size))
        else:
            size = np.array([2.0, 2.0, 2.0])
        return trimesh.creation.box(extents=size)
    if type_name != "Mesh":
        return None

    mesh_schema = UsdGeom.Mesh(prim)
    points_attr = mesh_schema.GetPointsAttr()
    indices_attr = mesh_schema.GetFaceVertexIndicesAttr()
    counts_attr = mesh_schema.GetFaceVertexCountsAttr()

    if not points_attr or not indices_attr:
        return None

    points = np.asarray(points_attr.Get(time), dtype=np.float64).reshape(-1, 3)
    indices = np.asarray(indices_attr.Get(time), dtype=np.int32)
    counts = (
        np.asarray(counts_attr.Get(time), dtype=np.int32)
        if counts_attr
        else np.full(max(1, len(indices) // 3), 3)
    )

    if points.size == 0 or indices.size == 0:
        return None

    triangles = _triangulate_face_indices(indices, counts)
    if len(triangles) == 0:
        return None

    return trimesh.Trimesh(vertices=points, faces=triangles)


def extract_meshes(
    usd_path: str | Path,
    *,
    root_prim: Usd.Prim | None = None,
    time: Usd.TimeCode | None = None,
) -> dict[str, trimesh.Trimesh]:
    """Extract all Mesh and Cube prims from a USD stage as trimesh.Trimesh instances.

    Traverses the stage from the default prim (or the given root_prim) and converts
    each UsdGeom.Mesh and UsdGeom.Cube into a trimesh. Names are taken from the prim
    name; if duplicates occur (e.g. same name under different paths), later prims
    overwrite earlier ones. Quad and n-gon faces are triangulated with a vertex fan.

    Args:
        usd_path: Path to the USD file.
        root_prim: Optional prim to use as root. If None, the stage default prim is used.
        time: Time code for attribute evaluation. Defaults to Usd.TimeCode.Default().

    Returns:
        Dictionary mapping prim names to trimesh.Trimesh objects.

    Raises:
        FileNotFoundError: If usd_path does not exist.
        RuntimeError: If the stage cannot be opened or has no default prim and root_prim is None.
    """
    path = Path(usd_path)
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    time = time if time is not None else Usd.TimeCode.Default()
    root = root_prim if root_prim is not None else stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(
            "Stage has no default prim and no root_prim was given. "
            "Set a default prim in USD or pass root_prim."
        )

    meshes: dict[str, trimesh.Trimesh] = {}
    queue = list(root.GetChildren())
    while queue:
        prim = queue.pop(0)
        tmesh = _usd_mesh_to_trimesh(prim, time)
        if tmesh is not None:
            meshes[prim.GetName()] = tmesh
        queue.extend(prim.GetChildren())

    return meshes

