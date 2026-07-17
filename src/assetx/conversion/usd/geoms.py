"""Extract mesh and collision geometry from USD stages (Isaac Lab robots)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError as exc:
    raise ImportError(
        "USD support requires the usd extra: pip install -e '.[usd]'"
    ) from exc


GeomKind = Literal["mesh", "box", "sphere", "capsule", "cylinder"]


@dataclass
class BodyGeom:
    """A visual or collision geom belonging to a rigid body, in body-local frame."""

    name: str
    kind: GeomKind
    is_collision: bool
    is_visual: bool
    pos: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,) wxyz
    size: np.ndarray  # meaning depends on kind; unused for mesh
    mesh: trimesh.Trimesh | None = None
    prim_path: str = ""


def gf_matrix_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    """Convert a Gf.Matrix4d (row-vector) to a column-vector 4x4 numpy matrix."""
    return np.array(matrix, dtype=np.float64).T


def gf_quat_to_wxyz(quat: Gf.Quatd | Gf.Quatf) -> np.ndarray:
    """Convert a Gf quaternion to wxyz."""
    imag = quat.GetImaginary()
    return np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)


def relative_transform(
    prim: Usd.Prim,
    ancestor: Usd.Prim,
    time: Usd.TimeCode,
) -> np.ndarray:
    """Return the 4x4 transform of ``prim`` expressed in ``ancestor``'s frame."""
    cache = UsdGeom.XformCache(time)
    prim_world = gf_matrix_to_np(cache.GetLocalToWorldTransform(prim))
    ancestor_world = gf_matrix_to_np(cache.GetLocalToWorldTransform(ancestor))
    return np.linalg.inv(ancestor_world) @ prim_world


def decompose_pos_quat_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4x4 transform into translation, wxyz quaternion, and column scales."""
    from scipy.spatial.transform import Rotation as sRot

    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    rotation = linear / scale
    # Ensure a proper rotation (handle reflections).
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1.0
        rotation[:, 0] *= -1.0
    quat_xyzw = sRot.from_matrix(rotation).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return matrix[:3, 3].copy(), quat_wxyz, scale


def _triangulate_face_indices(
    face_vertex_indices: np.ndarray,
    face_vertex_counts: np.ndarray,
) -> np.ndarray:
    """Convert polygon face indices to triangles using a fan from the first vertex."""
    triangles = []
    offset = 0
    for count in face_vertex_counts:
        if count < 3:
            offset += int(count)
            continue
        face = face_vertex_indices[offset : offset + int(count)]
        for i in range(1, int(count) - 1):
            triangles.append([face[0], face[i], face[i + 1]])
        offset += int(count)
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def usd_mesh_to_trimesh(prim: Usd.Prim, time: Usd.TimeCode) -> trimesh.Trimesh | None:
    """Convert a UsdGeom.Mesh or Cube prim to a trimesh in the prim's local frame."""
    type_name = prim.GetTypeName()
    if type_name == "Cube":
        cube = UsdGeom.Cube(prim)
        size_attr = cube.GetSizeAttr()
        size = float(size_attr.Get(time)) if size_attr else 2.0
        return trimesh.creation.box(extents=np.full(3, size))
    if type_name != "Mesh":
        return None

    mesh_schema = UsdGeom.Mesh(prim)
    points_attr = mesh_schema.GetPointsAttr()
    indices_attr = mesh_schema.GetFaceVertexIndicesAttr()
    counts_attr = mesh_schema.GetFaceVertexCountsAttr()
    if not points_attr or not indices_attr:
        return None

    points = points_attr.Get(time)
    indices = indices_attr.Get(time)
    if points is None or indices is None:
        return None

    points_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    indices_np = np.asarray(indices, dtype=np.int32)
    if points_np.size == 0 or indices_np.size == 0:
        return None

    if counts_attr and counts_attr.Get(time) is not None:
        counts_np = np.asarray(counts_attr.Get(time), dtype=np.int32)
    else:
        counts_np = np.full(max(1, len(indices_np) // 3), 3, dtype=np.int32)

    triangles = _triangulate_face_indices(indices_np, counts_np)
    if len(triangles) == 0:
        return None
    return trimesh.Trimesh(vertices=points_np, faces=triangles, process=False)


def transform_trimesh(mesh: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` with ``matrix`` applied to its vertices."""
    out = mesh.copy()
    out.apply_transform(matrix)
    return out


def _path_role(prim: Usd.Prim, body: Usd.Prim) -> tuple[bool, bool]:
    """Infer (is_collision, is_visual) from the prim path under the body."""
    rel = str(prim.GetPath().MakeRelativePath(body.GetPath())).lower()
    under_collisions = rel.startswith("collisions") or "/collisions/" in f"/{rel}/"
    under_visuals = rel.startswith("visuals") or "/visuals/" in f"/{rel}/"

    has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
    parent = prim.GetParent()
    while parent and parent != body and not has_collision_api:
        if parent.HasAPI(UsdPhysics.CollisionAPI):
            has_collision_api = True
            break
        parent = parent.GetParent()

    if under_visuals:
        # Gripper-style assets put CollisionAPI on the visuals xform itself.
        return has_collision_api, True
    if under_collisions:
        return True, False
    if has_collision_api:
        return True, False
    return False, False


def _finalize_body_geoms(geoms: list[BodyGeom]) -> list[BodyGeom]:
    """Drop duplicate collision meshes when visuals exist; promote visuals as needed."""
    visual_meshes = [
        g for g in geoms if g.kind == "mesh" and "/visuals/" in g.prim_path.lower()
    ]
    collision_meshes = [
        g for g in geoms if g.kind == "mesh" and "/collisions/" in g.prim_path.lower()
    ]
    other = [g for g in geoms if g.kind != "mesh" or (
        "/visuals/" not in g.prim_path.lower() and "/collisions/" not in g.prim_path.lower()
    )]
    primitives = [g for g in other if g.kind != "mesh"]
    misc_meshes = [g for g in other if g.kind == "mesh"]

    if visual_meshes:
        # Prefer visual meshes. When there are no primitive colliders, reuse visuals
        # as collision meshes instead of duplicating the /collisions copies.
        selected_meshes = list(visual_meshes)
        if not primitives:
            for g in selected_meshes:
                g.is_collision = True
    else:
        selected_meshes = list(collision_meshes)
        for g in selected_meshes:
            g.is_visual = True
            if primitives:
                g.is_collision = False

    return selected_meshes + primitives + misc_meshes


def _unique_geom_name(body_name: str, prim: Usd.Prim, used: set[str]) -> str:
    parts = [body_name] + [p for p in str(prim.GetPath()).split("/") if p][-2:]
    base = "_".join(parts).replace(" ", "_")
    name = base
    idx = 1
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def _axis_index(axis: str | None) -> int:
    if not axis:
        return 2
    return {"X": 0, "Y": 1, "Z": 2}.get(str(axis).upper(), 2)


def extract_body_geoms(
    body: Usd.Prim,
    *,
    time: Usd.TimeCode | None = None,
) -> list[BodyGeom]:
    """Extract mesh and primitive geoms under a rigid-body prim, in body-local frame.

    Triangle meshes have their body-relative transform baked into vertices (geom pose
    is identity). Primitive geoms keep an explicit pos/quat/size.
    """
    time = time if time is not None else Usd.TimeCode.Default()
    body_name = body.GetName()
    used_names: set[str] = set()
    geoms: list[BodyGeom] = []

    # Traverse instance proxies so instanceable visuals/collisions are included.
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim == body:
            continue
        # Bodies are siblings in Isaac Lab USDs; still guard against nested rigid bodies.
        ancestor = prim.GetParent()
        nested = False
        while ancestor and ancestor != body:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                nested = True
                break
            ancestor = ancestor.GetParent()
        if nested or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        type_name = prim.GetTypeName()
        is_collision, is_visual = _path_role(prim, body)
        if not (is_collision or is_visual):
            # Still consider typed geom prims under the body (e.g. grasp Sphere).
            if type_name not in ("Mesh", "Cube", "Sphere", "Capsule", "Cylinder"):
                continue
            is_visual = True
            is_collision = False

        matrix = relative_transform(prim, body, time)
        name = _unique_geom_name(body_name, prim, used_names)
        path_str = str(prim.GetPath())

        if type_name == "Mesh":
            mesh = usd_mesh_to_trimesh(prim, time)
            if mesh is None:
                continue
            geoms.append(
                BodyGeom(
                    name=name,
                    kind="mesh",
                    is_collision=is_collision,
                    is_visual=is_visual,
                    pos=np.zeros(3),
                    quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    size=np.zeros(3),
                    mesh=transform_trimesh(mesh, matrix),
                    prim_path=path_str,
                )
            )
            continue

        if type_name == "Cube":
            pos, quat, scale = decompose_pos_quat_scale(matrix)
            cube = UsdGeom.Cube(prim)
            size_attr = cube.GetSizeAttr()
            edge = float(size_attr.Get(time)) if size_attr else 2.0
            half_extents = 0.5 * edge * np.abs(scale)
            geoms.append(
                BodyGeom(
                    name=name,
                    kind="box",
                    is_collision=is_collision,
                    is_visual=is_visual,
                    pos=pos,
                    quat_wxyz=quat,
                    size=half_extents,
                    prim_path=path_str,
                )
            )
            continue

        if type_name == "Sphere":
            pos, quat, scale = decompose_pos_quat_scale(matrix)
            sphere = UsdGeom.Sphere(prim)
            radius = float(sphere.GetRadiusAttr().Get(time) or 1.0)
            radius *= float(np.mean(np.abs(scale)))
            geoms.append(
                BodyGeom(
                    name=name,
                    kind="sphere",
                    is_collision=is_collision,
                    is_visual=is_visual,
                    pos=pos,
                    quat_wxyz=quat,
                    size=np.array([radius, 0.0, 0.0]),
                    prim_path=path_str,
                )
            )
            continue

        if type_name == "Capsule":
            pos, quat, scale = decompose_pos_quat_scale(matrix)
            capsule = UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get(time) or 1.0)
            height = float(capsule.GetHeightAttr().Get(time) or 1.0)
            axis = _axis_index(capsule.GetAxisAttr().Get(time))
            radial = [i for i in range(3) if i != axis]
            radius *= float(np.mean(np.abs(scale[radial])))
            half_length = 0.5 * height * float(abs(scale[axis]))
            geoms.append(
                BodyGeom(
                    name=name,
                    kind="capsule",
                    is_collision=is_collision,
                    is_visual=is_visual,
                    pos=pos,
                    quat_wxyz=quat,
                    size=np.array([radius, half_length, 0.0]),
                    prim_path=path_str,
                )
            )
            continue

        if type_name == "Cylinder":
            pos, quat, scale = decompose_pos_quat_scale(matrix)
            cylinder = UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get(time) or 1.0)
            height = float(cylinder.GetHeightAttr().Get(time) or 1.0)
            axis = _axis_index(cylinder.GetAxisAttr().Get(time))
            radial = [i for i in range(3) if i != axis]
            radius *= float(np.mean(np.abs(scale[radial])))
            half_length = 0.5 * height * float(abs(scale[axis]))
            geoms.append(
                BodyGeom(
                    name=name,
                    kind="cylinder",
                    is_collision=is_collision,
                    is_visual=is_visual,
                    pos=pos,
                    quat_wxyz=quat,
                    size=np.array([radius, half_length, 0.0]),
                    prim_path=path_str,
                )
            )
            continue

    return _finalize_body_geoms(geoms)


def extract_meshes(
    usd_path: str | Path,
    *,
    root_prim: Usd.Prim | None = None,
    time: Usd.TimeCode | None = None,
) -> dict[str, trimesh.Trimesh]:
    """Extract all Mesh prims from a USD stage as trimesh.Trimesh instances.

    Traverses the stage from the default prim (or the given root_prim). Names are
    taken from the prim name; duplicate names overwrite earlier entries. Prefer
    :func:`extract_body_geoms` when converting robots (unique names + body frames).

    Raises:
        FileNotFoundError: If usd_path does not exist.
        RuntimeError: If the stage cannot be opened or has no default prim and
            root_prim is None.
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
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in ("Mesh", "Cube"):
            continue
        tmesh = usd_mesh_to_trimesh(prim, time)
        if tmesh is not None:
            meshes[prim.GetName()] = tmesh
    return meshes


def export_meshes(
    meshes: dict[str, trimesh.Trimesh],
    output_dir: str | Path,
    *,
    fmt: str = "stl",
) -> dict[str, Path]:
    """Write meshes to ``output_dir`` and return a name -> path mapping."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = fmt.lower().lstrip(".")
    written: dict[str, Path] = {}
    for name, mesh in meshes.items():
        out_path = out_dir / f"{name}.{ext}"
        mesh.export(str(out_path))
        written[name] = out_path
    return written
