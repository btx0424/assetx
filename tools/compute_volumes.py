"""CLI: compute and print per-body volumes from visual meshes in an MJCF."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh


@dataclass(frozen=True)
class VolumeEstimate:
    volume: float
    method: str  # "watertight" | "signed" | "voxel" | "nan"


def _is_visual_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    """Visual geoms are non-colliding (contype = conaffinity = 0)."""
    return int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0


def _mesh_trimesh(model: mujoco.MjModel, mesh_id: int) -> trimesh.Trimesh:
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    fadr = int(model.mesh_faceadr[mesh_id])
    fnum = int(model.mesh_facenum[mesh_id])
    vertices = model.mesh_vert[vadr : vadr + vnum] * model.mesh_scale[mesh_id]
    faces = model.mesh_face[fadr : fadr + fnum]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _try_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light repair: normals, hole fill, then trimesh process/validate."""
    out = mesh.copy()
    try:
        trimesh.repair.fix_normals(out)
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(out)
    except Exception:
        pass
    try:
        out.process(validate=True)
    except Exception:
        pass
    return out


def _voxel_volume(mesh: trimesh.Trimesh) -> float | None:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if not np.all(np.isfinite(extents)) or float(np.max(extents)) <= 0.0:
        return None
    pitch = float(np.max(extents)) / 64.0
    try:
        return float(mesh.voxelized(pitch=pitch).volume)
    except Exception:
        return None


def _mesh_volume(mesh: trimesh.Trimesh) -> VolumeEstimate:
    """Estimate mesh volume with fallbacks for non-watertight CAD exports.

    Order:
      1. watertight ``abs(volume)`` on the raw mesh
      2. light repair, then watertight ``abs(volume)``
      3. winding-consistent signed ``abs(volume)`` (approx)
      4. voxelization (last resort)
      5. NaN
    """
    if mesh.is_watertight:
        return VolumeEstimate(float(abs(mesh.volume)), "watertight")

    repaired = _try_repair(mesh)
    if repaired.is_watertight:
        return VolumeEstimate(float(abs(repaired.volume)), "watertight")

    # Many nearly-closed solids report a sensible signed volume even when
    # ``is_watertight`` is False (tiny holes / non-manifold edges).
    for candidate in (repaired, mesh):
        if bool(getattr(candidate, "is_winding_consistent", False)):
            signed = float(candidate.volume)
            if math.isfinite(signed) and abs(signed) > 0.0:
                return VolumeEstimate(abs(signed), "signed")

    voxel = _voxel_volume(repaired)
    if voxel is not None and math.isfinite(voxel) and voxel > 0.0:
        return VolumeEstimate(voxel, "voxel")

    return VolumeEstimate(float("nan"), "nan")


def _geom_volume(model: mujoco.MjModel, geom_id: int) -> VolumeEstimate:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return VolumeEstimate(float("nan"), "nan")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return VolumeEstimate(float("nan"), "nan")
    return _mesh_volume(_mesh_trimesh(model, mesh_id))


def _merge_methods(methods: set[str]) -> str:
    if not methods or methods == {"nan"}:
        return "nan"
    finite = methods - {"nan"}
    if not finite:
        return "nan"
    # Prefer the coarsest / least trusted method present when summing geoms.
    for name in ("voxel", "signed", "watertight"):
        if name in finite:
            return name
    return next(iter(finite))


def compute_body_volumes(
    model: mujoco.MjModel,
) -> dict[str, VolumeEstimate]:
    """Sum visual-mesh volumes per body.

    Collision geoms are skipped. Overlaps between visual meshes on the same body
    are ignored (volumes are summed).
    """
    volumes: dict[str, float] = {}
    methods: dict[str, set[str]] = {}
    # Skip world body (id 0).
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        volumes[name] = 0.0
        methods[name] = set()

    for geom_id in range(model.ngeom):
        if not _is_visual_geom(model, geom_id):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id <= 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        estimate = _geom_volume(model, geom_id)
        methods.setdefault(name, set()).add(estimate.method)
        volumes.setdefault(name, 0.0)
        if math.isnan(estimate.volume) or math.isnan(volumes[name]):
            volumes[name] = float("nan")
        else:
            volumes[name] += estimate.volume

    return {
        name: VolumeEstimate(volumes[name], _merge_methods(methods[name]))
        for name in volumes
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-body volumes from visual mesh geoms in an MJCF. "
            "Collision geoms are ignored. Non-watertight meshes fall back to "
            "signed volume (if winding-consistent) or voxelization."
        ),
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        required=True,
        help="Path to the MJCF (.xml) file.",
    )
    args = parser.parse_args()

    mjcf_path = args.path.resolve()
    if not mjcf_path.is_file():
        raise SystemExit(f"MJCF not found: {mjcf_path}")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    volumes = compute_body_volumes(model)

    print(f"mjcf: {mjcf_path}")
    print(f"bodies: {len(volumes)}")
    print("volume_m3:")
    for name, estimate in volumes.items():
        if math.isnan(estimate.volume):
            print(f"  {name}: nan")
        else:
            print(f"  {name}: {estimate.volume:.8g}  # {estimate.method}")

    total = float(np.nansum([e.volume for e in volumes.values()]))
    counts = {k: 0 for k in ("watertight", "signed", "voxel", "nan")}
    for estimate in volumes.values():
        counts[estimate.method] = counts.get(estimate.method, 0) + 1
    print(f"sum_finite_m3: {total:.8g}")
    print(
        "methods: "
        + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    )


if __name__ == "__main__":
    main()
