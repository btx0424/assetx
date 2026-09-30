"""Voxel IoU regression test for A2 front-calf capsule approximation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation as sRot

from assetx import (
    ApproximateWithCapsule,
    CapsuleFit,
    MujocoAsset,
    NormalizeGeomNames,
    fit_capsule_chain_pca,
)
from assetx.core.transforms._geom import (
    compile_asset_spec,
    geom_points_world,
    mesh_vertices_local,
    points_in_body_frame,
)

_VENDOR = Path(__file__).resolve().parents[1] / "artifacts" / "vendor" / "a2_description"
_A2_XML = _VENDOR / "a2.xml"
_CALF_GEOM = "FL_calf_visual0"
_CALF_BODY = "FL_calf"
_MESH_NAME = "left_front_Link3"
_VOXEL_PITCH = 0.003

# Default-chain settings with radius_scale=1.0 should stay within these bounds.
_MIN_IOU = 0.07
_MIN_PRECISION = 0.08
_MIN_RECALL = 0.30
_NUM_CAPSULES = 2


@dataclass(frozen=True)
class VoxelOverlap:
    iou: float
    precision: float
    recall: float


def _capsule_mesh(fit: CapsuleFit) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:3, :3] = sRot.from_quat(fit.quat, scalar_first=True).as_matrix()
    transform[:3, 3] = np.asarray(fit.pos, dtype=float)
    return trimesh.creation.capsule(
        radius=max(float(fit.radius), 1e-6),
        height=max(2.0 * float(fit.half_height), 1e-6),
        transform=transform,
    )


def _voxel_keys(mesh: trimesh.Trimesh, pitch: float) -> set[tuple[int, int, int]]:
    voxels = mesh.voxelized(pitch).fill()
    return set(map(tuple, np.round(voxels.points / pitch).astype(int)))


def voxel_overlap(
    target_mesh: trimesh.Trimesh,
    capsule_fits: list[CapsuleFit],
    *,
    pitch: float = _VOXEL_PITCH,
) -> VoxelOverlap:
    """Compute voxel IoU between a mesh and a union of capsule fits."""
    target_keys = _voxel_keys(target_mesh, pitch)
    capsule_mesh = trimesh.util.concatenate(
        [_capsule_mesh(fit) for fit in capsule_fits]
    )
    capsule_keys = _voxel_keys(capsule_mesh, pitch)
    intersection = len(target_keys & capsule_keys)
    union = len(target_keys | capsule_keys)
    if union == 0:
        return VoxelOverlap(iou=0.0, precision=0.0, recall=0.0)
    precision = intersection / max(len(capsule_keys), 1)
    recall = intersection / max(len(target_keys), 1)
    return VoxelOverlap(
        iou=intersection / union,
        precision=precision,
        recall=recall,
    )


def _load_fl_calf_fixture() -> tuple[np.ndarray, trimesh.Trimesh]:
    if not _A2_XML.is_file():
        pytest.skip(f"A2 vendor asset not found at {_A2_XML}")

    asset = NormalizeGeomNames().transform(MujocoAsset.from_file(_A2_XML))
    model = compile_asset_spec(asset, asset.spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _CALF_GEOM)
    if geom_id < 0:
        pytest.skip(f"{_CALF_GEOM} not found in normalized A2 model")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _CALF_BODY)
    mesh_id = int(model.geom_dataid[geom_id])
    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    assert mesh_name == _MESH_NAME

    local_vertices = mesh_vertices_local(model, mesh_id)
    face_adr = int(model.mesh_faceadr[mesh_id])
    facenum = int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[face_adr : face_adr + facenum], dtype=int)
    target_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=faces,
        process=False,
    )

    sample_points = geom_points_world(model, data, geom_id)
    body_points = points_in_body_frame(
        sample_points,
        np.asarray(data.xpos[body_id], dtype=float),
        np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3),
    )
    return body_points, target_mesh


def test_a2_front_calf_capsule_chain_voxel_iou() -> None:
    body_points, target_mesh = _load_fl_calf_fixture()

    segments = fit_capsule_chain_pca(
        body_points,
        _NUM_CAPSULES,
        radius_scale=1.0,
    )
    assert len(segments) == _NUM_CAPSULES

    overlap = voxel_overlap(target_mesh, segments)
    single = fit_capsule_chain_pca(body_points, 1, radius_scale=1.0)
    single_overlap = voxel_overlap(target_mesh, single)
    assert overlap.precision > single_overlap.precision, (overlap, single_overlap)
    assert overlap.recall >= _MIN_RECALL, overlap
    assert overlap.precision >= _MIN_PRECISION, overlap
    assert overlap.iou >= _MIN_IOU, overlap


def test_approximate_with_capsule_a2_front_calf_voxel_iou() -> None:
    if not _A2_XML.is_file():
        pytest.skip(f"A2 vendor asset not found at {_A2_XML}")

    asset = NormalizeGeomNames().transform(MujocoAsset.from_file(_A2_XML))
    out = ApproximateWithCapsule(
        [[_CALF_GEOM]],
        names=["FL_calf_collision0"],
        num_capsules=_NUM_CAPSULES,
        replace=False,
    ).transform(asset)

    model = compile_asset_spec(out, out.spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, _MESH_NAME)
    local_vertices = mesh_vertices_local(model, mesh_id)
    face_adr = int(model.mesh_faceadr[mesh_id])
    facenum = int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[face_adr : face_adr + facenum], dtype=int)
    target_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=faces,
        process=False,
    )

    capsule_fits: list[CapsuleFit] = []
    for idx in range(_NUM_CAPSULES):
        geom = out.spec.geom(f"FL_calf_collision0_{idx}")
        assert geom is not None
        capsule_fits.append(
            CapsuleFit(
                pos=tuple(float(x) for x in geom.pos),
                quat=tuple(float(x) for x in geom.quat),
                radius=float(geom.size[0]),
                half_height=float(geom.size[1]),
            )
        )

    overlap = voxel_overlap(target_mesh, capsule_fits)
    assert overlap.recall >= _MIN_RECALL, overlap
    assert overlap.precision >= _MIN_PRECISION, overlap
    assert overlap.iou >= _MIN_IOU, overlap
