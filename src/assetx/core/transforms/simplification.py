from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import MujocoAsset
from assetx.core.transforms._geom import (
    compile_asset_spec,
    geom_points_world,
    points_in_body_frame,
)
from assetx.core.transforms.base import Transform


class ReplaceCylinderWithCapsule(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        return replace(asset, spec=spec)


@dataclass(frozen=True)
class AABBFit:
    """Axis-aligned box pose/size in a host body frame (MuJoCo conventions)."""

    pos: tuple[float, float, float]
    half_sizes: tuple[float, float, float]  # MuJoCo box size (half-extents)


@dataclass(frozen=True)
class CapsuleFit:
    """Capsule pose/size in a host body frame (MuJoCo conventions)."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # wxyz; local +Z is the axis
    radius: float
    half_height: float  # cylindrical half-length (excludes hemispherical caps)


def fit_aabb(points: np.ndarray) -> AABBFit:
    """Fit an axis-aligned bounding box in the same frame as ``points``."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("points must be non-empty")

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return AABBFit(
        pos=tuple(float(x) for x in center),
        half_sizes=tuple(float(x) for x in half),
    )


def _quat_wxyz_align_z(direction: np.ndarray) -> np.ndarray:
    """Return wxyz quaternion mapping local +Z onto ``direction``."""
    axis = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    z = np.array([0.0, 0.0, 1.0])
    rot, _ = sRot.align_vectors(axis.reshape(1, 3), z.reshape(1, 3))
    return np.asarray(rot.as_quat(scalar_first=True), dtype=float)


def fit_capsule_pca(points: np.ndarray) -> CapsuleFit:
    """Fit a minimum encapsulating capsule along the PCA principal axis.

    Radius is the max distance to the principal axis through the centroid.
    Half-height is the shortest cylinder segment whose hemispherical caps still
    cover every point at that radius.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("points must be non-empty")

    if pts.shape[0] == 1:
        pos = tuple(float(x) for x in pts[0])
        return CapsuleFit(pos=pos, quat=(1.0, 0.0, 0.0, 0.0), radius=0.0, half_height=0.0)

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = np.asarray(eigvecs[:, int(np.argmax(eigvals))], dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        radius = float(np.linalg.norm(centered, axis=1).max())
        pos = tuple(float(x) for x in centroid)
        return CapsuleFit(pos=pos, quat=(1.0, 0.0, 0.0, 0.0), radius=radius, half_height=0.0)
    axis = axis / axis_norm

    axial = centered @ axis
    radial = np.linalg.norm(centered - np.outer(axial, axis), axis=1)
    radius = float(radial.max())
    if radius < 1e-12:
        radius = 1e-6
        lo = float(axial.min())
        hi = float(axial.max())
        center_t = 0.5 * (lo + hi)
        half_height = 0.5 * (hi - lo)
    else:
        slack = np.sqrt(np.clip(radius * radius - radial * radial, 0.0, None))
        lo = float(np.min(axial + slack))
        hi = float(np.max(axial - slack))
        if lo <= hi:
            center_t = 0.5 * (lo + hi)
            half_height = 0.5 * (hi - lo)
        else:
            center_t = 0.5 * (lo + hi)
            half_height = 0.0

    pos = centroid + center_t * axis
    quat = _quat_wxyz_align_z(axis)
    return CapsuleFit(
        pos=tuple(float(x) for x in pos),
        quat=tuple(float(x) for x in quat),
        radius=radius,
        half_height=float(half_height),
    )


def _resolve_group_host(
    *,
    label: str,
    geom_names: list[str],
    geoms_by_name: dict[str, mujoco.MjsGeom],
    body_paths: list[str | None] | None,
    group_idx: int,
    spec: mujoco.MjSpec,
) -> tuple[str, mujoco.MjsBody, mujoco.MjsGeom]:
    missing = [n for n in geom_names if n not in geoms_by_name]
    if missing:
        raise ValueError(f"{label}: geom(s) not found: {missing}")

    first_spec_geom = geoms_by_name[geom_names[0]]
    host_path = (
        body_paths[group_idx]
        if body_paths is not None and body_paths[group_idx] is not None
        else first_spec_geom.parent.name
    )
    if not host_path:
        raise ValueError(f"{label}: host body for group {geom_names} has empty name")
    host_body = spec.body(host_path)
    if host_body is None:
        raise ValueError(f"{label}: body {host_path!r} not found")
    return host_path, host_body, first_spec_geom


def _sample_group_points_in_body(
    *,
    label: str,
    geom_names: list[str],
    host_path: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    host_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, host_path)
    if host_id < 0:
        raise ValueError(f"{label}: compiled model missing body {host_path!r}")
    body_xpos = np.asarray(data.xpos[host_id], dtype=float)
    body_xmat = np.asarray(data.xmat[host_id], dtype=float).reshape(3, 3)

    point_chunks: list[np.ndarray] = []
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"{label}: compiled model missing geom {name!r}")
        point_chunks.append(geom_points_world(model, data, gid))
    return points_in_body_frame(np.vstack(point_chunks), body_xpos, body_xmat)


def _copy_contact_attrs(dst: mujoco.MjsGeom, template: mujoco.MjsGeom) -> None:
    dst.contype = template.contype
    dst.conaffinity = template.conaffinity
    dst.condim = template.condim
    dst.friction = template.friction
    dst.group = template.group
    dst.density = template.density


def _default_approx_rgba(
    template: mujoco.MjsGeom,
    rgba: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if rgba is not None:
        return rgba
    color = np.asarray(template.rgba, dtype=float).copy()
    color[3] = min(float(color[3]), 0.5)
    return tuple(float(x) for x in color)


class ApproximateWithAABB(Transform):
    """Add an axis-aligned box approximating each group of geoms.

    Points are sampled from the target geoms and expressed in a host body frame
    at qpos0. The fitted box is axis-aligned in that frame (identity quat).

    Parameters
    ----------
    groups:
        Each inner list is one box: the named geoms it approximates.
    names:
        Optional box geom names (defaults to ``{first}_aabb``).
    body_paths:
        Optional host body for each group (defaults to the parent of the first
        geom).
    replace:
        If True (default), delete the original geoms after adding the box.
    size_scale:
        Uniform or per-axis multiplier applied to fitted half-extents
        (default 1.0).
    rgba:
        Box color; defaults to the first geom's rgba with alpha 0.5.

    Example
    -------
    ::

        ApproximateWithAABB(
            [["link_collision"]],
            names=["link_box"],
            replace=True,
        )
    """

    def __init__(
        self,
        groups: list[list[str]],
        *,
        names: list[str] | None = None,
        body_paths: list[str | None] | None = None,
        replace: bool = True,
        size_scale: float | tuple[float, float, float] = 1.0,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not groups:
            raise ValueError("ApproximateWithAABB requires at least one geom group")
        for i, group in enumerate(groups):
            if not group:
                raise ValueError(f"ApproximateWithAABB: group {i} is empty")
        if isinstance(size_scale, (int, float)):
            scale = (float(size_scale),) * 3
        else:
            scale = tuple(float(x) for x in size_scale)
            if len(scale) != 3:
                raise ValueError("size_scale must be a float or a length-3 tuple")
        if any(s <= 0.0 for s in scale):
            raise ValueError(f"size_scale components must be > 0, got {scale}")
        self.groups = [list(g) for g in groups]
        self.names = list(names) if names is not None else None
        self.body_paths = list(body_paths) if body_paths is not None else None
        self.replace = bool(replace)
        self.size_scale = scale
        self.rgba = rgba
        if self.names is not None and len(self.names) != len(self.groups):
            raise ValueError("names must have the same length as groups")
        if self.body_paths is not None and len(self.body_paths) != len(self.groups):
            raise ValueError("body_paths must have the same length as groups")

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        geoms_by_name = {g.name: g for g in spec.geoms if g.name}
        planned: list[tuple[list[str], str, str, AABBFit, mujoco.MjsGeom]] = []

        model = compile_asset_spec(asset, spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for group_idx, geom_names in enumerate(self.groups):
            host_path, _, first_spec_geom = _resolve_group_host(
                label="ApproximateWithAABB",
                geom_names=geom_names,
                geoms_by_name=geoms_by_name,
                body_paths=self.body_paths,
                group_idx=group_idx,
                spec=spec,
            )
            points_body = _sample_group_points_in_body(
                label="ApproximateWithAABB",
                geom_names=geom_names,
                host_path=host_path,
                model=model,
                data=data,
            )
            fit = fit_aabb(points_body)
            box_name = (
                self.names[group_idx]
                if self.names is not None
                else f"{geom_names[0]}_aabb"
            )
            planned.append((geom_names, box_name, host_path, fit, first_spec_geom))

        for geom_names, box_name, host_path, fit, template in planned:
            if self.replace:
                for name in geom_names:
                    geom = geoms_by_name.get(name)
                    if geom is not None:
                        spec.delete(geom)
                        geoms_by_name.pop(name, None)

            host_body = spec.body(host_path)
            assert host_body is not None
            box = host_body.add_geom()
            box.name = box_name
            box.type = mujoco.mjtGeom.mjGEOM_BOX
            box.size = tuple(
                half * scale for half, scale in zip(fit.half_sizes, self.size_scale)
            )
            box.pos = fit.pos
            box.quat = (1.0, 0.0, 0.0, 0.0)
            _copy_contact_attrs(box, template)
            box.rgba = _default_approx_rgba(template, self.rgba)

        spec.compile()
        return replace(asset, spec=spec)


class ApproximateWithCapsule(Transform):
    """Add a PCA capsule approximating each group of geoms.

    For every group, samples points from the target geoms (mesh vertices or
    primitive surface samples), fits a PCA-aligned encapsulating capsule, and
    attaches it to a host body. Optionally deletes the original geoms.

    Parameters
    ----------
    groups:
        Each inner list is one capsule: the named geoms it approximates.
    names:
        Optional capsule geom names (defaults to ``{first}_capsule``).
    body_paths:
        Optional host body for each group (defaults to the parent of the first
        geom). Points are expressed in that body frame at the model's default
        configuration (qpos0).
    replace:
        If True (default), delete the original geoms after adding the capsule.
    radius_scale:
        Multiplier applied to the fitted capsule radius (default 1.0).
    height_scale:
        Multiplier applied to the fitted cylindrical half-height (default 1.0).
    rgba:
        Capsule color; defaults to the first geom's rgba with alpha 0.5.

    Example
    -------
    ::

        ApproximateWithCapsule(
            [["finger_collision"]],
            names=["finger_capsule"],
            replace=True,
            radius_scale=0.95,
        )
    """

    def __init__(
        self,
        groups: list[list[str]],
        *,
        names: list[str] | None = None,
        body_paths: list[str | None] | None = None,
        replace: bool = True,
        radius_scale: float = 1.0,
        height_scale: float = 1.0,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not groups:
            raise ValueError("ApproximateWithCapsule requires at least one geom group")
        for i, group in enumerate(groups):
            if not group:
                raise ValueError(f"ApproximateWithCapsule: group {i} is empty")
        if radius_scale <= 0.0:
            raise ValueError(f"radius_scale must be > 0, got {radius_scale}")
        if height_scale < 0.0:
            raise ValueError(f"height_scale must be >= 0, got {height_scale}")
        self.groups = [list(g) for g in groups]
        self.names = list(names) if names is not None else None
        self.body_paths = list(body_paths) if body_paths is not None else None
        self.replace = bool(replace)
        self.radius_scale = float(radius_scale)
        self.height_scale = float(height_scale)
        self.rgba = rgba
        if self.names is not None and len(self.names) != len(self.groups):
            raise ValueError("names must have the same length as groups")
        if self.body_paths is not None and len(self.body_paths) != len(self.groups):
            raise ValueError("body_paths must have the same length as groups")

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        geoms_by_name = {g.name: g for g in spec.geoms if g.name}
        planned: list[tuple[list[str], str, str, CapsuleFit, mujoco.MjsGeom]] = []

        model = compile_asset_spec(asset, spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for group_idx, geom_names in enumerate(self.groups):
            host_path, _, first_spec_geom = _resolve_group_host(
                label="ApproximateWithCapsule",
                geom_names=geom_names,
                geoms_by_name=geoms_by_name,
                body_paths=self.body_paths,
                group_idx=group_idx,
                spec=spec,
            )
            points_body = _sample_group_points_in_body(
                label="ApproximateWithCapsule",
                geom_names=geom_names,
                host_path=host_path,
                model=model,
                data=data,
            )
            fit = fit_capsule_pca(points_body)
            capsule_name = (
                self.names[group_idx]
                if self.names is not None
                else f"{geom_names[0]}_capsule"
            )
            planned.append((geom_names, capsule_name, host_path, fit, first_spec_geom))

        for geom_names, capsule_name, host_path, fit, template in planned:
            if self.replace:
                for name in geom_names:
                    geom = geoms_by_name.get(name)
                    if geom is not None:
                        spec.delete(geom)
                        geoms_by_name.pop(name, None)

            host_body = spec.body(host_path)
            assert host_body is not None
            capsule = host_body.add_geom()
            capsule.name = capsule_name
            capsule.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            capsule.size = (
                fit.radius * self.radius_scale,
                fit.half_height * self.height_scale,
                0.0,
            )
            capsule.pos = fit.pos
            capsule.quat = fit.quat
            _copy_contact_attrs(capsule, template)
            capsule.rgba = _default_approx_rgba(template, self.rgba)

        spec.compile()
        return replace(asset, spec=spec)
