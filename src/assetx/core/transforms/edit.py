from __future__ import annotations

from dataclasses import replace

import mujoco

from assetx.core.asset import JointCfg, MujocoAsset
from assetx.core.transforms.base import Transform


class RemoveJoints(Transform):
    def __init__(self, joint_names: list[str]) -> None:
        self.joint_names = joint_names

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for joint in list(spec.joints):
            if joint.name in self.joint_names:
                spec.delete(joint)
        spec.compile()
        return replace(asset, spec=spec)


class AddJoint(Transform):
    _JOINT_TYPE_MAP = {
        "hinge": mujoco.mjtJoint.mjJNT_HINGE,
        "slide": mujoco.mjtJoint.mjJNT_SLIDE,
        "free": mujoco.mjtJoint.mjJNT_FREE,
    }

    def __init__(self, body_path: str, joint_cfg: JointCfg) -> None:
        self.body_path = body_path
        self.joint_cfg = joint_cfg

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        body = spec.body(self.body_path)
        if body is None:
            raise ValueError(f"AddJoint: body {self.body_path!r} not found")
        if self.joint_cfg.type == "fixed":
            spec.compile()
            return replace(asset, spec=spec)
        joint = body.add_joint()
        joint.name = self.joint_cfg.name
        joint_type = self._JOINT_TYPE_MAP[self.joint_cfg.type]
        joint.type = joint_type
        if joint_type in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}:
            joint.axis = self.joint_cfg.axis
            joint.limited = self.joint_cfg.limited
            if self.joint_cfg.limited:
                joint.range = self.joint_cfg.range
        spec.compile()
        return replace(asset, spec=spec)


_MARKER_TYPE_MAP = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}


class AddSite(Transform):
    """Add a site to the asset."""

    def __init__(
        self,
        body_path: str,
        name: str,
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        size: tuple[float, float, float] = (0.005, 0.005, 0.005),
        type: str = "sphere",
        rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    ) -> None:
        if type not in _MARKER_TYPE_MAP:
            raise ValueError(f"Invalid site type: {type}")
        self.body_path = body_path
        self.name = name
        self.pos = pos
        self.quat = quat
        self.size = size
        self.type = _MARKER_TYPE_MAP[type]
        self.rgba = rgba

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        body = spec.body(self.body_path)
        if body is None:
            raise ValueError(f"AddSite: body {self.body_path!r} not found")

        site = body.add_site()
        site.name = self.name
        site.pos = self.pos
        site.quat = self.quat
        site.size = self.size
        site.type = self.type
        site.rgba = self.rgba
        spec.compile()
        return replace(asset, spec=spec)


class AddDummyBody(Transform):
    """Add a fixed, zero-mass child body under ``parent_path``.

    Prefer this over :class:`AddSite` when a named link is required by
    downstream pipelines (e.g. Isaac / USD), which often drop MJCF sites.

    By default attaches a visual-only marker geom (``contype=0``,
    ``conaffinity=0``) for preview, matching gripper grasp-point recipes.

    Example
    -------
    ::

        AddDummyBody(
            parent_path="gripper_base",
            name="grasp_point",
            pos=(0.05, 0.0, 0.0),
        )
    """

    def __init__(
        self,
        parent_path: str,
        name: str,
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        marker: bool = True,
        marker_size: float | tuple[float, float, float] = 0.01,
        marker_type: str = "sphere",
        rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.6),
        geom_name: str | None = None,
    ) -> None:
        if marker_type not in _MARKER_TYPE_MAP:
            raise ValueError(f"Invalid marker type: {marker_type}")
        if isinstance(marker_size, (int, float)):
            size = (float(marker_size), float(marker_size), float(marker_size))
        else:
            size = tuple(float(x) for x in marker_size)
            if len(size) != 3:
                raise ValueError("marker_size must be a float or a length-3 tuple")
        self.parent_path = parent_path
        self.name = name
        self.pos = pos
        self.quat = quat
        self.marker = bool(marker)
        self.marker_size = size
        self.marker_type = _MARKER_TYPE_MAP[marker_type]
        self.rgba = rgba
        self.geom_name = geom_name

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        parent = spec.body(self.parent_path)
        if parent is None:
            raise ValueError(f"AddDummyBody: body {self.parent_path!r} not found")
        if spec.body(self.name) is not None:
            raise ValueError(f"AddDummyBody: body {self.name!r} already exists")

        child = parent.add_body()
        child.name = self.name
        child.pos = self.pos
        child.quat = self.quat
        child.mass = 0.0
        child.ipos = (0.0, 0.0, 0.0)
        child.inertia = (0.0, 0.0, 0.0)
        child.iquat = (1.0, 0.0, 0.0, 0.0)
        child.explicitinertial = 1

        if self.marker:
            geom = child.add_geom()
            geom.name = self.geom_name if self.geom_name is not None else f"{self.name}_visual"
            geom.type = self.marker_type
            geom.size = self.marker_size
            geom.rgba = self.rgba
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1
            geom.density = 0

        spec.compile()
        return replace(asset, spec=spec)


class RemoveGeoms(Transform):
    def __init__(self, geom_names: list[str]) -> None:
        self.geom_names = geom_names

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in list(spec.geoms):
            if geom.name in self.geom_names:
                spec.delete(geom)
        spec.compile()
        return replace(asset, spec=spec)


class RenameBodies(Transform):
    def __init__(self, body_names: dict[str, str]) -> None:
        self.body_names = body_names

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for body in spec.bodies:
            if body.name in self.body_names:
                body.name = self.body_names[body.name]
        spec.compile()
        return replace(asset, spec=spec)


class NormalizeGeomNames(Transform):
    """Assign proper names for geoms without names.

    For pure visual geoms, name as body_name_visual{i}
    For collision geoms, name as body_name_collision{i}
    """

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        used_names = {geom.name for geom in spec.geoms if geom.name}
        next_indices: dict[tuple[str, str], int] = {}

        for geom in spec.geoms:
            if geom.name:
                continue

            body_name = geom.parent.name
            if not body_name:
                raise ValueError(
                    "NormalizeGeomNames: cannot name a geom whose parent body has no name"
                )

            role = (
                "visual"
                if int(geom.contype) == 0 and int(geom.conaffinity) == 0
                else "collision"
            )
            key = (body_name, role)
            index = next_indices.get(key, 0)
            candidate = f"{body_name}_{role}{index}"
            while candidate in used_names:
                index += 1
                candidate = f"{body_name}_{role}{index}"

            geom.name = candidate
            used_names.add(candidate)
            next_indices[key] = index + 1

        spec.compile()
        return replace(asset, spec=spec)
