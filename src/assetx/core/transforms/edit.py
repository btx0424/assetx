from __future__ import annotations

import re
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import JointCfg, MujocoAsset
from assetx.core.transforms._geom import compile_asset_spec
from assetx.core.transforms.base import Transform


def _name_matches(name: str, patterns: list[str]) -> bool:
    """Return True if ``name`` fully matches any regex in ``patterns``."""
    if not name:
        return False
    return any(re.fullmatch(pat, name) is not None for pat in patterns)


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


class RemoveSensors(Transform):
    """Delete sensors whose names match any of the given regex patterns.

    Example
    -------
    ::

        RemoveSensors(names=[".*pos", ".*torque", "imu.*"])
    """

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for sensor in list(spec.sensors):
            if _name_matches(sensor.name or "", self.names):
                spec.delete(sensor)
        spec.compile()
        return replace(asset, spec=spec)


class RemoveActuators(Transform):
    """Delete actuators whose names match any of the given regex patterns.

    Keyframe ``ctrl`` vectors are cleared when their length no longer matches
    the remaining actuator count (common after stripping vendor actuators).

    Example
    -------
    ::

        RemoveActuators(names=[".*"])  # strip all actuators
    """

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for actuator in list(spec.actuators):
            if _name_matches(actuator.name or "", self.names):
                spec.delete(actuator)
        n_act = len(list(spec.actuators))
        for key in list(spec.keys):
            ctrl = getattr(key, "ctrl", None)
            if ctrl is not None and len(ctrl) != n_act:
                key.ctrl = []
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

    ``align_to`` sets both pose semantics at qpos0 (``"world"`` or a body
    name): local axes match the reference, and ``pos`` is an offset from
    the parent origin expressed in that reference frame. Pass
    ``align_to=None`` to keep ``pos`` / ``quat`` parent-relative.

    Example
    -------
    ::

        AddDummyBody(
            parent_path="gripper_base",
            name="grasp_point",
            pos=(0.05, 0.0, 0.0),  # 5 cm along world +X
            align_to="world",
        )
    """

    def __init__(
        self,
        parent_path: str,
        name: str,
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        align_to: str | None = "world",
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
        self.align_to = align_to
        self.marker = bool(marker)
        self.marker_size = size
        self.marker_type = _MARKER_TYPE_MAP[marker_type]
        self.rgba = rgba
        self.geom_name = geom_name

    def _aligned_pose(
        self, asset: MujocoAsset, spec: mujoco.MjSpec
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Parent-frame (pos, quat) matching ``align_to`` at qpos0."""
        model = compile_asset_spec(asset, spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        parent_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.parent_path
        )
        if parent_id < 0:
            raise ValueError(
                f"AddDummyBody: compiled model missing body {self.parent_path!r}"
            )
        parent_xmat = np.asarray(data.xmat[parent_id], dtype=float).reshape(3, 3)

        if self.align_to == "world":
            ref_xmat = np.eye(3, dtype=float)
        else:
            assert self.align_to is not None
            ref_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.align_to
            )
            if ref_id < 0:
                raise ValueError(
                    f"AddDummyBody: align_to body {self.align_to!r} not found"
                )
            ref_xmat = np.asarray(data.xmat[ref_id], dtype=float).reshape(3, 3)

        # Offset in ref axes -> parent local: p_parent = R_parent.T @ R_ref @ p_ref
        pos_ref = np.asarray(self.pos, dtype=float)
        pos_parent = parent_xmat.T @ (ref_xmat @ pos_ref)

        # R_parent @ R_local = R_ref  =>  R_local = R_parent.T @ R_ref
        r_local = sRot.from_matrix(parent_xmat.T @ ref_xmat)
        quat_parent = tuple(float(x) for x in r_local.as_quat(scalar_first=True))
        return tuple(float(x) for x in pos_parent), quat_parent

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        parent = spec.body(self.parent_path)
        if parent is None:
            raise ValueError(f"AddDummyBody: body {self.parent_path!r} not found")
        if spec.body(self.name) is not None:
            raise ValueError(f"AddDummyBody: body {self.name!r} already exists")

        if self.align_to is not None:
            child_pos, child_quat = self._aligned_pose(asset, spec)
        else:
            child_pos, child_quat = self.pos, self.quat

        child = parent.add_body()
        child.name = self.name
        child.pos = child_pos
        child.quat = child_quat
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
