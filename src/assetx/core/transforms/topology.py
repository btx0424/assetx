from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import MujocoAsset
from assetx.core.transforms.base import Transform


def _body_inertial_in_parent(
    body: mujoco.MjsBody, body_pos: np.ndarray, body_rot: sRot
) -> tuple[float, np.ndarray, np.ndarray]:
    ipos = np.asarray(body.ipos, dtype=float)
    com_parent = body_pos + body_rot.apply(ipos)
    inertia_diag = np.diag(np.asarray(body.inertia, dtype=float))
    inertia_rot = sRot.from_quat(body.iquat, scalar_first=True)
    rot = (body_rot * inertia_rot).as_matrix()
    inertia_about_com = rot @ inertia_diag @ rot.T
    return float(body.mass), com_parent, inertia_about_com


def _combine_inertials_in_parent(
    entries: list[tuple[float, np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray, np.ndarray]:
    total_mass = sum(mass for mass, _, _ in entries)
    com_parent = sum(mass * com for mass, com, _ in entries) / total_mass

    inertia_parent = np.zeros((3, 3))
    for mass, com, inertia_about_com in entries:
        offset = com - com_parent
        inertia_parent += inertia_about_com + mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return total_mass, com_parent, inertia_parent


def _set_body_inertial_in_frame(
    body: mujoco.MjsBody,
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
    frame_pos: np.ndarray,
    frame_rot: sRot,
) -> None:
    frame_rot_inv = frame_rot.inv()
    com_local = frame_rot_inv.apply(com - frame_pos)
    inertia_local = frame_rot_inv.as_matrix() @ inertia @ frame_rot_inv.as_matrix().T

    eigvals, eigvecs = np.linalg.eigh(inertia_local)
    eigvals = np.clip(eigvals, 1e-12, None)
    body.mass = mass
    body.ipos = tuple(com_local.tolist())
    body.inertia = tuple(eigvals.tolist())
    body.iquat = tuple(sRot.from_matrix(eigvecs).as_quat(scalar_first=True).tolist())
    body.explicitinertial = 1


def _copy_geom_to_body(
    geom: mujoco.MjsGeom,
    target_body: mujoco.MjsBody,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    *,
    name: str | None = None,
) -> mujoco.MjsGeom:
    new_geom = target_body.add_geom()
    new_geom.type = geom.type
    new_geom.size = geom.size
    new_geom.pos = pos
    new_geom.quat = quat
    new_geom.fromto = geom.fromto
    new_geom.rgba = geom.rgba
    new_geom.name = geom.name if name is None else name
    new_geom.contype = geom.contype
    new_geom.conaffinity = geom.conaffinity
    new_geom.mass = geom.mass
    new_geom.friction = geom.friction
    new_geom.condim = geom.condim
    new_geom.meshname = geom.meshname
    new_geom.density = geom.density
    new_geom.group = geom.group
    new_geom.priority = geom.priority
    new_geom.margin = geom.margin
    new_geom.gap = geom.gap
    new_geom.solref = geom.solref
    new_geom.solimp = geom.solimp
    new_geom.classname = geom.classname
    return new_geom


def _geom_role(geom: mujoco.MjsGeom) -> str:
    if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
        return "visual"
    return "collision"


def _assign_normalized_geom_names(
    body: mujoco.MjsBody, used_names: set[str]
) -> None:
    """Rename body geoms with the :class:`NormalizeGeomNames` naming rule."""
    next_indices: dict[str, int] = {}
    body_name = body.name
    if not body_name:
        raise ValueError("cannot normalize geom names on an unnamed body")
    for geom in body.geoms:
        role = _geom_role(geom)
        index = next_indices.get(role, 0)
        candidate = f"{body_name}_{role}{index}"
        while candidate in used_names:
            index += 1
            candidate = f"{body_name}_{role}{index}"
        geom.name = candidate
        used_names.add(candidate)
        next_indices[role] = index + 1


def _iter_subtree_bodies_with_pose_in_parent(
    root: mujoco.MjsBody,
) -> list[tuple[mujoco.MjsBody, np.ndarray, sRot]]:
    """Return subtree bodies with their pose expressed in root's parent frame."""
    root_pos = np.asarray(root.pos, dtype=float)
    root_rot = sRot.from_quat(root.quat, scalar_first=True)
    out: list[tuple[mujoco.MjsBody, np.ndarray, sRot]] = [(root, root_pos, root_rot)]
    stack = [(root, root_pos, root_rot)]
    while stack:
        body, body_pos, body_rot = stack.pop()
        for child in body.bodies:
            child_pos = body_pos + body_rot.apply(np.asarray(child.pos, dtype=float))
            child_rot = body_rot * sRot.from_quat(child.quat, scalar_first=True)
            out.append((child, child_pos, child_rot))
            stack.append((child, child_pos, child_rot))
    return out


class MergeBodies(Transform):
    """Merge bodies sharing the same parent. Recompute combined inertia in the first body's frame."""

    def __init__(self, body_paths: list[str]) -> None:
        self.body_paths = body_paths

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        if len(self.body_paths) < 2:
            raise ValueError("MergeBodies requires at least 2 body paths")

        spec = asset.spec.copy()
        primary_path = self.body_paths[0]
        primary_body = spec.body(primary_path)
        if primary_body is None:
            raise ValueError(f"MergeBodies: body {primary_path!r} not found")

        parent_body = primary_body.parent
        primary_pos = np.asarray(primary_body.pos, dtype=float)
        primary_rot = sRot.from_quat(primary_body.quat, scalar_first=True)
        inertial_entries = [_body_inertial_in_parent(primary_body, primary_pos, primary_rot)]

        for secondary_path in self.body_paths[1:]:
            secondary_body = spec.body(secondary_path)
            if secondary_body is None:
                raise ValueError(f"MergeBodies: body {secondary_path!r} not found")
            if secondary_body.parent is not parent_body:
                raise ValueError(
                    f"MergeBodies: bodies {primary_path!r} and {secondary_path!r} do not share the same parent"
                )
            if secondary_body.joints:
                raise ValueError(
                    f"MergeBodies: body {secondary_path!r} is not fixed (has {len(secondary_body.joints)} joint(s)); "
                    "only bodies without joints are merged."
                )

            secondary_pos = np.asarray(secondary_body.pos, dtype=float)
            secondary_rot = sRot.from_quat(secondary_body.quat, scalar_first=True)

            inertial_entries.append(
                _body_inertial_in_parent(secondary_body, secondary_pos, secondary_rot)
            )

            for geom in secondary_body.geoms:
                geom_rot = sRot.from_quat(geom.quat, scalar_first=True)
                geom_pos_parent = secondary_pos + secondary_rot.apply(geom.pos)
                geom_pos_primary = primary_rot.inv().apply(geom_pos_parent - primary_pos)
                geom_rot_primary = primary_rot.inv() * secondary_rot * geom_rot
                _copy_geom_to_body(
                    geom,
                    primary_body,
                    tuple(geom_pos_primary.tolist()),
                    tuple(geom_rot_primary.as_quat(scalar_first=True).tolist()),
                )

            child_bodies = list(secondary_body.bodies)
            spec.delete(secondary_body)
            tmp = mujoco.MjSpec()
            tmp.copy_during_attach = True
            tmp_frame = tmp.worldbody.add_frame()

            for body in child_bodies:
                body_pos_parent = secondary_pos + secondary_rot.apply(body.pos)
                body_rot_parent = secondary_rot * sRot.from_quat(body.quat, scalar_first=True)
                body_pos_primary = primary_rot.inv().apply(body_pos_parent - primary_pos)
                body_rot_primary = primary_rot.inv() * body_rot_parent

                moved_body = tmp_frame.attach_body(body)
                for mesh in list(tmp.meshes):
                    tmp.delete(mesh)

                moved_body = primary_body.add_frame().attach_body(moved_body)
                moved_body.pos = tuple(body_pos_primary.tolist())
                moved_body.quat = tuple(body_rot_primary.as_quat(scalar_first=True).tolist())

        total_mass, com_parent, inertia_parent = _combine_inertials_in_parent(inertial_entries)
        _set_body_inertial_in_frame(
            primary_body, total_mass, com_parent, inertia_parent, primary_pos, primary_rot
        )
        return replace(asset, spec=spec)


class MergeBodiesParentChild(Transform):
    """
    Merge (immediate) child bodies into a parent body. Useful for merging dummy links.
    Raise an error if any of the child bodies have non-fixed joints.
    """

    def __init__(self, parent_path: str, child_paths: list[str]) -> None:
        self.parent_path = parent_path
        self.child_paths = child_paths

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        if not self.child_paths:
            raise ValueError("MergeBodiesParentChild requires at least 1 child path")

        spec = asset.spec.copy()

        parent_body = spec.body(self.parent_path)
        if parent_body is None:
            raise ValueError(f"MergeBodiesParentChild: body {self.parent_path!r} not found")

        inertial_entries = [
            _body_inertial_in_parent(
                parent_body, np.zeros(3, dtype=float), sRot.identity()
            )
        ]
        tmp = mujoco.MjSpec()
        tmp.copy_during_attach = True
        tmp_frame = tmp.worldbody.add_frame()

        for child_path in self.child_paths:
            child_body = spec.body(child_path)
            if child_body is None:
                raise ValueError(f"MergeBodiesParentChild: body {child_path!r} not found")
            if child_body.parent is not parent_body:
                raise ValueError(
                    f"MergeBodiesParentChild: body {child_path!r} is not an immediate child of {self.parent_path!r}"
                )
            joints = child_body.joints
            if len(joints):
                raise ValueError(
                    f"MergeBodiesParentChild: body {child_path!r} is not fixed (has {len(joints)} non-fixed joint(s)); "
                    "only fixed children can be merged."
                )

            child_pos = np.asarray(child_body.pos, dtype=float)
            child_rot = sRot.from_quat(child_body.quat, scalar_first=True)
            inertial_entries.append(_body_inertial_in_parent(child_body, child_pos, child_rot))

            for geom in child_body.geoms:
                geom_pos = child_pos + child_rot.apply(np.asarray(geom.pos, dtype=float))
                geom_rot = child_rot * sRot.from_quat(geom.quat, scalar_first=True)
                _copy_geom_to_body(
                    geom,
                    parent_body,
                    tuple(geom_pos.tolist()),
                    tuple(geom_rot.as_quat(scalar_first=True).tolist()),
                )

            child_bodies = list(child_body.bodies)
            spec.delete(child_body)
            for body in child_bodies:
                body_pos_parent = child_pos + child_rot.apply(np.asarray(body.pos, dtype=float))
                body_rot_parent = child_rot * sRot.from_quat(body.quat, scalar_first=True)

                moved_body = tmp_frame.attach_body(body)
                for mesh in list(tmp.meshes):
                    tmp.delete(mesh)

                moved_body = parent_body.add_frame().attach_body(moved_body)
                moved_body.pos = tuple(body_pos_parent.tolist())
                moved_body.quat = tuple(body_rot_parent.as_quat(scalar_first=True).tolist())

        total_mass, com_parent, inertia_parent = _combine_inertials_in_parent(inertial_entries)
        _set_body_inertial_in_frame(
            parent_body,
            total_mass,
            com_parent,
            inertia_parent,
            np.zeros(3, dtype=float),
            sRot.identity(),
        )

        return replace(asset, spec=spec)


class MergeSubtree(Transform):
    """Merge a subtree into the subtree root. Useful for merging dummy links.
    Raise an error if the subtree has non-fixed joints.
    """

    def __init__(self, subtree_path: str) -> None:
        self.subtree_path = subtree_path

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        subtree_root = spec.body(self.subtree_path)
        if subtree_root is None:
            raise ValueError(f"MergeSubtree: body {self.subtree_path!r} not found")

        root_pos = np.asarray(subtree_root.pos, dtype=float)
        root_rot = sRot.from_quat(subtree_root.quat, scalar_first=True)
        root_rot_inv = root_rot.inv()
        inertial_entries = [
            _body_inertial_in_parent(subtree_root, np.zeros(3, dtype=float), sRot.identity())
        ]

        descendants = _iter_subtree_bodies_with_pose_in_parent(subtree_root)[1:]
        descendants_to_delete: list[mujoco.MjsBody] = []
        for body, body_pos_parent, body_rot_parent in descendants:
            joints = body.joints
            if len(joints):
                raise ValueError(
                    f"MergeSubtree: body {body.name!r} in subtree {self.subtree_path!r} is not fixed "
                    f"(has {len(joints)} joint(s)); only fixed descendants can be merged."
                )

            body_pos_root = root_rot_inv.apply(body_pos_parent - root_pos)
            body_rot_root = root_rot_inv * body_rot_parent
            inertial_entries.append(
                _body_inertial_in_parent(body, body_pos_root, body_rot_root)
            )

            for geom in body.geoms:
                geom_pos = body_pos_root + body_rot_root.apply(np.asarray(geom.pos, dtype=float))
                geom_rot = body_rot_root * sRot.from_quat(geom.quat, scalar_first=True)
                _copy_geom_to_body(
                    geom,
                    subtree_root,
                    tuple(geom_pos.tolist()),
                    tuple(geom_rot.as_quat(scalar_first=True).tolist()),
                )
            descendants_to_delete.append(body)

        for body in reversed(descendants_to_delete):
            spec.delete(body)

        total_mass, com_root, inertia_root = _combine_inertials_in_parent(inertial_entries)
        _set_body_inertial_in_frame(
            subtree_root,
            total_mass,
            com_root,
            inertia_root,
            np.zeros(3, dtype=float),
            sRot.identity(),
        )
        return replace(asset, spec=spec)


class RemoveSubtrees(Transform):
    def __init__(self, subtree_paths: list[str]) -> None:
        self.subtree_paths = subtree_paths

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for subtree_path in self.subtree_paths:
            subtree = spec.body(subtree_path)
            if subtree is None:
                raise ValueError(f"RemoveSubtrees: body {subtree_path!r} not found")
            spec.delete(subtree)
        return replace(asset, spec=spec)


class SelectSubtree(Transform):
    def __init__(self, subtree_path: str) -> None:
        self.subtree_path = subtree_path

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        subtree = asset.spec.body(self.subtree_path)
        if subtree is None:
            raise ValueError(f"SelectSubtree: body {self.subtree_path!r} not found")

        spec = mujoco.MjSpec()
        spec.copy_during_attach = True
        frame = spec.worldbody.add_frame()
        frame.attach_body(subtree)
        spec.compile()
        return replace(asset, spec=spec)


class GeomsToBody(Transform):
    """Move geoms onto a new fixed child body and rename them.

    All ``geom_names`` must currently share the same parent. The new body is
    placed at the first geom's pose; each geom is re-expressed relative to that
    frame (so a single foot sphere at ``pos=(0,0,-0.275)`` becomes a body at
    that pose with the geom at the origin — matching aa-robot-models A2 feet).

    Moved geoms are renamed with the same rule as :class:`NormalizeGeomNames`
    (``{body}_visual{i}`` / ``{body}_collision{i}``).

    Pass ``mass`` (and optionally ``inertia``) to give the new body an explicit
    inertial — useful when moved geoms are visual-only / zero-density and a
    massless body would break downstream simulators. When ``mass`` is set,
    moved geoms have their density/mass cleared so the explicit inertial is
    authoritative.

    Example
    -------
    ::

        GeomsToBody(["FL_calf_collision4"], "FL_foot", mass=0.05)
    """

    def __init__(
        self,
        geom_names: list[str],
        body_name: str,
        *,
        mass: float | None = None,
        inertia: tuple[float, float, float] | None = None,
        ipos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not geom_names:
            raise ValueError("GeomsToBody requires at least one geom name")
        if not body_name:
            raise ValueError("GeomsToBody requires a non-empty body_name")
        if mass is not None and mass < 0:
            raise ValueError(f"GeomsToBody: mass must be >= 0, got {mass}")
        if inertia is not None:
            if len(inertia) != 3:
                raise ValueError("GeomsToBody: inertia must be a length-3 diaginertia")
            if any(x < 0 for x in inertia):
                raise ValueError(f"GeomsToBody: inertia components must be >= 0, got {inertia}")
            if mass is None:
                raise ValueError("GeomsToBody: inertia requires mass to be set")
        self.geom_names = list(geom_names)
        self.body_name = body_name
        self.mass = mass
        self.inertia = inertia
        self.ipos = ipos

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        if spec.body(self.body_name) is not None:
            raise ValueError(f"GeomsToBody: body {self.body_name!r} already exists")

        geoms: list[mujoco.MjsGeom] = []
        for name in self.geom_names:
            geom = spec.geom(name)
            if geom is None:
                raise ValueError(f"GeomsToBody: geom {name!r} not found")
            geoms.append(geom)

        parent = geoms[0].parent
        if parent is None:
            raise ValueError(
                f"GeomsToBody: geom {self.geom_names[0]!r} has no parent body"
            )
        for geom, name in zip(geoms, self.geom_names):
            if geom.parent is not parent:
                raise ValueError(
                    f"GeomsToBody: geom {name!r} parent {geom.parent.name!r} "
                    f"differs from {parent.name!r}; all geoms must share a parent"
                )

        body_pos = np.asarray(geoms[0].pos, dtype=float)
        body_rot = sRot.from_quat(geoms[0].quat, scalar_first=True)
        child = parent.add_body()
        child.name = self.body_name
        child.pos = tuple(float(x) for x in body_pos)
        child.quat = tuple(float(x) for x in body_rot.as_quat(scalar_first=True))

        used_names = {g.name for g in spec.geoms if g.name}
        # Reserve names that will be freed when source geoms are deleted.
        for geom in geoms:
            if geom.name:
                used_names.discard(geom.name)

        for geom in geoms:
            geom_pos = np.asarray(geom.pos, dtype=float)
            geom_rot = sRot.from_quat(geom.quat, scalar_first=True)
            rel_pos = body_rot.inv().apply(geom_pos - body_pos)
            rel_quat = body_rot.inv() * geom_rot
            new_geom = _copy_geom_to_body(
                geom,
                child,
                tuple(float(x) for x in rel_pos),
                tuple(float(x) for x in rel_quat.as_quat(scalar_first=True)),
                name="",
            )
            if self.mass is not None:
                # Explicit body inertial owns the mass; keep geoms collision/visual only.
                new_geom.density = 0
                new_geom.mass = 0
            spec.delete(geom)

        if self.mass is not None:
            diag = (
                self.inertia
                if self.inertia is not None
                else (1e-6, 1e-6, 1e-6)
            )
            child.mass = float(self.mass)
            child.ipos = tuple(float(x) for x in self.ipos)
            child.inertia = tuple(float(x) for x in diag)
            child.iquat = (1.0, 0.0, 0.0, 0.0)
            child.explicitinertial = 1

        _assign_normalized_geom_names(child, used_names)
        spec.compile()
        return replace(asset, spec=spec)


class Body2Site(Transform):
    """Convert dummy bodies to sites, preserving its pos and quat.

    Note that Isaac Sim does not convert sites into dummy bodies.

    Raise an error if the body has non-fixed joints or non-trivial inertia or children.
    """

    def __init__(self, body_paths: list[str], mass_threshold: float = 0.001) -> None:
        self.body_paths = body_paths
        self.mass_threshold = mass_threshold

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for body_path in self.body_paths:
            body = spec.body(body_path)
            if body is None:
                raise ValueError(f"Body2Site: body {body_path!r} not found")
            if body.parent is None:
                raise ValueError(
                    f"Body2Site: body {body_path!r} has no parent and cannot be converted to a site"
                )
            joints = body.joints
            if len(joints):
                raise ValueError(
                    f"Body2Site: body {body_path!r} is not fixed (has {len(joints)} joint(s)); "
                    "only fixed dummy bodies can be converted."
                )
            if len(body.bodies):
                raise ValueError(
                    f"Body2Site: body {body_path!r} has {len(body.bodies)} child bod(ies); "
                    "only leaf bodies can be converted."
                )
            if len(body.geoms):
                raise ValueError(
                    f"Body2Site: body {body_path!r} has {len(body.geoms)} geom(s); "
                    "only dummy bodies without geoms can be converted."
                )
            if float(body.mass) > self.mass_threshold:
                raise ValueError(
                    f"Body2Site: body {body_path!r} has mass {float(body.mass):.6g}, "
                    f"which is larger than threshold {self.mass_threshold:.6g}"
                )
            if float(np.max(np.abs(np.asarray(body.inertia, dtype=float)))) > self.mass_threshold:
                raise ValueError(
                    f"Body2Site: body {body_path!r} has non-trivial inertia {tuple(body.inertia)}; "
                    f"max component exceeds threshold {self.mass_threshold:.6g}"
                )

            site = body.parent.add_site()
            site.name = body.name
            site.pos = body.pos
            site.quat = body.quat
            spec.delete(body)

        spec.compile()
        return replace(asset, spec=spec)
