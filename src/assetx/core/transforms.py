from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import mujoco
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import JointCfg, MujocoAsset


class Transform(ABC):
    @abstractmethod
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        for transform in self.transforms:
            asset = transform.transform(asset)
        return asset


class ReplaceCylinderWithCapsule(Transform):
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        return replace(asset, spec=spec)


class MergeBodies(Transform):
    def __init__(self, parent_path: str, child_path: str) -> None:
        self.parent_path = parent_path
        self.child_path = child_path

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()

        parent_body = spec.body(self.parent_path)
        child_body = spec.body(self.child_path)
        joints = child_body.joints
        if len(joints):
            raise ValueError(
                f"MergeBodies: body {self.child_path!r} is not fixed (has {len(joints)} non-fixed joint(s)); "
                "only bodies connected by a fixed joint are merged."
            )

        child_pos = child_body.pos
        child_rot = sRot.from_quat(child_body.quat, scalar_first=True)

        for geom in child_body.geoms:
            geom_pos = geom.pos
            geom_rot = sRot.from_quat(geom.quat, scalar_first=True)

            new_geom = parent_body.add_geom()
            new_geom.type = geom.type
            new_geom.size = geom.size
            new_geom.pos = child_pos + child_rot.apply(geom_pos)
            new_geom.quat = (child_rot * geom_rot).as_quat(scalar_first=True)
            new_geom.rgba = geom.rgba
            new_geom.name = geom.name
            new_geom.contype = geom.contype
            new_geom.conaffinity = geom.conaffinity
            new_geom.mass = geom.mass
            new_geom.friction = geom.friction
            new_geom.condim = geom.condim
            new_geom.meshname = geom.meshname

        child_bodies = list(child_body.bodies)
        spec.delete(child_body)
        tmp = mujoco.MjSpec()
        tmp.copy_during_attach = True
        tmp_frame = tmp.worldbody.add_frame()

        for body in child_bodies:
            body_pos = body.pos
            body_rot = sRot.from_quat(body.quat, scalar_first=True)
            moved_body = tmp_frame.attach_body(body)
            for mesh in list(tmp.meshes):
                tmp.delete(mesh)

            moved_body = parent_body.add_frame().attach_body(moved_body)
            moved_body.pos = child_pos + child_rot.apply(body_pos)
            moved_body.quat = (child_rot * body_rot).as_quat(scalar_first=True)

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
    def __init__(self, body_path: str, joint_cfg: JointCfg) -> None:
        self.body_path = body_path
        self.joint_cfg = joint_cfg

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        body = spec.body(self.body_path)
        if body is None:
            raise ValueError(f"AddJoint: body {self.body_path!r} not found")
        joint = body.add_joint()
        joint.name = self.joint_cfg.name
        joint.type = self.joint_cfg.type
        joint.axis = self.joint_cfg.axis
        joint.range = self.joint_cfg.range
        spec.compile()
        return replace(asset, spec=spec)


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
        if type not in {"sphere", "capsule", "ellipsoid", "cylinder", "box"}:
            raise ValueError(f"Invalid site type: {type}")
        self.body_path = body_path
        self.name = name
        self.pos = pos
        self.quat = quat
        self.size = size
        self.type = type
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


def apply_transforms(asset: MujocoAsset, *transforms: Transform) -> MujocoAsset:
    return Compose(list(transforms)).transform(asset)
