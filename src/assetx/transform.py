from __future__ import annotations

import mujoco
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as sRot
from dataclasses import replace
from assetx.common import MujocoAsset, JointCfg


class Transform(ABC):
    @abstractmethod
    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        pass


class Compose(Transform):
    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        for transform in self.transforms:
            asset = transform.transform(asset)
        return asset


class ReplaceCylinderWithCapsule(Transform):
    """Replace every cylinder geom with a capsule (same size: radius, half-length)."""

    def __init__(self) -> None:
        pass

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            geom: mujoco.MjsGeom
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        return replace(asset, spec=spec)


class MergeBodies(Transform):
    """
    Merge bodies connected by a fixed joint: move the child's contents into the
    parent and remove the fixed joint and the child body.
    """

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
            geom: mujoco.MjsGeom
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

        spec.delete(child_body)
        tmp = mujoco.MjSpec()
        tmp.copy_during_attach = True
        tmp_frame = tmp.worldbody.add_frame()

        for body in child_body.bodies:
            body: mujoco.MjsBody
            body_pos = body.pos
            body_rot = sRot.from_quat(body.quat, scalar_first=True)
            body = tmp_frame.attach_body(body)
            for mesh in tmp.meshes:
                tmp.delete(mesh)
            
            body = parent_body.add_frame().attach_body(body)
            body.pos = child_pos + child_rot.apply(body_pos)
            body.quat = (child_rot * body_rot).as_quat(scalar_first=True)

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
        for joint in spec.joints:
            joint: mujoco.MjsJoint
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


class SelectSubtree(Transform):
    def __init__(self, subtree_path: str) -> None:
        self.subtree_path = subtree_path

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = mujoco.MjSpec()
        spec.copy_during_attach = True
        frame = spec.worldbody.add_frame()
        subtree = spec.body(self.subtree_path)
        if subtree is None:
            raise ValueError(f"SelectSubtree: body {self.subtree_path!r} not found")
        frame.attach_body(subtree)
        return replace(asset, spec=spec)


class RemoveGeoms(Transform):
    def __init__(self, geom_names: list[str]) -> None:
        self.geom_names = geom_names

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            geom: mujoco.MjsGeom
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
            body: mujoco.MjsBody
            if body.name in self.body_names:
                body.name = self.body_names[body.name]
        spec.compile()
        return replace(asset, spec=spec)

