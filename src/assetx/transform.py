from __future__ import annotations

import mujoco
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation as sRot
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List
import shutil


@dataclass
class MujocoAsset:
    xml_path: str
    spec: mujoco.MjSpec
    meshdir: Path

    @staticmethod
    def from_file(xml_path: str) -> MujocoAsset:
        spec = mujoco.MjSpec.from_file(xml_path)
        # resolve the actual meshdir from the spec
        meshdir = Path(xml_path).parent / spec.meshdir
        if not meshdir.exists():
            raise FileNotFoundError(f"Meshdir {meshdir} not found")
        return MujocoAsset(xml_path, spec, meshdir)
    
    def save(self, path: str | Path) -> MujocoAsset:
        """Save the asset to a directory.

        Writes model.xml and the mesh directory under the given path.
        The directory must not already exist.

        Args:
            path: Output directory path (not an XML file path). Created if missing.

        Returns:
            A new MujocoAsset loaded from the saved files.

        Raises:
            ValueError: If path exists and is a file (path must be a directory).
        """
        root = Path(path)
        if root.exists() and root.is_file():
            raise ValueError(
                f"path must be a directory, not a file: {path!r}. "
                "Pass the output directory where model.xml and meshes will be written."
            )
        root.mkdir(parents=True, exist_ok=False)
        self.spec.compile()
        self.spec.to_file(str(root / "model.xml"))
        # copy the meshdir to the new file
        dest = root / self.spec.meshdir
        shutil.copytree(self.meshdir, dest, symlinks=True)
        return MujocoAsset.from_file(str(root / "model.xml"))


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


class RemoveGeoms(Transform):
    def __init__(self, geom_names: list[str]) -> None:
        self.geom_names = geom_names

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            geom: mujoco.MjsGeom
            if geom.name in self.geom_names:
                spec.delete(geom)
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
        return replace(asset, spec=spec)

