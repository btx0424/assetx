from __future__ import annotations

from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import JointCfg, MujocoAsset


def _mesh_file_path(asset: MujocoAsset, mesh_file: str) -> Path:
    """Resolve a mesh ``file`` attribute to an absolute filesystem path."""
    path = Path(mesh_file)
    if path.is_absolute():
        return path.resolve()
    return (asset.resolved_meshdir / path).resolve()


def _absolutize_mesh_files(asset: MujocoAsset, spec: mujoco.MjSpec) -> None:
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str(_mesh_file_path(asset, mesh.file))


def assemble(
    parent: MujocoAsset,
    child: MujocoAsset,
    parent_link: str,
    child_prefix: str = "child_",
    translation: tuple[float, float, float] = (0, 0, 0),
    rotation: tuple[float, float, float] = (0, 0, 0),
    joint_cfg: JointCfg | None = None,
) -> MujocoAsset:
    """Attach ``child`` under ``parent_link`` and return an in-memory asset.

    Mesh ``file`` attributes are left as **absolute** paths into the parent /
    child source trees (no temp staging, no symlinks). Call
    :meth:`MujocoAsset.save` to write a durable artifact with relative mesh
    paths copied into the output directory.
    """
    # Work on copies so caller assets stay unchanged. Absolutize mesh paths
    # *before* attach: MuJoCo may auto-name unnamed meshes on attach, so
    # reconstructing ``{prefix}{old_name}`` is unreliable.
    spec = parent.spec.copy()
    child_spec = child.spec.copy()
    _absolutize_mesh_files(parent, spec)
    _absolutize_mesh_files(child, child_spec)

    child_root = child_spec.worldbody.first_body()
    frame = spec.body(parent_link).add_frame()
    frame.pos = translation
    frame.quat = sRot.from_euler("xyz", rotation).as_quat(scalar_first=True)
    attached_root = frame.attach_body(child_root, child_prefix)
    cfg = joint_cfg or JointCfg(type="fixed")
    if cfg.type != "fixed":
        joint = attached_root.add_joint()
        if cfg.name:
            joint.name = cfg.name
        joint_type_map = {
            "hinge": mujoco.mjtJoint.mjJNT_HINGE,
            "slide": mujoco.mjtJoint.mjJNT_SLIDE,
            "free": mujoco.mjtJoint.mjJNT_FREE,
        }
        joint.type = joint_type_map[cfg.type]
        if cfg.type in {"hinge", "slide"}:
            joint.axis = cfg.axis
            joint.limited = cfg.limited
            if cfg.limited:
                joint.range = cfg.range

    # Ensure every mesh path is still absolute after attach.
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        path = Path(mesh.file)
        if not path.is_absolute():
            # Prefer parent meshdir, then child (attach should not rewrite these).
            for base in (parent.resolved_meshdir, child.resolved_meshdir):
                candidate = (base / path).resolve()
                if candidate.is_file():
                    mesh.file = str(candidate)
                    break

    spec.compile()
    # xml_path is provenance only; mesh files on the spec are absolute.
    # meshdir="meshes" is the conventional relative layout used by save().
    return MujocoAsset(parent.xml_path, spec, Path("meshes"))
