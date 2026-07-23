from __future__ import annotations

import tempfile
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


def _mesh_source_dir(asset: MujocoAsset) -> Path:
    """Directory that actually contains ``asset``'s mesh files.

    Prefer the common parent of absolute mesh paths (URDF→MJCF often leaves
    ``meshdir`` empty while ``file`` is absolute). Fall back to
    ``resolved_meshdir``.
    """
    parents = {_mesh_file_path(asset, mesh.file).parent for mesh in asset.spec.meshes}
    if len(parents) == 1:
        return parents.pop()
    if not parents:
        return asset.resolved_meshdir
    raise ValueError(
        f"Meshes for {asset.spec.modelname!r} span multiple directories: "
        f"{sorted(str(p) for p in parents)}. Put them under one folder or set meshdir."
    )


def assemble(
    parent: MujocoAsset,
    child: MujocoAsset,
    parent_link: str,
    child_prefix: str = "child_",
    translation: tuple[float, float, float] = (0, 0, 0),
    rotation: tuple[float, float, float] = (0, 0, 0),
    joint_cfg: JointCfg | None = None,
) -> MujocoAsset:
    spec = parent.spec.copy()
    parent_meshes = [mesh.name for mesh in spec.meshes]
    child_meshes = [f"{child_prefix}{mesh.name}" for mesh in child.spec.meshes]

    child_root = child.spec.worldbody.first_body()
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

    # Own the temp dir for the lifetime of the returned asset (until GC after save).
    tmp = tempfile.TemporaryDirectory(prefix="assetx-assemble-")
    tmp_dir = Path(tmp.name)
    tmp_xml_path = tmp_dir / "assembled.xml"
    meshdir = tmp_dir / "meshes"
    meshdir.mkdir(parents=True, exist_ok=True)

    parent_mesh_dir = _mesh_source_dir(parent)
    child_mesh_dir = _mesh_source_dir(child)

    for name in parent_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        mesh.file = str(_mesh_file_path(parent, mesh.file))

    for name in child_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        # After attach, mesh.file still refers to the child's original path.
        mesh.file = str(_mesh_file_path(child, mesh.file))

    spec.compile()
    spec.to_file(str(tmp_xml_path))

    spec = mujoco.MjSpec.from_file(str(tmp_xml_path))
    resolved_meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
    # Nested ``<modelname>/file.stl`` under meshdir is valid; symlink each
    # asset's real mesh folder there (not necessarily ``resolved_meshdir``).
    (resolved_meshdir / parent.spec.modelname).symlink_to(parent_mesh_dir)
    (resolved_meshdir / child.spec.modelname).symlink_to(child_mesh_dir)

    for name in parent_meshes:
        mesh = spec.mesh(name)
        mesh.file = str(Path(parent.spec.modelname) / Path(mesh.file).name)

    for name in child_meshes:
        mesh = spec.mesh(name)
        mesh.file = str(Path(child.spec.modelname) / Path(mesh.file).name)

    spec.compile()
    final_xml_path = tmp_dir / "model.xml"
    spec.to_file(str(final_xml_path))
    return MujocoAsset(final_xml_path, spec, Path(spec.meshdir), _tmpdir=tmp)
