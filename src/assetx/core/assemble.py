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


def _absolutize_mesh_files(asset: MujocoAsset, spec: mujoco.MjSpec) -> None:
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str(_mesh_file_path(asset, mesh.file))


def _mesh_source_dir(asset: MujocoAsset, spec: mujoco.MjSpec | None = None) -> Path:
    """Directory that actually contains ``asset``'s mesh files.

    Prefer the common parent of absolute mesh paths (URDF→MJCF often leaves
    ``meshdir`` empty while ``file`` is absolute). Fall back to
    ``resolved_meshdir``.
    """
    use_spec = spec if spec is not None else asset.spec
    parents: set[Path] = set()
    for mesh in use_spec.meshes:
        if not mesh.file:
            continue
        parents.add(_mesh_file_path(asset, mesh.file).parent)
    if len(parents) == 1:
        return parents.pop()
    if not parents:
        return asset.resolved_meshdir
    raise ValueError(
        f"Meshes for {asset.spec.modelname!r} span multiple directories: "
        f"{sorted(str(p) for p in parents)}. Put them under one folder or set meshdir."
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assemble(
    parent: MujocoAsset,
    child: MujocoAsset,
    parent_link: str,
    child_prefix: str = "child_",
    translation: tuple[float, float, float] = (0, 0, 0),
    rotation: tuple[float, float, float] = (0, 0, 0),
    joint_cfg: JointCfg | None = None,
) -> MujocoAsset:
    # Work on copies so caller assets stay unchanged. Absolutize mesh paths
    # *before* attach: MuJoCo may auto-name unnamed meshes on attach, so
    # reconstructing ``{prefix}{old_name}`` is unreliable.
    spec = parent.spec.copy()
    child_spec = child.spec.copy()
    _absolutize_mesh_files(parent, spec)
    _absolutize_mesh_files(child, child_spec)
    parent_mesh_dir = _mesh_source_dir(parent, spec)
    child_mesh_dir = _mesh_source_dir(child, child_spec)
    parent_model = parent.spec.modelname or "parent"
    child_model = child.spec.modelname or "child"

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

    # Own the temp dir for the lifetime of the returned asset (until GC after save).
    tmp = tempfile.TemporaryDirectory(prefix="assetx-assemble-")
    tmp_dir = Path(tmp.name)
    tmp_xml_path = tmp_dir / "assembled.xml"
    meshdir = tmp_dir / "meshes"
    meshdir.mkdir(parents=True, exist_ok=True)

    spec.compile()
    spec.to_file(str(tmp_xml_path))

    spec = mujoco.MjSpec.from_file(str(tmp_xml_path))
    resolved_meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
    # Nested ``<modelname>/file.stl`` under meshdir is valid; symlink each
    # asset's real mesh folder there (not necessarily ``resolved_meshdir``).
    (resolved_meshdir / parent_model).symlink_to(parent_mesh_dir)
    (resolved_meshdir / child_model).symlink_to(child_mesh_dir)

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        src = Path(mesh.file)
        if not src.is_absolute():
            src = (Path(spec.modelfiledir) / src).resolve()
        if _is_relative_to(src, parent_mesh_dir):
            mesh.file = str(Path(parent_model) / src.name)
        elif _is_relative_to(src, child_mesh_dir):
            mesh.file = str(Path(child_model) / src.name)

    spec.compile()
    final_xml_path = tmp_dir / "model.xml"
    spec.to_file(str(final_xml_path))
    return MujocoAsset(final_xml_path, spec, Path(spec.meshdir), _tmpdir=tmp)
