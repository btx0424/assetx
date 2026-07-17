from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation as sRot

from assetx.core.asset import JointCfg, MujocoAsset


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

    for name in parent_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        mesh.file = str(parent.resolved_meshdir / mesh.file)

    for name in child_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        mesh.file = str(child.resolved_meshdir / mesh.file)

    spec.compile()
    spec.to_file(str(tmp_xml_path))

    spec = mujoco.MjSpec.from_file(str(tmp_xml_path))
    resolved_meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
    (resolved_meshdir / parent.spec.modelname).symlink_to(parent.resolved_meshdir)
    (resolved_meshdir / child.spec.modelname).symlink_to(child.resolved_meshdir)

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
