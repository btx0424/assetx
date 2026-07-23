"""Convert a single-robot Isaac Lab USD asset to MJCF.

Isaac Lab robot USDs store rigid bodies as sibling Xforms under the robot root
and encode the kinematic tree via UsdPhysics joints (body0 = parent, body1 = child).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from pxr import Sdf, Usd, UsdPhysics
from scipy.spatial.transform import Rotation as sRot

from assetx.conversion.usd.geoms import (
    BodyGeom,
    export_meshes,
    extract_body_geoms,
    gf_quat_to_wxyz,
)


@dataclass
class JointInfo:
    prim: Usd.Prim
    name: str
    type_name: str
    parent: Sdf.Path
    child: Sdf.Path


@dataclass
class KinematicTree:
    """Kinematic tree inferred from UsdPhysics joints."""

    bodies: dict[Sdf.Path, Usd.Prim]
    joints: list[JointInfo]
    parent_of: dict[Sdf.Path, Sdf.Path]
    joint_to_child: dict[Sdf.Path, JointInfo]
    children_of: dict[Sdf.Path, list[Sdf.Path]] = field(default_factory=dict)
    root: Sdf.Path | None = None

    def __post_init__(self) -> None:
        children: dict[Sdf.Path, list[Sdf.Path]] = defaultdict(list)
        for child, parent in self.parent_of.items():
            children[parent].append(child)
        child_order = {j.child: i for i, j in enumerate(self.joints)}
        for parent in children:
            children[parent].sort(key=lambda p: child_order.get(p, 10**9))
        self.children_of = dict(children)

        roots = set(self.bodies) - set(self.parent_of)
        if len(roots) != 1:
            raise ValueError(
                f"Expected exactly one kinematic root, found {len(roots)}: "
                f"{sorted(str(p) for p in roots)}"
            )
        self.root = roots.pop()

    def format(self) -> str:
        """Pretty-print the kinematic tree."""
        assert self.root is not None
        lines: list[str] = [
            f"root: {self.root.name} ({self.root})",
            f"bodies: {len(self.bodies)}  joints: {len(self.joints)}",
            "",
        ]

        def walk(path: Sdf.Path, prefix: str = "", is_last: bool = True) -> None:
            connector = "└── " if is_last else "├── "
            if path == self.root:
                lines.append(f"{path.name}")
            else:
                joint = self.joint_to_child[path]
                lines.append(
                    f"{prefix}{connector}{path.name}  "
                    f"← {joint.name} [{joint.type_name}]"
                )
            kids = self.children_of.get(path, [])
            child_prefix = (
                "" if path == self.root else prefix + ("    " if is_last else "│   ")
            )
            for i, child in enumerate(kids):
                walk(child, child_prefix, i == len(kids) - 1)

        walk(self.root)
        return "\n".join(lines)


def load_usd(path: str | Path) -> Usd.Stage:
    """Step 1: open a USD stage and require a default prim (robot root)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {path}")

    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"USD stage has no default prim: {path}")

    return stage


def collect_bodies_and_joints(
    stage: Usd.Stage,
) -> tuple[dict[Sdf.Path, Usd.Prim], list[Usd.Prim]]:
    """Collect RigidBodyAPI prims and UsdPhysics *Joint prims under the default prim."""
    root = stage.GetDefaultPrim()
    bodies: dict[Sdf.Path, Usd.Prim] = {}
    joints: list[Usd.Prim] = []

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetPath()] = prim
        if "Joint" in prim.GetTypeName():
            joints.append(prim)

    return bodies, joints


def build_kinematic_tree(stage: Usd.Stage) -> KinematicTree:
    """Step 2: build the kinematic tree from joint body0/body1 relationships."""
    bodies, joint_prims = collect_bodies_and_joints(stage)

    joints: list[JointInfo] = []
    parent_of: dict[Sdf.Path, Sdf.Path] = {}
    joint_to_child: dict[Sdf.Path, JointInfo] = {}

    for joint_prim in joint_prims:
        rel0 = joint_prim.GetRelationship("physics:body0")
        rel1 = joint_prim.GetRelationship("physics:body1")
        if not rel0 or not rel1:
            continue
        targets0 = rel0.GetTargets()
        targets1 = rel1.GetTargets()
        if not targets0 or not targets1:
            continue

        parent, child = targets0[0], targets1[0]
        if parent not in bodies or child not in bodies:
            raise ValueError(
                f"Joint {joint_prim.GetPath()} references unknown body: "
                f"body0={parent}, body1={child}"
            )
        if child in parent_of:
            raise ValueError(
                f"Body {child} has multiple parents: "
                f"{parent_of[child]} and {parent} (via {joint_prim.GetPath()})"
            )

        info = JointInfo(
            prim=joint_prim,
            name=joint_prim.GetName(),
            type_name=joint_prim.GetTypeName(),
            parent=parent,
            child=child,
        )
        joints.append(info)
        parent_of[child] = parent
        joint_to_child[child] = info

    return KinematicTree(
        bodies=bodies,
        joints=joints,
        parent_of=parent_of,
        joint_to_child=joint_to_child,
    )


def extract_robot_geoms(
    tree: KinematicTree,
) -> dict[Sdf.Path, list[BodyGeom]]:
    """Step 2.5: extract visual/collision geoms for every rigid body."""
    return {path: extract_body_geoms(prim) for path, prim in tree.bodies.items()}


def _joint_frame(joint: Usd.Prim, which: int) -> np.ndarray:
    """Return the 4x4 joint local frame (body0 or body1) as a column-vector matrix."""
    pos_attr = joint.GetAttribute(f"physics:localPos{which}")
    rot_attr = joint.GetAttribute(f"physics:localRot{which}")
    pos = np.zeros(3, dtype=np.float64)
    if pos_attr and pos_attr.Get() is not None:
        pos = np.asarray(pos_attr.Get(), dtype=np.float64)

    if rot_attr and rot_attr.Get() is not None:
        quat_wxyz = gf_quat_to_wxyz(rot_attr.Get())
    else:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    matrix = np.eye(4)
    matrix[:3, :3] = sRot.from_quat(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    ).as_matrix()
    matrix[:3, 3] = pos
    return matrix


def child_pose_from_joint(joint: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Body1 pose in body0 frame from joint local frames: T0 @ inv(T1)."""
    rel = _joint_frame(joint, 0) @ np.linalg.inv(_joint_frame(joint, 1))
    pos = rel[:3, 3]
    quat_xyzw = sRot.from_matrix(rel[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def _axis_vector(axis_token: str | None) -> np.ndarray:
    mapping = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    if not axis_token:
        return mapping["Z"]
    return mapping.get(str(axis_token).upper(), mapping["Z"])


def _set_body_inertial(mj_body: mujoco.MjsBody, usd_body: Usd.Prim) -> None:
    mj_body.explicitinertial = True
    if not usd_body.HasAPI(UsdPhysics.MassAPI):
        mj_body.mass = 1e-3
        mj_body.inertia = np.array([1e-6, 1e-6, 1e-6])
        mj_body.ipos = np.zeros(3)
        mj_body.iquat = np.array([1.0, 0.0, 0.0, 0.0])
        return

    mass_api = UsdPhysics.MassAPI(usd_body)
    mass = mass_api.GetMassAttr().Get()
    com = mass_api.GetCenterOfMassAttr().Get()
    diag = mass_api.GetDiagonalInertiaAttr().Get()
    axes = mass_api.GetPrincipalAxesAttr().Get()

    mj_body.mass = float(mass) if mass is not None else 1e-3
    mj_body.ipos = np.asarray(com, dtype=np.float64) if com is not None else np.zeros(3)
    mj_body.inertia = (
        np.asarray(diag, dtype=np.float64) if diag is not None else np.array([1e-6, 1e-6, 1e-6])
    )
    mj_body.iquat = (
        gf_quat_to_wxyz(axes) if axes is not None else np.array([1.0, 0.0, 0.0, 0.0])
    )


_GEOM_TYPE = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


def _indexed_name(prefix: str, index: int, total: int) -> str:
    """Return ``prefix`` or ``prefix{i}`` when multiple geoms share a role."""
    if total <= 1:
        return prefix
    return f"{prefix}{index}"


def _add_geoms_to_body(
    spec: mujoco.MjSpec,
    mj_body: mujoco.MjsBody,
    geoms: list[BodyGeom],
    *,
    body_name: str,
    mesh_assets: set[str],
    mesh_files: dict[str, str] | None = None,
) -> None:
    has_primitive_collision = any(
        g.kind != "mesh" and g.is_collision for g in geoms
    )
    # Collision mesh geoms are only emitted when the body has no primitives.
    collision_geoms = [
        g
        for g in geoms
        if g.is_collision
        and (g.kind != "mesh" or not has_primitive_collision)
    ]
    visual_mesh_geoms = [g for g in geoms if g.kind == "mesh" and g.is_visual]
    n_collision = len(collision_geoms)
    n_visual_mesh = len(visual_mesh_geoms)

    col_idx = 0
    vis_mesh_idx = 0

    for geom in geoms:
        if geom.kind == "mesh":
            if geom.mesh is None:
                continue
            if geom.name not in mesh_assets:
                if mesh_files and geom.name in mesh_files:
                    spec.add_mesh(name=geom.name, file=mesh_files[geom.name])
                else:
                    asset = spec.add_mesh(name=geom.name)
                    asset.uservert = np.asarray(geom.mesh.vertices, dtype=np.float64).ravel()
                    asset.userface = np.asarray(geom.mesh.faces, dtype=np.int32).ravel()
                mesh_assets.add(geom.name)

            if geom.is_visual:
                g = mj_body.add_geom(
                    name=_indexed_name(f"{body_name}_visual", vis_mesh_idx, n_visual_mesh),
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=geom.name,
                )
                g.contype = 0
                g.conaffinity = 0
                g.group = 1
                g.density = 0
                vis_mesh_idx += 1

            if geom.is_collision and not has_primitive_collision:
                g = mj_body.add_geom(
                    name=_indexed_name(f"{body_name}_collision", col_idx, n_collision),
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=geom.name,
                )
                g.group = 0
                col_idx += 1
            continue

        if not geom.is_collision and not geom.is_visual:
            continue

        if geom.is_collision:
            name = _indexed_name(f"{body_name}_collision", col_idx, n_collision)
            col_idx += 1
        else:
            name = f"{body_name}_visual"

        g = mj_body.add_geom(
            name=name,
            type=_GEOM_TYPE[geom.kind],
            size=geom.size,
            pos=geom.pos,
            quat=geom.quat_wxyz,
        )
        if geom.is_visual and not geom.is_collision:
            g.contype = 0
            g.conaffinity = 0
            g.group = 1
            g.density = 0
        else:
            g.group = 0


def _add_joint(mj_body: mujoco.MjsBody, joint_info: JointInfo) -> None:
    joint = joint_info.prim
    type_name = joint_info.type_name

    if type_name == "PhysicsFixedJoint":
        return

    t1 = _joint_frame(joint, 1)
    axis_token = joint.GetAttribute("physics:axis").Get() if joint.GetAttribute("physics:axis") else "Z"
    axis = t1[:3, :3] @ _axis_vector(axis_token)
    pos = t1[:3, 3]

    lower = joint.GetAttribute("physics:lowerLimit").Get()
    upper = joint.GetAttribute("physics:upperLimit").Get()

    if type_name == "PhysicsRevoluteJoint":
        mj_joint = mj_body.add_joint(
            name=joint_info.name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis,
            pos=pos,
        )
        if lower is not None and upper is not None:
            lo = math.radians(float(lower))
            hi = math.radians(float(upper))
            # Skip ±inf / non-finite bounds — omit range for a truly unlimited hinge.
            if math.isfinite(lo) and math.isfinite(hi):
                mj_joint.range = np.array([lo, hi])
        return

    if type_name == "PhysicsPrismaticJoint":
        mj_joint = mj_body.add_joint(
            name=joint_info.name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            pos=pos,
        )
        if lower is not None and upper is not None:
            lo, hi = float(lower), float(upper)
            if math.isfinite(lo) and math.isfinite(hi):
                mj_joint.range = np.array([lo, hi])
        return

    raise ValueError(f"Unsupported joint type: {type_name} ({joint.GetPath()})")


def build_mjcf(
    tree: KinematicTree,
    body_geoms: dict[Sdf.Path, list[BodyGeom]],
    *,
    model_name: str = "robot",
    freejoint: bool = True,
    meshdir: str = "meshes",
    mesh_files: dict[str, str] | None = None,
) -> mujoco.MjSpec:
    """Step 3: build an MjSpec from the kinematic tree and extracted geoms."""
    assert tree.root is not None
    spec = mujoco.MjSpec()
    spec.modelname = model_name
    spec.meshdir = meshdir
    spec.compiler.degree = False
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE

    mesh_assets: set[str] = set()
    mj_bodies: dict[Sdf.Path, mujoco.MjsBody] = {}

    def add_body(path: Sdf.Path, parent_mj: mujoco.MjsBody | None) -> None:
        usd_body = tree.bodies[path]
        if parent_mj is None:
            mj_body = spec.worldbody.add_body(name=path.name)
            if freejoint:
                mj_body.add_freejoint(name=f"{path.name}_freejoint")
        else:
            joint_info = tree.joint_to_child[path]
            pos, quat = child_pose_from_joint(joint_info.prim)
            mj_body = parent_mj.add_body(name=path.name, pos=pos, quat=quat)
            _add_joint(mj_body, joint_info)

        _set_body_inertial(mj_body, usd_body)
        _add_geoms_to_body(
            spec,
            mj_body,
            body_geoms.get(path, []),
            body_name=path.name,
            mesh_assets=mesh_assets,
            mesh_files=mesh_files,
        )
        mj_bodies[path] = mj_body

        for child in tree.children_of.get(path, []):
            add_body(child, mj_body)

    add_body(tree.root, None)
    return spec


def convert_usd_to_mjcf(
    usd_path: str | Path,
    *,
    out_xml: str | Path | None = None,
    mesh_dir: str | Path | None = None,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, Path, KinematicTree]:
    """Convert a single-robot Isaac Lab USD file to MJCF beside the input.

    Writes mesh STL files under ``mesh_dir`` (default: ``<usd_dir>/meshes``) and
    the MJCF XML to ``out_xml`` (default: same stem as the USD with ``.xml``).

    Returns ``(spec, model, xml_path, kinematic_tree)``.
    """
    usd_path = Path(usd_path).resolve()
    xml_path = Path(out_xml).resolve() if out_xml is not None else usd_path.with_suffix(".xml")
    meshes_path = Path(mesh_dir).resolve() if mesh_dir is not None else usd_path.parent / "meshes"

    stage = load_usd(usd_path)
    tree = build_kinematic_tree(stage)
    body_geoms = extract_robot_geoms(tree)

    mesh_geoms = {
        g.name: g.mesh
        for geoms in body_geoms.values()
        for g in geoms
        if g.kind == "mesh" and g.mesh is not None
    }
    written = export_meshes(mesh_geoms, meshes_path)
    mesh_files = {name: path.name for name, path in written.items()}

    spec = build_mjcf(
        tree,
        body_geoms,
        model_name=usd_path.stem,
        meshdir="meshes",
        mesh_files=mesh_files,
    )

    prev_cwd = Path.cwd()
    try:
        os.chdir(usd_path.parent)
        model = spec.compile()
        if xml_path.parent.resolve() == usd_path.parent.resolve():
            spec.to_file(xml_path.name)
        else:
            spec.to_file(str(xml_path))
    finally:
        os.chdir(prev_cwd)

    return spec, model, xml_path, tree
