"""MJCF -> URDF conversion helpers."""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def _vec3(arr: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(arr).reshape(-1)
    return float(a[0]), float(a[1]), float(a[2])


def _quat_wxyz_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(quat).reshape(-1)
    rot = R.from_quat(q, scalar_first=True)
    # URDF requires fixed-axis RPY; singular configurations can trigger
    # expected gimbal-lock warnings during conversion.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        return tuple(float(x) for x in rot.as_euler("xyz"))


def _sanitize_urdf_name(name: str, used: dict[str, int]) -> str:
    s = name.strip() or "link"
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    if not re.match(r"^[A-Za-z_]", s):
        s = f"L_{s}"
    base = s
    n = used.get(base, 0)
    used[base] = n + 1
    if n > 0:
        s = f"{base}_{n}"
    return s


def _joint_range(jnt: mujoco.MjsJoint) -> tuple[float, float]:
    return float(jnt.range[0]), float(jnt.range[1])


def _joint_limited(jnt: mujoco.MjsJoint) -> bool:
    """True when the joint has a finite position range that should become URDF limits.

    ``range="-inf inf"`` (or any non-finite bound) is treated as unlimited: MuJoCo may
    still mark such joints ``limited``, but URDF should use ``continuous`` / omit
    ``<limit>`` rather than emit ``±inf``.
    """
    lo, hi = _joint_range(jnt)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        return False
    if jnt.limited == mujoco.mjtLimited.mjLIMITED_FALSE:
        return False
    # TRUE or AUTO with a proper finite range.
    return True


def _mesh_filename(spec: mujoco.MjSpec, meshname: str, meshdir: str) -> str | None:
    prefix = meshdir.strip()
    if prefix in {"", "."}:
        prefix = ""
    for mesh in spec.meshes:
        if mesh.name == meshname and mesh.file:
            rel = Path(mesh.file).as_posix()
            return rel if not prefix else f"{prefix.rstrip('/')}/{rel}"
    return None


def _geom_to_urdf(
    spec: mujoco.MjSpec,
    geom: mujoco.MjsGeom,
    meshdir: str,
    *,
    visual: bool,
) -> ET.Element | None:
    g = ET.Element("visual" if visual else "collision")
    origin = ET.SubElement(g, "origin")
    px, py, pz = _vec3(geom.pos)
    rr, pp, yy = _quat_wxyz_to_rpy(geom.quat)
    origin.set("xyz", f"{px} {py} {pz}")
    origin.set("rpy", f"{rr} {pp} {yy}")

    gt = geom.type
    if gt == mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = float(geom.size[0])
        ET.SubElement(g, "geometry").append(
            ET.Element("sphere", {"radius": f"{radius}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius, half = float(geom.size[0]), float(geom.size[1])
        ET.SubElement(g, "geometry").append(
            ET.Element("capsule", {"radius": f"{radius}", "length": f"{2.0 * half}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = float(geom.size[0]), float(geom.size[1])
        ET.SubElement(g, "geometry").append(
            ET.Element("cylinder", {"radius": f"{radius}", "length": f"{2.0 * half}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = (float(geom.size[i]) * 2.0 for i in range(3))
        ET.SubElement(g, "geometry").append(
            ET.Element("box", {"size": f"{sx} {sy} {sz}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        filename = _mesh_filename(spec, geom.meshname, meshdir)
        if not filename:
            return None
        ET.SubElement(g, "geometry").append(ET.Element("mesh", {"filename": filename}))
        return g
    return None


def _add_inertial(link: ET.Element, body: mujoco.MjsBody) -> None:
    mass = float(body.mass)
    if mass <= 0.0 and not int(getattr(body, "explicitinertial", 0)):
        return
    if mass <= 0.0:
        mass = 1e-9

    inertial = ET.SubElement(link, "inertial")
    ox, oy, oz = _vec3(body.ipos)
    rr, pp, yy = _quat_wxyz_to_rpy(body.iquat)
    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", f"{ox} {oy} {oz}")
    origin.set("rpy", f"{rr} {pp} {yy}")
    ET.SubElement(inertial, "mass", {"value": f"{mass}"})

    fi = np.asarray(body.fullinertia).reshape(-1)
    use_full = (
        fi.size >= 6
        and bool(np.all(np.isfinite(fi[:6])))
        and abs(float(fi[3])) + abs(float(fi[4])) + abs(float(fi[5])) > 1e-12
    )
    if use_full:
        ixx, iyy, izz = float(fi[0]), float(fi[1]), float(fi[2])
        ixy, ixz, iyz = float(fi[3]), float(fi[4]), float(fi[5])
    else:
        ixx = float(body.inertia[0])
        iyy = float(body.inertia[1])
        izz = float(body.inertia[2])
        ixy = ixz = iyz = 0.0

    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx}",
            "ixy": f"{ixy}",
            "ixz": f"{ixz}",
            "iyy": f"{iyy}",
            "iyz": f"{iyz}",
            "izz": f"{izz}",
        },
    )


def _classify_joints(
    joints: Iterable[mujoco.MjsJoint],
) -> tuple[list[mujoco.MjsJoint], list[mujoco.MjsJoint]]:
    free: list[mujoco.MjsJoint] = []
    other: list[mujoco.MjsJoint] = []
    for joint in joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            free.append(joint)
        else:
            other.append(joint)
    return free, other


def _append_joint_and_recurse(
    robot: ET.Element,
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    parent_link: str | None,
    meshdir: str,
    used_names: dict[str, int],
) -> None:
    link_name = _sanitize_urdf_name(body.name or f"body_{body.id}", used_names)
    link = ET.SubElement(robot, "link", {"name": link_name})
    _add_inertial(link, body)

    for geom in body.geoms:
        visual_only = int(geom.contype) == 0 and int(geom.conaffinity) == 0
        visual_el = _geom_to_urdf(spec, geom, meshdir, visual=True)
        collision_el = _geom_to_urdf(spec, geom, meshdir, visual=False)
        emit_visual = visual_only or geom.type == mujoco.mjtGeom.mjGEOM_MESH

        if emit_visual and visual_el is not None:
            link.append(visual_el)
        if collision_el is not None and not visual_only:
            link.append(collision_el)

    free_j, other_j = _classify_joints(body.joints)

    if parent_link is None:
        if other_j:
            raise ValueError(
                f"Body {body.name!r}: root body has actuated joints but URDF has no "
                "parent link; add a fixed base in MJCF or use a non-root chain."
            )
        if len(free_j) > 1:
            raise ValueError(
                f"Body {body.name!r}: multiple free joints are not supported."
            )
    else:
        if free_j:
            raise ValueError(
                f"Body {body.name!r}: free joint is only allowed on the root MJCF body."
            )
        if not other_j:
            joint_type = "fixed"
            joint_name = _sanitize_urdf_name(
                f"fixed_{parent_link}_to_{link_name}", used_names
            )
        elif len(other_j) > 1:
            raise ValueError(
                f"Body {body.name!r}: multiple non-free joints on one body are not supported."
            )
        else:
            jnt = other_j[0]
            joint_name = _sanitize_urdf_name(jnt.name or f"joint_{link_name}", used_names)
            if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_type = "continuous"
                if _joint_limited(jnt):
                    joint_type = "revolute"
            elif jnt.type == mujoco.mjtJoint.mjJNT_SLIDE:
                joint_type = "prismatic"
            elif jnt.type == mujoco.mjtJoint.mjJNT_BALL:
                joint_type = "spherical"
            else:
                raise ValueError(
                    f"Body {body.name!r}: unsupported joint type {jnt.type!r}."
                )

        joint_el = ET.SubElement(robot, "joint", {"name": joint_name, "type": joint_type})
        ET.SubElement(joint_el, "parent", {"link": parent_link})
        ET.SubElement(joint_el, "child", {"link": link_name})

        ox, oy, oz = _vec3(body.pos)
        rr, pp, yy = _quat_wxyz_to_rpy(body.quat)
        origin = ET.SubElement(joint_el, "origin")
        origin.set("xyz", f"{ox} {oy} {oz}")
        origin.set("rpy", f"{rr} {pp} {yy}")

        if joint_type in ("revolute", "continuous", "prismatic"):
            jnt = other_j[0]
            ax, ay, az = _vec3(jnt.axis)
            ET.SubElement(joint_el, "axis", {"xyz": f"{ax} {ay} {az}"})
            if joint_type in ("revolute", "prismatic") and _joint_limited(jnt):
                lo, hi = _joint_range(jnt)
                ET.SubElement(
                    joint_el,
                    "limit",
                    {"lower": f"{lo}", "upper": f"{hi}", "effort": "0", "velocity": "0"},
                )

    for child in body.bodies:
        _append_joint_and_recurse(
            robot,
            spec,
            child,
            link_name,
            meshdir,
            used_names,
        )


def mjcf_to_urdf(
    spec: mujoco.MjSpec,
    *,
    robot_name: str,
    meshdir: str,
) -> ET.ElementTree:
    used_names: dict[str, int] = {}
    root = ET.Element("robot", {"name": robot_name})

    children = list(spec.worldbody.bodies)
    if not children:
        raise ValueError("MJCF has no bodies under <worldbody>.")
    if len(children) > 1:
        raise ValueError(
            "MJCF has multiple direct children under <worldbody>; URDF expects a single "
            "root link. Use one base body or attach extras under that body in MJCF."
        )

    _append_joint_and_recurse(
        root,
        spec,
        children[0],
        None,
        meshdir,
        used_names,
    )
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_urdf(
    spec: mujoco.MjSpec,
    output_path: str | Path,
    *,
    robot_name: str | None = None,
    meshdir: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = mjcf_to_urdf(
        spec,
        robot_name=robot_name or (spec.modelname or output.stem),
        meshdir=meshdir if meshdir is not None else str(spec.meshdir),
    )
    tree.write(
        output,
        encoding="unicode",
        xml_declaration=True,
        method="xml",
    )
    return output
