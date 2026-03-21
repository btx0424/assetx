"""Convert a MuJoCo MJCF model to a URDF file (best-effort, for tooling pipelines).

Limitations: assumes a single base body under ``worldbody``, a tree of descendant bodies,
one movable joint (or weld) per non-root child body, and joint origins aligned with
typical MJCF (``body.pos`` / ``body.quat``). The MJCF free joint on the base is not
represented in URDF (no dummy world link). Exotic joints (multiple non-free joints on
one body, flex, etc.) are not supported.
"""

from __future__ import annotations

import argparse
import os
import re
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


def _joint_limited(jnt: mujoco.MjsJoint) -> bool:
    if jnt.limited == mujoco.mjtLimited.mjLIMITED_TRUE:
        return True
    if jnt.limited == mujoco.mjtLimited.mjLIMITED_AUTO:
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        return lo < hi
    return False


def _mesh_filename(spec: mujoco.MjSpec, meshname: str, meshdir: str) -> str | None:
    for m in spec.meshes:
        if m.name == meshname and m.file:
            rel = Path(m.file).as_posix()
            return f"{meshdir.rstrip('/')}/{rel}"
    return None


def _geom_to_urdf(
    spec: mujoco.MjSpec,
    geom: mujoco.MjsGeom,
    meshdir: str,
    *,
    visual: bool,
) -> ET.Element | None:
    g = ET.Element("visual" if visual else "collision")
    o = ET.SubElement(g, "origin")
    px, py, pz = _vec3(geom.pos)
    rr, pp, yy = _quat_wxyz_to_rpy(geom.quat)
    o.set("xyz", f"{px} {py} {pz}")
    o.set("rpy", f"{rr} {pp} {yy}")

    gt = geom.type
    if gt == mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(geom.size[0])
        ET.SubElement(g, "geometry").append(
            ET.Element("sphere", {"radius": f"{r}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
        rad, half = float(geom.size[0]), float(geom.size[1])
        length = 2.0 * half
        ET.SubElement(g, "geometry").append(
            ET.Element("capsule", {"radius": f"{rad}", "length": f"{length}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        rad, half = float(geom.size[0]), float(geom.size[1])
        length = 2.0 * half
        ET.SubElement(g, "geometry").append(
            ET.Element("cylinder", {"radius": f"{rad}", "length": f"{length}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = (float(geom.size[i]) * 2.0 for i in range(3))
        ET.SubElement(g, "geometry").append(
            ET.Element("box", {"size": f"{sx} {sy} {sz}"})
        )
        return g
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        path = _mesh_filename(spec, geom.meshname, meshdir)
        if not path:
            return None
        ET.SubElement(g, "geometry").append(ET.Element("mesh", {"filename": path}))
        return g
    return None


def _add_inertial(link: ET.Element, body: mujoco.MjsBody) -> None:
    mass = float(body.mass)
    if mass <= 0.0 and not int(getattr(body, "explicitinertial", 0)):
        return
    if mass <= 0.0:
        mass = 1e-9

    iner = ET.SubElement(link, "inertial")
    ox, oy, oz = _vec3(body.ipos)
    rr, pp, yy = _quat_wxyz_to_rpy(body.iquat)
    origin = ET.SubElement(iner, "origin")
    origin.set("xyz", f"{ox} {oy} {oz}")
    origin.set("rpy", f"{rr} {pp} {yy}")
    ET.SubElement(iner, "mass", {"value": f"{mass}"})

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
        iner,
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
    for j in joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            free.append(j)
        else:
            other.append(j)
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
        vis_el = _geom_to_urdf(spec, geom, meshdir, visual=True)
        col_el = _geom_to_urdf(spec, geom, meshdir, visual=False)
        # Visual-only geoms: always show. Collision-enabled meshes: dual visual+collision.
        # Collision primitives (box/capsule/...) without a separate visual geom: URDF collision only.
        emit_visual = visual_only or geom.type == mujoco.mjtGeom.mjGEOM_MESH

        if emit_visual and vis_el is not None:
            link.append(vis_el)
        if col_el is not None:
            if not visual_only:
                link.append(col_el)
            elif vis_el is None or (
                ET.tostring(vis_el, encoding="unicode")
                != ET.tostring(col_el, encoding="unicode")
            ):
                link.append(col_el)

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
            jtype = "fixed"
            jname = _sanitize_urdf_name(
                f"fixed_{parent_link}_to_{link_name}", used_names
            )
        elif len(other_j) > 1:
            raise ValueError(
                f"Body {body.name!r}: multiple non-free joints on one body are not supported."
            )
        else:
            jnt = other_j[0]
            jname = _sanitize_urdf_name(jnt.name or f"joint_{link_name}", used_names)
            if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
                jtype = "continuous"
                if _joint_limited(jnt):
                    jtype = "revolute"
            elif jnt.type == mujoco.mjtJoint.mjJNT_SLIDE:
                jtype = "prismatic"
            elif jnt.type == mujoco.mjtJoint.mjJNT_BALL:
                jtype = "spherical"
            else:
                raise ValueError(
                    f"Body {body.name!r}: unsupported joint type {jnt.type!r}."
                )

        joint_el = ET.SubElement(robot, "joint", {"name": jname, "type": jtype})
        ET.SubElement(joint_el, "parent", {"link": parent_link})
        ET.SubElement(joint_el, "child", {"link": link_name})

        ox, oy, oz = _vec3(body.pos)
        rr, pp, yy = _quat_wxyz_to_rpy(body.quat)
        origin = ET.SubElement(joint_el, "origin")
        origin.set("xyz", f"{ox} {oy} {oz}")
        origin.set("rpy", f"{rr} {pp} {yy}")

        if jtype in ("revolute", "continuous", "prismatic"):
            jnt = other_j[0]
            ax, ay, az = _vec3(jnt.axis)
            ET.SubElement(joint_el, "axis", {"xyz": f"{ax} {ay} {az}"})
            if jtype in ("revolute", "prismatic") and _joint_limited(jnt):
                lo, hi = float(jnt.range[0]), float(jnt.range[1])
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
    used: dict[str, int] = {}
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
        used,
    )

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an MJCF file to URDF (best-effort; see module docstring)."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the MJCF (.xml) file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output URDF path. Default: input stem with .urdf in the same directory.",
    )
    parser.add_argument(
        "--meshdir",
        "-m",
        type=str,
        default="meshes",
        help='Prefix for mesh filenames in URDF (default: "meshes").',
    )
    parser.add_argument(
        "--robot-name",
        "-r",
        type=str,
        default=None,
        help="URDF robot name attribute. Default: MJCF model name or file stem.",
    )
    args = parser.parse_args()

    in_path = Path(args.path).resolve()
    os_cwd = Path.cwd()
    try:
        os.chdir(in_path.parent)
        spec = mujoco.MjSpec.from_file(in_path.name)
    finally:
        os.chdir(os_cwd)

    robot_name = args.robot_name or (spec.modelname or in_path.stem)
    tree = mjcf_to_urdf(spec, robot_name=robot_name, meshdir=args.meshdir)
    out_path = (
        in_path.with_suffix(".urdf") if args.output is None else Path(args.output)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        out_path,
        encoding="unicode",
        xml_declaration=True,
        method="xml",
    )
    print(out_path)


if __name__ == "__main__":
    main()
