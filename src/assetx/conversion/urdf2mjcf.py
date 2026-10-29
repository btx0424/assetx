"""URDF -> MJCF conversion helpers."""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

_PACKAGE_URI_RE = re.compile(r"^package://([^/]+)/(.*)$")
_FILE_URI_RE = re.compile(r"^file://(.*)$")
_MUJOCO_MESH_SUFFIXES = {".stl", ".obj", ".msh"}
# ``mjuu_fullInertia`` rejects a smallest eigenvalue below ``mjEPS``.
_MJ_INERTIA_EPS = 1e-14

logger = logging.getLogger(__name__)


def _tag_local(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _read_package_name(package_xml: Path) -> str | None:
    try:
        root = ET.fromstring(package_xml.read_text())
    except ET.ParseError:
        return None
    name_el = root.find("name")
    if name_el is not None and name_el.text:
        return name_el.text.strip()
    return None


def find_ros_package(
    package_name: str,
    start: Path,
    *,
    package_paths: dict[str, Path] | None = None,
) -> Path:
    """Locate a ROS package root for ``package://`` URI resolution.

    Search order:
    1. Explicit ``package_paths[package_name]`` override.
    2. Walk ``start`` and its parents:
       - directory named ``package_name`` (prefer one with ``package.xml``);
       - sibling ``parent / package_name``;
       - ancestor whose ``package.xml`` ``<name>`` matches ``package_name``.
    """
    if package_paths and package_name in package_paths:
        root = Path(package_paths[package_name]).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"package_paths['{package_name}'] is not a directory: {root}"
            )
        return root

    start = start.resolve()
    candidates: list[Path] = []
    for parent in [start, *start.parents]:
        named = parent if parent.name == package_name else parent / package_name
        if named.is_dir():
            candidates.append(named)
        pkg_xml = parent / "package.xml"
        if pkg_xml.is_file() and _read_package_name(pkg_xml) == package_name:
            candidates.append(parent)

    def score(p: Path) -> tuple[int, int]:
        return (
            int((p / "package.xml").is_file()),
            int((p / "meshes").is_dir()),
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        raise FileNotFoundError(
            f"Could not resolve ROS package '{package_name}' starting from {start}. "
            f"Pass package_paths={{'{package_name}': Path('...')}} or --package-path."
        )
    unique.sort(key=score, reverse=True)
    return unique[0]


def resolve_mesh_filename(
    filename: str,
    *,
    urdf_path: Path,
    package_paths: dict[str, Path] | None = None,
) -> str:
    """Resolve a URDF mesh ``filename`` to a filesystem path string.

    Handles:
    - ``package://pkg/rel/path`` (ROS share-style URIs)
    - ``file:///abs/path`` / ``file://abs/path``
    - plain relative / absolute paths (returned unchanged)
    """
    filename = filename.strip()
    m = _PACKAGE_URI_RE.match(filename)
    if m:
        pkg_name, rel = m.group(1), m.group(2)
        pkg_root = find_ros_package(
            pkg_name, urdf_path.parent, package_paths=package_paths
        )
        resolved = (pkg_root / rel).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Mesh from package://{pkg_name}/{rel} not found at {resolved}"
            )
        return str(resolved)

    m = _FILE_URI_RE.match(filename)
    if m:
        path_str = m.group(1)
        if path_str.startswith("localhost/"):
            path_str = path_str[len("localhost") :]
        resolved = Path(path_str).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Mesh from file URI not found: {resolved}")
        return str(resolved)

    return filename


def rewrite_ros_mesh_uris(
    root: ET.Element,
    *,
    urdf_path: Path,
    package_paths: dict[str, Path] | None = None,
) -> None:
    """In-place: rewrite ``package://`` / ``file://`` mesh filenames to real paths."""
    for elem in root.iter():
        if _tag_local(elem) != "mesh":
            continue
        filename = elem.get("filename")
        if not filename:
            continue
        if filename.startswith("package://") or filename.startswith("file://"):
            elem.set(
                "filename",
                resolve_mesh_filename(
                    filename, urdf_path=urdf_path, package_paths=package_paths
                ),
            )


def _inertia_matrix(elem: ET.Element) -> np.ndarray:
    def component(name: str) -> float:
        raw = elem.get(name)
        return 0.0 if raw is None else float(raw)

    ixx, iyy, izz = component("ixx"), component("iyy"), component("izz")
    ixy, ixz, iyz = component("ixy"), component("ixz"), component("iyz")
    return np.array(
        [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float
    )


def _repair_inertia(mat: np.ndarray) -> np.ndarray | None:
    """Return a MuJoCo-legal inertia, or ``None`` when ``mat`` already is.

    The smallest eigenvalue must be at least ``mjEPS`` (``1e-14``), and the
    principal moments must satisfy the triangle inequality.
    """
    if not np.all(np.isfinite(mat)):
        return np.eye(3) * _MJ_INERTIA_EPS

    symmetric = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    if float(np.max(np.abs(eigvals))) < _MJ_INERTIA_EPS:
        # CAD placeholder (mass ~ 0). Eigenvectors are noise; use a tiny sphere.
        return np.eye(3) * _MJ_INERTIA_EPS

    repaired = np.maximum(eigvals, _MJ_INERTIA_EPS)
    order = np.argsort(repaired)
    smallest, middle, largest = (float(repaired[i]) for i in order)
    changed = bool(np.any(eigvals < _MJ_INERTIA_EPS))
    if smallest + middle < largest:
        repaired = repaired.copy()
        repaired[order[0]] = largest - middle
        changed = True
    if not changed:
        return None

    out = eigvecs @ np.diag(repaired) @ eigvecs.T
    out = 0.5 * (out + out.T)
    eig_out = np.linalg.eigvalsh(out)
    short = _MJ_INERTIA_EPS - float(eig_out.min())
    if short > 0.0:
        out = out + short * np.eye(3)
    return out


def _sanitize_inertias(root: ET.Element) -> None:
    """Rewrite link inertias that ``MjSpec.compile`` would reject."""
    for link in root.iter():
        if _tag_local(link) != "link":
            continue
        inertial = next((child for child in link if _tag_local(child) == "inertial"), None)
        if inertial is None:
            continue
        inertia = next((child for child in inertial if _tag_local(child) == "inertia"), None)
        if inertia is None:
            continue
        repaired = _repair_inertia(_inertia_matrix(inertia))
        if repaired is None:
            continue
        logger.info("repaired inertia on link %s", link.get("name"))
        for name, value in (
            ("ixx", repaired[0, 0]),
            ("iyy", repaired[1, 1]),
            ("izz", repaired[2, 2]),
            ("ixy", repaired[0, 1]),
            ("ixz", repaired[0, 2]),
            ("iyz", repaired[1, 2]),
        ):
            inertia.set(name, f"{float(value):.16g}")


def prepare_urdf_for_mujoco(
    xml_string: str,
    meshdir: str | None = None,
    *,
    urdf_path: Path | None = None,
    package_paths: dict[str, Path] | None = None,
    mesh_work_dir: Path | None = None,
) -> str:
    """Inject a MuJoCo ``<compiler>`` block and resolve ROS mesh URIs.

    Inertias MuJoCo would reject are repaired first: eigenvalues below
    ``1e-14`` are lifted, and principal moments are adjusted so ``A + B >= C``.
    Keeps a single ``<material>`` per ``<visual>`` (MuJoCo rejects repeats).
    When ``mesh_work_dir`` is set, meshes MuJoCo cannot load (such as Collada)
    are converted to STL there. ``urdf_to_mjcf`` records those originals so USD
    export can put the colored meshes back.
    """
    xml, _sources = _prepare_urdf_xml(
        xml_string,
        meshdir=meshdir,
        urdf_path=urdf_path,
        package_paths=package_paths,
        mesh_work_dir=mesh_work_dir,
    )
    return xml


def _prepare_urdf_xml(
    xml_string: str,
    meshdir: str | None = None,
    *,
    urdf_path: Path | None = None,
    package_paths: dict[str, Path] | None = None,
    mesh_work_dir: Path | None = None,
) -> tuple[str, dict[str, Path]]:
    """Return ``(xml, stem_to_source)`` for meshes rewritten to STL."""
    root = ET.fromstring(xml_string)
    visual_sources: dict[str, Path] = {}
    _sanitize_inertias(root)

    for child in list(root):
        if _tag_local(child) == "mujoco":
            root.remove(child)

    if urdf_path is not None:
        rewrite_ros_mesh_uris(root, urdf_path=urdf_path, package_paths=package_paths)
        _keep_one_visual_material(root)
        if mesh_work_dir is not None:
            visual_sources = _convert_unsupported_meshes(root, urdf_path, mesh_work_dir)

    mujoco_elem = ET.Element("mujoco")
    compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("angle", "radian")
    # Absolute mesh paths after URI rewrite — do not strip directories.
    compiler.set("strippath", "false")
    if meshdir is not None:
        compiler.set("meshdir", meshdir)
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    root.insert(0, mujoco_elem)

    ET.indent(root, space="  ", level=0)
    return (
        ET.tostring(root, encoding="unicode", xml_declaration=True, method="xml"),
        visual_sources,
    )


def _keep_one_visual_material(root: ET.Element) -> None:
    for elem in root.iter():
        if _tag_local(elem) != "visual":
            continue
        materials = [child for child in list(elem) if _tag_local(child) == "material"]
        for extra in materials[1:]:
            elem.remove(extra)


def _convert_unsupported_meshes(
    root: ET.Element, urdf_path: Path, mesh_work_dir: Path
) -> dict[str, Path]:
    import trimesh

    mesh_work_dir.mkdir(parents=True, exist_ok=True)
    converted: dict[Path, Path] = {}
    taken: set[Path] = set()
    visual_sources: dict[str, Path] = {}
    for elem in root.iter():
        if _tag_local(elem) != "mesh":
            continue
        filename = elem.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_file():
            mesh_path = (urdf_path.parent / filename).resolve()
        else:
            mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh not found: {filename} (from {urdf_path})")
        if mesh_path.suffix.lower() in _MUJOCO_MESH_SUFFIXES:
            elem.set("filename", str(mesh_path))
            continue
        dest = converted.get(mesh_path)
        if dest is None:
            dest = _unique_stl(mesh_work_dir, mesh_path.stem, taken)
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.to_mesh()
            if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
                raise ValueError(f"Cannot convert {mesh_path} to STL: no triangles")
            loaded.export(str(dest), file_type="stl")
            converted[mesh_path] = dest
            taken.add(dest)
            visual_sources[dest.stem] = mesh_path
        elem.set("filename", str(dest))
    return visual_sources


def _unique_stl(mesh_dir: Path, stem: str, taken: set[Path]) -> Path:
    candidate = mesh_dir / f"{stem}.stl"
    index = 1
    while candidate in taken or candidate.exists():
        candidate = mesh_dir / f"{stem}_{index}.stl"
        index += 1
    return candidate


def _converter_geom_name(geom: mujoco.MjsGeom) -> str:
    """Prim name ``mujoco-usd-converter`` assigns this geom."""
    if geom.name:
        return geom.name
    if geom.meshname:
        return geom.meshname
    fallback = {
        mujoco.mjtGeom.mjGEOM_PLANE: "Plane",
        mujoco.mjtGeom.mjGEOM_SPHERE: "Sphere",
        mujoco.mjtGeom.mjGEOM_BOX: "Box",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "Cylinder",
        mujoco.mjtGeom.mjGEOM_CAPSULE: "Capsule",
    }
    return fallback.get(int(geom.type), "Mesh")


def _unique_name(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    index = 1
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def _dae_visual(
    spec: mujoco.MjSpec, geom: mujoco.MjsGeom, stem_to_source: dict[str, Path]
) -> tuple[str, Path] | None:
    """``(stl_stem, original_dae)`` for a non-colliding geom baked from DAE."""
    if int(geom.contype) or int(geom.conaffinity) or not geom.meshname:
        return None
    mesh = spec.mesh(geom.meshname)
    if mesh is None or not mesh.file:
        return None
    stem = Path(mesh.file).stem
    source = stem_to_source.get(stem)
    if source is None:
        return None
    return stem, source


def _visual_prim_sources(
    spec: mujoco.MjSpec, stem_to_source: dict[str, Path]
) -> dict[str, Path]:
    """Map each DAE visual's USD prim name to its original mesh.

    The converter names a geom from ``geom.name``, otherwise from the mesh
    name, so a named visual does not match the STL stem. A prim name that
    would collide with a sibling or with a different DAE is renamed to
    ``<stem>_visual``.
    """
    out: dict[str, Path] = {}
    taken = {geom.name for geom in spec.geoms if geom.name}
    for body in spec.bodies:
        geoms = list(body.geoms)
        for index, geom in enumerate(geoms):
            found = _dae_visual(spec, geom, stem_to_source)
            if found is None:
                continue
            stem, source = found
            if geom.name and out.get(geom.name, source) == source:
                out[geom.name] = source
                continue
            siblings = {
                _converter_geom_name(other) for i, other in enumerate(geoms) if i != index
            }
            if not geom.name and stem not in siblings and out.get(stem, source) == source:
                out[stem] = source
                continue
            name = _unique_name(f"{stem}_visual", taken | siblings)
            geom.name = name
            taken.add(name)
            out[name] = source
    return out


def _parse_package_path_args(entries: list[str] | None) -> dict[str, Path]:
    """Parse CLI-style ``NAME=/abs/path`` entries into a mapping."""
    out: dict[str, Path] = {}
    if not entries:
        return out
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --package-path '{entry}'. Expected NAME=/path/to/package"
            )
        name, path = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --package-path '{entry}': empty package name")
        out[name] = Path(path).expanduser().resolve()
    return out


def urdf_to_mjcf(
    urdf_path: str | Path,
    *,
    meshdir: str | None = None,
    out_xml: str | Path | None = None,
    package_paths: dict[str, Path] | list[str] | None = None,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, Path]:
    """Convert a URDF file to MJCF beside the input (same stem, ``.xml``).

    Resolves ``package://`` mesh URIs relative to the URDF's ROS package (or
    ``package_paths`` overrides). Returns ``(spec, model, xml_path)``.
    """
    in_path = Path(urdf_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {in_path}")

    if isinstance(package_paths, list):
        package_paths = _parse_package_path_args(package_paths)

    xml_out = Path(out_xml).resolve() if out_xml is not None else in_path.with_suffix(".xml")
    xml_out.parent.mkdir(parents=True, exist_ok=True)
    prepared, visual_sources = _prepare_urdf_xml(
        in_path.read_text(),
        meshdir=meshdir,
        urdf_path=in_path,
        package_paths=package_paths,
        mesh_work_dir=xml_out.parent / f"{xml_out.stem}_meshes",
    )
    spec = mujoco.MjSpec.from_string(prepared)
    if visual_sources:
        visual_sources = _visual_prim_sources(spec, visual_sources)

    prev_cwd = Path.cwd()
    try:
        # Relative mesh filenames (if any remain) resolve against the URDF dir.
        os.chdir(in_path.parent)
        model = spec.compile()
        if xml_out.parent.resolve() == in_path.parent.resolve():
            spec.to_file(xml_out.name)
        else:
            spec.to_file(str(xml_out))
    finally:
        os.chdir(prev_cwd)

    if visual_sources:
        from assetx.conversion.usd.visual_meshes import (
            sources_path_for_mjcf,
            write_visual_mesh_sources,
        )

        write_visual_mesh_sources(sources_path_for_mjcf(xml_out), visual_sources)

    return spec, model, xml_out
