"""URDF -> MJCF conversion helpers."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

_PACKAGE_URI_RE = re.compile(r"^package://([^/]+)/(.*)$")
_FILE_URI_RE = re.compile(r"^file://(.*)$")


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


def prepare_urdf_for_mujoco(
    xml_string: str,
    meshdir: str | None = None,
    *,
    urdf_path: Path | None = None,
    package_paths: dict[str, Path] | None = None,
) -> str:
    """Inject a MuJoCo ``<compiler>`` block and resolve ROS mesh URIs."""
    root = ET.fromstring(xml_string)

    for child in list(root):
        if _tag_local(child) == "mujoco":
            root.remove(child)

    if urdf_path is not None:
        rewrite_ros_mesh_uris(root, urdf_path=urdf_path, package_paths=package_paths)

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
    return ET.tostring(root, encoding="unicode", xml_declaration=True, method="xml")


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
    prepared = prepare_urdf_for_mujoco(
        in_path.read_text(),
        meshdir=meshdir,
        urdf_path=in_path,
        package_paths=package_paths,
    )
    spec = mujoco.MjSpec.from_string(prepared)

    prev_cwd = Path.cwd()
    try:
        # Relative mesh filenames (if any remain) resolve against the URDF dir.
        os.chdir(in_path.parent)
        model = spec.compile()
        xml_out.parent.mkdir(parents=True, exist_ok=True)
        if xml_out.parent.resolve() == in_path.parent.resolve():
            spec.to_file(xml_out.name)
        else:
            spec.to_file(str(xml_out))
    finally:
        os.chdir(prev_cwd)

    return spec, model, xml_out
