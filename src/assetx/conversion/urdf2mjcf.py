"""URDF -> MJCF conversion helpers."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def _tag_local(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def prepare_urdf_for_mujoco(xml_string: str, meshdir: str | None = None) -> str:
    """Inject a MuJoCo ``<compiler>`` block into URDF XML so MjSpec can load it."""
    root = ET.fromstring(xml_string)

    for child in list(root):
        if _tag_local(child) == "mujoco":
            root.remove(child)

    mujoco_elem = ET.Element("mujoco")
    compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("angle", "radian")
    compiler.set("strippath", "false")
    if meshdir is not None:
        compiler.set("meshdir", meshdir)
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    root.insert(0, mujoco_elem)

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode", xml_declaration=True, method="xml")


def urdf_to_mjcf(
    urdf_path: str | Path,
    *,
    meshdir: str | None = None,
    out_xml: str | Path | None = None,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, Path]:
    """Convert a URDF file to MJCF beside the input (same stem, ``.xml``).

    Returns ``(spec, model, xml_path)``.
    """
    in_path = Path(urdf_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {in_path}")

    xml_out = Path(out_xml).resolve() if out_xml is not None else in_path.with_suffix(".xml")
    prepared = prepare_urdf_for_mujoco(in_path.read_text(), meshdir=meshdir)
    spec = mujoco.MjSpec.from_string(prepared)

    prev_cwd = Path.cwd()
    try:
        os.chdir(in_path.parent)
        model = spec.compile()
        if xml_out.parent.resolve() == in_path.parent.resolve():
            spec.to_file(xml_out.name)
        else:
            spec.to_file(str(xml_out))
    finally:
        os.chdir(prev_cwd)

    return spec, model, xml_out
