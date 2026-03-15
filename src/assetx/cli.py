import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from assetx.utils import extract_meshes


def _tag_local(elem: ET.Element) -> str:
    """Return the local tag name without namespace."""
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _prepare_urdf_for_mujoco(xml_string: str, meshdir: str) -> str:
    """Parse URDF XML, set/override MuJoCo compiler directives, return XML string."""
    root = ET.fromstring(xml_string)

    # Remove existing mujoco block(s)
    for child in list(root):
        if _tag_local(child) == "mujoco":
            root.remove(child)

    # Insert mujoco compiler block at the start
    mujoco_elem = ET.Element("mujoco")
    compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", meshdir)
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    root.insert(0, mujoco_elem)

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode", xml_declaration=True, method="xml")


def extract_meshes_cli() -> None:
    """CLI: extract meshes from a USD file and save them to a directory."""
    parser = argparse.ArgumentParser(
        description="Extract Mesh and Cube prims from a USD file and export as STL or OBJ.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the USD file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for mesh files.",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="stl",
        choices=["stl", "obj"],
        help="Output mesh format (default: stl).",
    )
    args = parser.parse_args()

    meshes = extract_meshes(args.path)
    if not meshes:
        print("No mesh or cube prims found.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format.lower()
    for name, mesh in meshes.items():
        out_path = out_dir / f"{name}.{ext}"
        mesh.export(str(out_path))
        print(out_path)


def urdf2mjcf_cli() -> None:
    """CLI: convert a URDF file to a MJCF file."""
    parser = argparse.ArgumentParser(
        description="Convert a URDF file to a MJCF file.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the URDF file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output MJCF path. Default: input stem with .xml in the same directory.",
    )
    parser.add_argument(
        "--meshdir",
        "-m",
        type=str,
        default="meshes",
        help="MuJoCo compiler meshdir (default: meshes). Always overrides any existing value.",
    )
    # parser.add_argument(
    #     "--fixed_base",
    #     "-f",
    #     action="store_true",
    #     help="Use a fixed joint to world (default: floating). Only applies when a world link is inserted.",
    # )
    args = parser.parse_args()

    xml_string = Path(args.path).read_text()
    xml_string = _prepare_urdf_for_mujoco(xml_string, meshdir=args.meshdir)

    spec = mujoco.MjSpec.from_string(xml_string)
    os.chdir(Path(args.path).parent)
    spec.compile()
    out_path = args.output
    if out_path is None:
        out_path = Path(args.path).with_suffix(".xml")
    else:
        out_path = Path(out_path)
    spec.to_file(str(out_path))
    print(out_path)
