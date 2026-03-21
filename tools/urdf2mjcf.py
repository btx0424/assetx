import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer


def _tag_local(elem: ET.Element) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _prepare_urdf_for_mujoco(xml_string: str, meshdir: str) -> str:
    root = ET.fromstring(xml_string)

    for child in list(root):
        if _tag_local(child) == "mujoco":
            root.remove(child)

    mujoco_elem = ET.Element("mujoco")
    compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", meshdir)
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    root.insert(0, mujoco_elem)

    ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode", xml_declaration=True, method="xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a URDF file to a MJCF file.")
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to the URDF file."
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
    args = parser.parse_args()

    xml_string = Path(args.path).read_text()
    xml_string = _prepare_urdf_for_mujoco(xml_string, meshdir=args.meshdir)

    spec = mujoco.MjSpec.from_string(xml_string)
    os.chdir(Path(args.path).parent)
    model = spec.compile()
    data = mujoco.MjData(model)
    out_path = (
        Path(args.path).with_suffix(".xml")
        if args.output is None
        else Path(args.output)
    )
    spec.to_file(str(out_path))
    print(out_path)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            viewer.sync()


if __name__ == "__main__":
    main()
