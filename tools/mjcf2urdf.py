"""CLI: convert an MJCF file to URDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco

from assetx.conversion.mjcf2urdf import write_urdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an MJCF file to URDF (best-effort).",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the MJCF (.xml) file.",
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
    out_path = in_path.with_suffix(".urdf")
    prev = Path.cwd()
    try:
        os.chdir(in_path.parent)
        spec = mujoco.MjSpec.from_file(in_path.name)
    finally:
        os.chdir(prev)

    written = write_urdf(
        spec,
        out_path,
        robot_name=args.robot_name or (spec.modelname or in_path.stem),
        meshdir=args.meshdir,
    )
    print(written)


if __name__ == "__main__":
    main()
