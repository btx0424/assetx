"""CLI: convert a URDF file to MJCF."""

from __future__ import annotations

import argparse

from assetx import launch_preview
from assetx.conversion.urdf2mjcf import urdf_to_mjcf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a URDF file to a MJCF file.")
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to the URDF file."
    )
    parser.add_argument(
        "--meshdir",
        "-m",
        type=str,
        default=None,
        help="MuJoCo compiler meshdir. Overrides any existing value when set.",
    )
    parser.add_argument(
        "--package-path",
        action="append",
        default=None,
        metavar="NAME=/path/to/pkg",
        help=(
            "Map a ROS package name to a filesystem root for package:// URIs. "
            "Repeatable. Auto-discovery from the URDF location is used when omitted."
        ),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after conversion.",
    )
    args = parser.parse_args()

    spec, _model, out_path = urdf_to_mjcf(
        args.path,
        meshdir=args.meshdir,
        package_paths=args.package_path,
    )
    print(out_path)

    if args.no_viewer:
        return

    launch_preview(spec)


if __name__ == "__main__":
    main()
