"""Isaac-free URDF -> USD CLI (see :func:`assetx.conversion.usd.newton.urdf_to_usd`).

Prefer ``mjcf2usd.py`` for assetx robots: the MJCF route keeps the floating
base, contact excludes and sites. Use this for URDF-only sources.

Example::

    uv run tools/urdf2usd.py artifacts/spot/model.urdf
    uv run tools/urdf2usd.py artifacts/spot/model.urdf --fix-base
    uv run tools/urdf2usd.py artifacts/spot/model.urdf --keep-hierarchy --layered
"""

from __future__ import annotations

import argparse
from pathlib import Path

from assetx.conversion.usd.newton import (
    URDF_CONVERTER_PACKAGE as CONVERTER_PACKAGE,
    URDF_CONVERTER_VERSION as CONVERTER_VERSION,
    urdf_to_usd,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a URDF to USD without Isaac Sim (newton urdf-usd-converter)."
    )
    parser.add_argument("urdf", type=Path, help="Input URDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <urdf dir>/usd).",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Keep the converter's joint welding the root link to the world.",
    )
    parser.add_argument(
        "--keep-hierarchy",
        action="store_true",
        help="Keep nested rigid bodies (Newton style) instead of flattening.",
    )
    parser.add_argument(
        "--layered",
        action="store_true",
        help="Write an Atomic Component layer structure (requires --keep-hierarchy).",
    )
    parser.add_argument(
        "--physics-scene",
        action="store_true",
        help="Also author a UsdPhysics.Scene prim (off by default).",
    )
    parser.add_argument(
        "-p",
        "--package",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="ROS package mapping for package:// URIs (repeatable).",
    )
    parser.add_argument("-c", "--comment", default="", help="Comment for the USD file.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--converter-version",
        default=CONVERTER_VERSION,
        help=f"{CONVERTER_PACKAGE} version (default: {CONVERTER_VERSION}).",
    )
    args = parser.parse_args()
    if args.layered and not args.keep_hierarchy:
        parser.error("--layered requires --keep-hierarchy")

    packages: dict[str, str | Path] = {}
    for item in args.package:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--package expects NAME=PATH, got {item!r}")
        packages[name] = path

    usd_path = urdf_to_usd(
        args.urdf,
        args.output,
        fix_base=args.fix_base,
        flatten=not args.keep_hierarchy,
        layer_structure=args.layered,
        physics_scene=args.physics_scene,
        packages=packages,
        comment=args.comment,
        verbose=args.verbose,
        converter_version=args.converter_version,
    )
    print(usd_path)


if __name__ == "__main__":
    main()
