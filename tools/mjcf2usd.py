"""Isaac-free MJCF -> USD CLI (see :func:`assetx.conversion.usd.newton.mjcf_to_usd`).

By default the output is a single USDC layer in the Isaac Lab articulation
layout. Pass ``--keep-hierarchy`` to keep the converter's nested kinematic
tree (Newton / MuJoCo USD convention).

Example::

    uv run tools/mjcf2usd.py artifacts/spot/model.xml
    uv run tools/mjcf2usd.py artifacts/spot/model.xml --keep-hierarchy --layered
"""

from __future__ import annotations

import argparse
from pathlib import Path

from assetx.conversion.usd.newton import (
    MJCF_CONVERTER_PACKAGE as CONVERTER_PACKAGE,
    MJCF_CONVERTER_VERSION as CONVERTER_VERSION,
    mjcf_to_usd,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an MJCF to USD without Isaac Sim (newton mujoco-usd-converter)."
    )
    parser.add_argument("mjcf", type=Path, help="Input MJCF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <mjcf dir>/usd).",
    )
    parser.add_argument(
        "--keep-hierarchy",
        action="store_true",
        help="Keep nested rigid bodies (Newton/MuJoCo style) instead of flattening.",
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

    usd_path = mjcf_to_usd(
        args.mjcf,
        args.output,
        flatten=not args.keep_hierarchy,
        layer_structure=args.layered,
        physics_scene=args.physics_scene,
        comment=args.comment,
        verbose=args.verbose,
        converter_version=args.converter_version,
    )
    print(usd_path)


if __name__ == "__main__":
    main()
