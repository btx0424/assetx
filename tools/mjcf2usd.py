"""CLI: convert an MJCF file to USD via mujoco-usd-converter."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco_usd_converter
import usdex.core
from pxr import Sdf, Usd


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an MJCF (.xml) file to USD.")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=True,
        help="Path to the MJCF (.xml) file.",
    )
    args = parser.parse_args()

    path = Path(args.path).resolve()
    if path.suffix != ".xml":
        raise ValueError("Path must end with .xml")

    converter = mujoco_usd_converter.Converter()
    asset: Sdf.AssetPath = converter.convert(
        input_file=str(path),
        output_dir=str(path.parent),
    )
    stage: Usd.Stage = Usd.Stage.Open(asset.path)
    usdex.core.saveStage(stage)
    print(asset.path)


if __name__ == "__main__":
    main()
