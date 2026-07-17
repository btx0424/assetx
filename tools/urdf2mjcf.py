"""CLI: convert a URDF file to MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

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
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after conversion.",
    )
    args = parser.parse_args()

    _spec, model, out_path = urdf_to_mjcf(args.path, meshdir=args.meshdir)
    print(out_path)

    if args.no_viewer:
        return

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
