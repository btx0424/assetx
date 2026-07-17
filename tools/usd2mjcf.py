"""CLI: convert a single-robot Isaac Lab USD file to MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

from assetx.conversion.usd import convert_usd_to_mjcf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a single-robot Isaac Lab USD file to MJCF.",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        required=True,
        help="Path to the robot USD file.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after conversion.",
    )
    args = parser.parse_args()

    usd_path = args.path.resolve()
    print(f"loaded: {usd_path}")

    _spec, model, xml_path, tree = convert_usd_to_mjcf(usd_path)
    print(tree.format())
    print()
    print(f"saved: {xml_path}")
    print(
        f"compiled: nbody={model.nbody} njnt={model.njnt} "
        f"ngeom={model.ngeom} nmesh={model.nmesh}"
    )
    print("bodies:", [model.body(i).name for i in range(1, model.nbody)])
    print("joints:", [model.joint(i).name for i in range(model.njnt)])

    if args.no_viewer:
        return

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
