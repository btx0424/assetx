from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    Compose,
    GeomsToBody,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    launch_preview,
)
from assetx.fetch import add_fetch_argument, fetch_method_from_args
from assetx.vendor import fetch_vendor

_ARTIFACTS = Path("artifacts")

UNITREE_AS2 = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "38a9b00ecbff0b70e85a7627eb9c65db9a2341f7/robots/as2_description"
)
AS2_MJCF = "as2.xml"

_FOOT_GEOM_GROUPS = (
    ("FL", "FL_calf_collision3", "FL_calf_visual1"),
    ("FR", "FR_calf_collision3", "FR_calf_visual1"),
    ("RL", "RL_calf_collision3", "RL_calf_visual1"),
    ("RR", "RR_calf_collision3", "RR_calf_visual1"),
)


def build_as2(vendor_dir: Path) -> MujocoAsset:
    asset = MujocoAsset.from_file(vendor_dir / AS2_MJCF)
    transform = Compose([
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*pos", ".*torque", "imu.*", ".*vel"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        *(
            GeomsToBody([collision, visual], f"{prefix}_foot", mass=0.05)
            for prefix, collision, visual in _FOOT_GEOM_GROUPS
        ),
    ])
    return transform.transform(asset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Unitree AS2 MJCF asset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "as2",
        help="Write model.xml and meshes here.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Disable the viewer.")
    add_fetch_argument(parser)
    args = parser.parse_args()

    vendor_dir = fetch_vendor(UNITREE_AS2, method=fetch_method_from_args(args))
    print(f"Using AS2 vendor: {vendor_dir / AS2_MJCF}")
    robot = build_as2(vendor_dir)
    saved = robot.save(args.output)
    print(saved.xml_path)

    if args.no_viewer:
        return

    launch_preview(robot)


if __name__ == "__main__":
    main()
