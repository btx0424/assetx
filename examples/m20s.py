"""Build the Deep Robotics M20S wheeled quadruped from the vendor MJCF.

The vendor file is a scene (floor, light, motors, IMU) around primitive
collisions. This recipe keeps the robot, drops the scene, and turns cylinders
into capsules except the wheels. A wheel cylinder is 9 cm in radius and 2.7 cm
in half-length; a capsule would add hemispherical caps and make a 23 cm tire.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from assetx import (
    Compose,
    FitRoundedWheels,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveGeoms,
    RemoveSceneSettings,
    RemoveSensors,
    ReplaceCylinderWithCapsule,
    Transform,
    launch_preview,
)
from assetx.fetch import add_fetch_argument, fetch_method_from_args
from assetx.vendor import fetch_vendor

_ARTIFACTS = Path("artifacts")

DEEP_M20S = (
    "https://github.com/DeepRoboticsLab/deep_robotics_model/tree/"
    "a547a0289deb7cda23578b252b4d54debfdcec19/M20S/mjcf"
)
M20S_MJCF = "M20S.xml"


class _RemoveUnnamedSensors(Transform):
    """Drop sensors that ``RemoveSensors`` cannot match because they have no name."""

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for sensor in list(spec.sensors):
            if not sensor.name:
                spec.delete(sensor)
        spec.compile()
        return replace(asset, spec=spec)


def build_m20s(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*"]),
            _RemoveUnnamedSensors(),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            RemoveGeoms(["floor"]),
            # Wheel half-length is 2.7 cm against a 9 cm radius, so a capsule
            # would add caps and turn the tire into a 23 cm ball.
            ReplaceCylinderWithCapsule(exclude=[".*_wheel"]),
            FitRoundedWheels(
                [[f"{leg}_wheel_collision0"] for leg in ("fl", "fr", "hl", "hr")]
            ),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / M20S_MJCF))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "m20s",
        help="Write model.xml and meshes here.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Disable the viewer.")
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Use the Viser browser preview instead of the MuJoCo viewer.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser HTTP port when --viser is set (default: 8080).",
    )
    parser.add_argument(
        "--convex-hulls",
        action="store_true",
        help="With --viser, show MuJoCo collision-mesh convex hull wireframes.",
    )
    add_fetch_argument(parser)
    args = parser.parse_args()

    vendor_dir = fetch_vendor(DEEP_M20S, method=fetch_method_from_args(args))
    print(f"Using M20S vendor: {vendor_dir / M20S_MJCF}")
    robot = build_m20s(vendor_dir)
    saved = robot.save(args.output)
    print(saved.xml_path)

    if args.no_viewer:
        return

    launch_preview(
        robot,
        viewer="viser" if args.viser else "mujoco",
        port=args.port,
        convex_hulls=args.convex_hulls,
    )


if __name__ == "__main__":
    main()
