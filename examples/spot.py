"""Build Boston Dynamics Spot and Spot+arm MJCF from mujoco_menagerie."""

from __future__ import annotations

import argparse
from pathlib import Path

from assetx import MujocoAsset, launch_preview
from assetx.core.transforms import (
    AddDummyBody,
    ApproximateWithCapsule,
    Compose,
    GeomsToBody,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    RenameBodies,
    Transform,
)
from assetx.fetch import add_fetch_argument, fetch_method_from_args
from assetx.vendor import fetch_vendor

_ARTIFACTS = Path("artifacts")

MENAGERIE_SPOT = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/"
    "71f066ad0be9cd271f7ed58c030243ef157af9f4/boston_dynamics_spot"
)


def _base_transforms() -> list[Transform]:
    return [
        NormalizeGeomNames(),
        NormalizeJointNames(),
        RemoveSensors(names=[".*"]),
        RemoveActuators(names=[".*"]),
        RemoveSceneSettings(),
        RenameBodies({"body": "base_link"}),
        *(
            GeomsToBody([leg.upper()], f"{leg}_foot", mass=0.05)
            for leg in ("fl", "fr", "hl", "hr")
        ),
        ApproximateWithCapsule(
            [[f"{leg}_uleg_collision0"] for leg in ("fl", "fr", "hl", "hr")],
            names=[f"{leg}_uleg_collision0" for leg in ("fl", "fr", "hl", "hr")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.6,
        ),
        ApproximateWithCapsule(
            [[f"{leg}_lleg_collision0"] for leg in ("fl", "fr", "hl", "hr")],
            names=[f"{leg}_lleg_collision0" for leg in ("fl", "fr", "hl", "hr")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.9,
        ),
    ]


def build_spot(vendor_dir: Path) -> MujocoAsset:
    return Compose(_base_transforms()).transform(
        MujocoAsset.from_file(vendor_dir / "spot.xml")
    )


def build_spot_arm(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            *_base_transforms(),
            ApproximateWithCapsule(
                [["arm_link_hr0_collision0"], ["arm_link_el1_collision0"]],
                names=["arm_link_hr0_collision0", "arm_link_el1_collision0"],
                replace=True,
                coverage_quantile=0.95,
            ),
            AddDummyBody(
                parent_path="arm_link_wr1",
                name="grasp_point",
                pos=(0.18, 0.0, -0.02),
                align_to="world",
                marker_size=0.01,
                rgba=(1.0, 0.0, 0.0, 0.6),
            ),
        ]
    ).transform(MujocoAsset.from_file(vendor_dir / "spot_arm.xml"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Build spot_arm (with grasp_point) instead of spot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/spot or artifacts/spot_arm).",
    )
    parser.add_argument("--no-viewer", action="store_true")
    add_fetch_argument(parser)
    args = parser.parse_args()

    vendor_dir = fetch_vendor(MENAGERIE_SPOT, method=fetch_method_from_args(args))
    if args.arm:
        robot = build_spot_arm(vendor_dir)
        out = args.output or _ARTIFACTS / "spot_arm"
    else:
        robot = build_spot(vendor_dir)
        out = args.output or _ARTIFACTS / "spot"
    saved = robot.save(out)
    print(saved.xml_path)

    if args.no_viewer:
        return
    launch_preview(robot)


if __name__ == "__main__":
    main()
