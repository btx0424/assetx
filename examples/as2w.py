"""Build a Unitree AS2W MJCF from the vendor URDF.

``as2w_description`` ships ``as2w.urdf`` only. The URDF is fixed to the world,
so the MJCF adds ``floating_base_joint``. Calf collision meshes are straight
enough for one capsule. The thigh is a bent link with a motor housing: full
coverage cuts into the hip, and a thinner capsule still rounds through the
knee, so those meshes stay. Hips, wheels, the torso, and the JT128 lidar are
not tubes.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

from assetx import (
    AddJoint,
    ApproximateWithCapsule,
    Compose,
    JointCfg,
    MujocoAsset,
    NormalizeGeomNames,
    NormalizeJointNames,
    RemoveActuators,
    RemoveSceneSettings,
    RemoveSensors,
    Transform,
    launch_preview,
)
from assetx.conversion.urdf2mjcf import urdf_to_mjcf
from assetx.fetch import add_fetch_argument, fetch_method_from_args
from assetx.vendor import fetch_vendor

_ARTIFACTS = Path("artifacts")

UNITREE_AS2W = (
    "https://github.com/unitreerobotics/unitree_ros/tree/"
    "4ddbf6df0aa5bf8c8789d3edfa83e5e3ca45fe48/robots/as2w_description"
)
AS2W_URDF = "as2w.urdf"

_LEGS = ("FL", "FR", "RL", "RR")


class _CollisionGeomGroup(Transform):
    """Move collision geoms to group 3.

    URDF import leaves them in group 0, which the viewer draws on top of the
    visual meshes. Group 3 matches the Unitree AS2 collision class.
    """

    def transform(self, asset: MujocoAsset) -> MujocoAsset:
        spec = asset.spec.copy()
        for geom in spec.geoms:
            if int(geom.contype) or int(geom.conaffinity):
                geom.group = 3
        spec.compile()
        return replace(asset, spec=spec)


def _asset_from_urdf(vendor_dir: Path) -> MujocoAsset:
    xml_out = Path(tempfile.mkdtemp(prefix="assetx-as2w-")) / "as2w.xml"
    urdf_to_mjcf(vendor_dir / AS2W_URDF, out_xml=xml_out)
    return MujocoAsset.from_file(xml_out)


def build_as2w(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(),
            _CollisionGeomGroup(),
            AddJoint(
                "base_link",
                JointCfg(name="floating_base_joint", type="free"),
            ),
            # 95% coverage is ~4.2 cm, which grazes the wheel hub. The calf
            # cross-section is 7.0 cm by 10.2 cm, so the scale stays near 1.
            ApproximateWithCapsule(
                [[f"{leg}_calf_collision0"] for leg in _LEGS],
                names=[f"{leg}_calf_collision0" for leg in _LEGS],
                replace=True,
                coverage_quantile=0.95,
                radius_scale=0.9,
            ),
        ]
    ).transform(_asset_from_urdf(vendor_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "as2w",
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

    vendor_dir = fetch_vendor(UNITREE_AS2W, method=fetch_method_from_args(args))
    print(f"Using AS2W vendor: {vendor_dir / AS2W_URDF}")
    robot = build_as2w(vendor_dir)
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
