"""Boston Dynamics Spot (mujoco_menagerie), with and without the arm."""

from __future__ import annotations

from pathlib import Path

from assetx.core.asset import MujocoAsset
from assetx.core.transforms import (
    AddDummyBody,
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
from assetx.recipes.registry import recipe

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
        # Vendor foot spheres FL/FR/HL/HR sit on *_lleg; give them their own
        # bodies (at the sphere center) for per-foot contact / height terms.
        *(
            GeomsToBody([leg.upper()], f"{leg}_foot", mass=0.05)
            for leg in ("fl", "fr", "hl", "hr")
        ),
    ]


@recipe("spot", vendor=MENAGERIE_SPOT)
def spot(vendor_dir: Path) -> MujocoAsset:
    return Compose(_base_transforms()).transform(MujocoAsset.from_file(vendor_dir / "spot.xml"))


@recipe("spot_arm", vendor=MENAGERIE_SPOT)
def spot_arm(vendor_dir: Path) -> MujocoAsset:
    return Compose(
        [
            *_base_transforms(),
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
