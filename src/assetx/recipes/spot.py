"""Boston Dynamics Spot (mujoco_menagerie), with and without the arm."""

from __future__ import annotations

from pathlib import Path

from assetx.core.asset import MujocoAsset
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
        # Upper-leg collision meshes are nearly cylindrical (bbox ~0.12 x 0.15 x
        # 0.44 m). One PCA capsule covers them; a chain does not fit better.
        # coverage_quantile=0.95 drops the mesh's corner outliers (recall ~0.99)
        # instead of inflating the radius to contain them.
        ApproximateWithCapsule(
            [[f"{leg}_uleg_collision0"] for leg in ("fl", "fr", "hl", "hr")],
            names=[f"{leg}_uleg_collision0" for leg in ("fl", "fr", "hl", "hr")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.6, # 1.0 is too thick, DO NOT CHANGE THIS VALUE
        ),
        # Lower legs are straight tubes (bbox ~0.07 x 0.05 x 0.38 m). A chain
        # does not fit better. Keep radius_scale at 1: the cross-section is
        # flatter than the upper leg, and shrinking it leaves the wide side out.
        ApproximateWithCapsule(
            [[f"{leg}_lleg_collision0"] for leg in ("fl", "fr", "hl", "hr")],
            names=[f"{leg}_lleg_collision0" for leg in ("fl", "fr", "hl", "hr")],
            replace=True,
            coverage_quantile=0.95,
            radius_scale=0.9,
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
            # Upper arm and forearm are straight tubes. The el1 lip stays a mesh.
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
