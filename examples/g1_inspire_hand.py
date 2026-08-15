"""Approximate G1 Inspire finger collision meshes with PCA capsules."""

from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    ApproximateWithCapsule,
    Compose,
    MujocoAsset,
    SelectSubtree,
    asset_builder,
    launch_preview,
)

# Default path inside the lab51 object_hoi package.
_DEFAULT_MJCF = (
    Path(__file__).resolve().parents[2]
    / "object_hoi"
    / "src"
    / "assets"
    / "unitree_g1"
    / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)

# One capsule per finger-link collision mesh (left + right).
THUMB_COLLISION_GEOMS = [
    "L_thumb_proximal_base_collision",
    "L_thumb_proximal_collision",
    "L_thumb_intermediate_collision",
    "L_thumb_distal_collision",
    "R_thumb_proximal_base_collision",
    "R_thumb_proximal_collision",
    "R_thumb_intermediate_collision",
    "R_thumb_distal_collision",
]
FINGER_COLLISION_GEOMS = [
    "L_index_proximal_collision",
    "L_index_intermediate_collision",
    "L_middle_proximal_collision",
    "L_middle_intermediate_collision",
    "L_ring_proximal_collision",
    "L_ring_intermediate_collision",
    "L_pinky_proximal_collision",
    "L_pinky_intermediate_collision",
    "R_index_proximal_collision",
    "R_index_intermediate_collision",
    "R_middle_proximal_collision",
    "R_middle_intermediate_collision",
    "R_ring_proximal_collision",
    "R_ring_intermediate_collision",
    "R_pinky_proximal_collision",
    "R_pinky_intermediate_collision",
]


def _capsule_groups(geom_names: list[str]) -> tuple[list[list[str]], list[str]]:
    groups = [[name] for name in geom_names]
    names = [name.replace("_collision", "_capsule") for name in geom_names]
    return groups, names


@asset_builder
def build_g1_inspire_finger_capsules(
    base: MujocoAsset,
    *,
    replace: bool = True,
) -> MujocoAsset:
    """Approximate each finger collision mesh with a PCA-fitted capsule."""
    thumb_groups, thumb_names = _capsule_groups(THUMB_COLLISION_GEOMS)
    finger_groups, finger_names = _capsule_groups(FINGER_COLLISION_GEOMS)
    return Compose(
        [
            ApproximateWithCapsule(
                thumb_groups,
                names=thumb_names,
                replace=replace,
                radius_scale=0.9,
                height_scale=1.0,
                rgba=(0.2, 0.8, 0.3, 0.45),
            ),
            ApproximateWithCapsule(
                finger_groups,
                names=finger_names,
                replace=replace,
                radius_scale=0.95,
                height_scale=1.0,
                rgba=(0.2, 0.8, 0.3, 0.45),
            ),
        ]
    ).transform(base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approximate G1 Inspire finger collisions with capsules."
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=_DEFAULT_MJCF,
        help="Path to the G1 + Inspire hand MJCF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g1_inspire_finger_capsules"),
        help="Output directory for the transformed model.",
    )
    parser.add_argument(
        "--hand",
        choices=("both", "left", "right"),
        default="both",
        help="Optionally crop to one hand subtree for preview.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original collision meshes (do not replace).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the interactive MuJoCo preview.",
    )
    args = parser.parse_args()

    if not args.mjcf.is_file():
        raise SystemExit(f"MJCF not found: {args.mjcf}")

    robot = build_g1_inspire_finger_capsules(
        MujocoAsset.from_file(args.mjcf),
        replace=not args.keep_original,
    )

    if args.hand == "left":
        robot = SelectSubtree("L_hand_base_link").transform(robot)
    elif args.hand == "right":
        robot = SelectSubtree("R_hand_base_link").transform(robot)

    saved = robot.save(args.output)
    print(saved.xml_path)

    # Print fitted capsule sizes for a quick sanity check.
    for geom in saved.spec.geoms:
        if geom.name and geom.name.endswith("_capsule"):
            r, h, _ = geom.size
            print(f"{geom.name}: radius={r:.5f} half_height={h:.5f}")

    if not args.no_viewer:
        launch_preview(robot)


if __name__ == "__main__":
    main()
