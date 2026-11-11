"""Build the Seeed reBot B601 DevArm from the vendor colored MJCF.

The upstream file is a full pick-and-place scene (floor, table, objects, mocap
target). This recipe keeps only ``base_link``, strips scene settings, normalizes
names, and adds a ``grasp_point`` body at the vendor TCP site.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from assetx import (
    AddDummyBody,
    ApproximateWithConvexHull,
    Compose,
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

# Colored MJCF under Rebot_Arm_description/DM/mjcf/; meshes at ../meshes.
REBOT_ARM_DM = (
    "https://github.com/Seeed-Projects/reBot-DevArm/tree/"
    "96de545cd76ba3df64f05605553d3f06ab9298ec/Rebot_Arm_description/DM"
)
REBOT_ARM_MJCF = Path("mjcf") / "rebotarm_b601_colored.xml"

# Vendor ``tcp`` site on ``end_link`` (parent-relative).
_TCP_POS = (-0.105, 0.0, 0.0)

# Colored visual pieces per arm body (gripper already has collision meshes).
_ARM_LINK_VISUALS: dict[str, tuple[str, ...]] = {
    "base_link": (
        "base_link_hardware_black_visual",
        "base_link_matte_black_visual",
        "base_link_silver_trim_visual",
    ),
    "link1": (
        "link1_anodized_grey_visual",
        "link1_hardware_black_visual",
        "link1_matte_black_visual",
    ),
    "link2": (
        "link2_anodized_grey_visual",
        "link2_hardware_black_visual",
        "link2_matte_black_visual",
        "link2_seeed_yellow_visual",
    ),
    "link3": (
        "link3_anodized_grey_visual",
        "link3_hardware_black_visual",
        "link3_matte_black_visual",
        "link3_seeed_yellow_visual",
    ),
    "link4": (
        "link4_anodized_grey_visual",
        "link4_hardware_black_visual",
        "link4_matte_black_visual",
        "link4_seeed_yellow_visual",
    ),
    "link5": (
        "link5_anodized_grey_visual",
        "link5_hardware_black_visual",
        "link5_matte_black_visual",
    ),
    "link6": (
        "link6_hardware_black_visual",
        "link6_matte_black_visual",
    ),
}

_SCENE_BODY_NAMES = (
    "task_table",
    "red_cube",
    "blue_block",
    "yellow_cylinder",
    "ik_target",
)
_SCENE_MATERIAL_NAMES = (
    "floor_mat",
    "table_mat",
    "pick_zone_mat",
    "place_zone_mat",
    "object_red_mat",
    "object_blue_mat",
    "object_yellow_mat",
    "target_mat",
)
_SCENE_TEXTURE_NAMES = ("floor_checker",)


def _load_robot(xml_path: Path) -> MujocoAsset:
    """Load the scene MJCF and keep only the arm under ``base_link``."""
    path = xml_path.resolve()
    spec = mujoco.MjSpec.from_file(str(path))
    for body in list(spec.worldbody.bodies):
        if body.name in _SCENE_BODY_NAMES:
            spec.delete(body)
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for camera in list(spec.cameras):
        spec.delete(camera)
    for material in list(spec.materials):
        if material.name in _SCENE_MATERIAL_NAMES:
            spec.delete(material)
    for texture in list(spec.textures):
        if texture.name in _SCENE_TEXTURE_NAMES:
            spec.delete(texture)
    if len(spec.worldbody.bodies) != 1:
        names = [b.name for b in spec.worldbody.bodies]
        raise ValueError(f"expected a single robot body after stripping scene, got {names}")
    meshdir = Path(spec.meshdir) if spec.meshdir else Path(".")
    resolved = (path.parent / meshdir).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Meshdir {resolved} not found")
    return MujocoAsset(path, spec, meshdir)


def build_rebot_arm(vendor_dir: Path) -> MujocoAsset:
    asset = _load_robot(vendor_dir / REBOT_ARM_MJCF)
    return Compose(
        [
            NormalizeGeomNames(),
            NormalizeJointNames(),
            RemoveSensors(names=[".*"]),
            RemoveActuators(names=[".*"]),
            RemoveSceneSettings(visual=True),
            ApproximateWithConvexHull(
                [list(names) for names in _ARM_LINK_VISUALS.values()],
                names=[f"{body}_collision0" for body in _ARM_LINK_VISUALS],
                body_paths=list(_ARM_LINK_VISUALS),
                scale=0.9,
            ),
            AddDummyBody(
                parent_path="end_link",
                name="grasp_point",
                pos=_TCP_POS,
                align_to=None,
                marker_size=0.01,
                rgba=(1.0, 0.0, 0.0, 0.6),
            ),
        ]
    ).transform(asset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "rebot_arm",
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

    vendor_dir = fetch_vendor(REBOT_ARM_DM, method=fetch_method_from_args(args))
    print(f"Using reBot Arm vendor: {vendor_dir / REBOT_ARM_MJCF}")
    robot = build_rebot_arm(vendor_dir)
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
