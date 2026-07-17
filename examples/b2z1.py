from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    MujocoAsset,
    asset_builder,
    Compose,
    AddJoint,
    JointCfg,
    ReplaceCylinderWithCapsule,
    MergeBodiesParentChild,
    MergeBodies,
    MergeSubtree,
    RenameBodies,
    Body2Site,
)


@asset_builder
def load_b2z1(xml_path: str | Path) -> MujocoAsset:
    asset = MujocoAsset.from_file(xml_path)
    transform = Compose(
        [
            ReplaceCylinderWithCapsule(),
            AddJoint(body_path="base_link", joint_cfg=JointCfg(name="floating_base_joint", type="free", limited=False)),
            Body2Site(body_paths=[
                "f_dc_link",
                "r_dc_link",
                "f_oc_link",
                "r_oc_link",
                "arm_plat_a_link",
                "arm_plat_b_link",
            ]),
            MergeBodiesParentChild("base_link", ["head_Link", "tail_link"]),
            MergeSubtree("lidar_plat_link"),
            MergeBodies(["lidar_link", "lidar_plat_link"]),
            RenameBodies({
                "link00": "arm_link_0",
                "link01": "arm_link_1",
                "link02": "arm_link_2",
                "link03": "arm_link_3",
                "link04": "arm_link_4",
                "link05": "arm_link_5",
                "link06": "arm_link_6",
                "arm_virtual_base_link": "arm_base_link",
            }),
        ]
    )
    asset = transform.transform(asset)
    return asset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a B2Z1 MJCF asset.")
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to the B2Z1 MJCF XML file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=Path("artifacts/b2z1"),
        help="Output directory for the assembled model.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Launch MuJoCo passive viewer after export.",
    )
    args = parser.parse_args()

    asset = load_b2z1(args.path)
    print(asset.resolved_meshdir)
    asset = asset.save(args.output)
    print(asset.xml_path)

    if args.view:
        import mujoco
        import mujoco.viewer

        model = asset.spec.compile()
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while True:
                viewer.sync()
