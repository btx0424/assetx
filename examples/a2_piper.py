from __future__ import annotations

import argparse
from pathlib import Path

from assetx import MujocoAsset, assemble, asset_builder
from assetx.transform import Compose, ReplaceCylinderWithCapsule, RenameBodies, AddSite


@asset_builder
def load_a2(xml_path: str | Path) -> MujocoAsset:
    return MujocoAsset.from_file(xml_path)


@asset_builder
def load_piper(xml_path: str | Path) -> MujocoAsset:
    return MujocoAsset.from_file(xml_path)


@asset_builder
def build_a2_piper(base: MujocoAsset, arm: MujocoAsset) -> MujocoAsset:
    asset = assemble(
        parent=base,
        child=arm,
        parent_link="base_link",
        child_prefix="arm_",
        translation=(0.05, 0.0, 0.10),
        rotation=(0.0, 0.0, 0.0),
    )
    transform = Compose(
        [
            ReplaceCylinderWithCapsule(),
            RenameBodies(
                {
                    "arm_link7": "gripper_right",
                    "arm_link8": "gripper_left",
                    "arm_link6": "gripper_base",
                }
            ),
            AddSite(
                body_path="gripper_base",
                name="grasp_site",
                pos=(0.05, 0.0, 0.0),
                size=(0.01, 0.01, 0.01),
                type="sphere",
                rgba=(1.0, 0.0, 0.0, 0.6),
            ),
        ]
    )
    return transform.transform(asset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an A2 + Piper MJCF asset.")
    parser.add_argument(
        "--a2", type=Path, required=True, help="Path to the A2 MJCF XML file."
    )
    parser.add_argument(
        "--piper", type=Path, required=True, help="Path to the Piper MJCF XML file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/a2_piper"),
        help="Output directory for the assembled model.",
    )
    args = parser.parse_args()

    robot = build_a2_piper(load_a2(args.a2), load_piper(args.piper))
    saved = robot.save(args.output)
    print(saved.xml_path)

    import mujoco.viewer

    model = robot.spec.compile()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            viewer.sync()


if __name__ == "__main__":
    main()
