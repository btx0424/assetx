from __future__ import annotations

import argparse
import math
from pathlib import Path

from assetx import (
    MujocoAsset,
    Compose,
    RenameBodies,
    AddSite,
    assemble,
    asset_builder,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]  # lab51/
DEFAULT_ROV = _REPO_ROOT / "aa-projects/aa-robot-models/underwater/BlueROVHeavy.xml"
DEFAULT_ARX = _REPO_ROOT / "reference/ARX_Model/X5/X5A/urdf/X5A.xml"


@asset_builder
def load_rov(xml_path: str | Path) -> MujocoAsset:
    return MujocoAsset.from_file(xml_path)


@asset_builder
def load_arx(xml_path: str | Path) -> MujocoAsset:
    return MujocoAsset.from_file(xml_path)


@asset_builder
def build_rov_arx(base: MujocoAsset, arm: MujocoAsset) -> MujocoAsset:
    """Mount ARX X5A on BlueROVHeavy ``base_link``.

    Child bodies are prefixed with ``arm_``. Gripper finger / wrist links are
    renamed for downstream policies (same convention as ``a2_piper``).
    """
    asset = assemble(
        parent=base,
        child=arm,
        parent_link="base_link",
        child_prefix="arm_",
        # Top-center of the ROV hull; tweak after visual inspection.
        translation=(0.0, 0.0, -0.1),
        rotation=(math.pi, 0.0, 0.0),
    )
    transform = Compose(
        [
            RenameBodies(
                {
                    "arm_link6": "gripper_base",
                    "arm_link7": "gripper_right",
                    "arm_link8": "gripper_left",
                }
            ),
            AddSite(
                body_path="gripper_base",
                name="grasp_site",
                # Between the X5A finger mounts (link7/8 at ~x=0.087).
                pos=(0.09, 0.0, 0.0),
                size=(0.01, 0.01, 0.01),
                type="sphere",
                rgba=(1.0, 0.0, 0.0, 0.6),
            ),
        ]
    )
    return transform.transform(asset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble BlueROVHeavy + ARX X5A into one MJCF asset."
    )
    parser.add_argument(
        "--rov",
        type=Path,
        default=DEFAULT_ROV,
        help="Path to BlueROVHeavy MJCF XML.",
    )
    parser.add_argument(
        "--arx",
        type=Path,
        default=DEFAULT_ARX,
        help="Path to ARX X5A MJCF XML (from urdf2mjcf).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("artifacts/rov_arx"),
        help="Output directory for the assembled model.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after export.",
    )
    args = parser.parse_args()

    robot = build_rov_arx(load_rov(args.rov), load_arx(args.arx))
    # Assemble stages meshes under a TemporaryDirectory; copy so the artifact
    # outlives that temp tree (plain symlink would dangle after GC).
    saved = robot.save(args.output, copy_meshes=True)
    print(saved.xml_path)

    if args.no_viewer:
        return

    from assetx import launch_preview

    launch_preview(robot)


if __name__ == "__main__":
    main()
