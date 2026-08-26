from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    AddDummyBody,
    Compose,
    MujocoAsset,
    NormalizeGeomNames,
    RemoveActuators,
    RemoveSensors,
    RenameBodies,
    assemble,
    asset_builder,
)
from assetx.fetch import download_github_dir

_ARTIFACTS = Path("artifacts")
_VENDOR = _ARTIFACTS / "vendor"

B2_GITHUB = (
    "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/b2_description_mujoco"
)
B2_FILE = "xml/b2.xml"

KINOVA_GITHUB = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3"
)
KINOVA_FILE = "gen3.xml"


@asset_builder
def load_b2(xml_path: str | Path) -> MujocoAsset:
    asset = MujocoAsset.from_file(xml_path)
    # sensors and actuators should be added by downstream applications
    # do not hard-code in the xml
    transform = Compose([
        RemoveSensors(names=[".*pos", ".*vel", "imu.*"]),
        RemoveActuators(names=[".*"]),
    ])
    return transform.transform(asset)


@asset_builder
def load_kinova(xml_path: str | Path) -> MujocoAsset:
    asset = MujocoAsset.from_file(xml_path)
    transform = RemoveActuators(names=[".*"])
    return transform.transform(asset)


@asset_builder
def build_b2_kinova(base: MujocoAsset, arm: MujocoAsset) -> MujocoAsset:
    """Mount Kinova Gen3 on B2 ``base_link``."""
    asset = assemble(
        parent=base,
        child=arm,
        parent_link="base_link",
        child_prefix="arm_",
        # Top of the B2 torso; tweak after visual inspection.
        translation=(0.0, 0.0, 0.12),
        rotation=(0.0, 0.0, 0.0),
    )
    transform = Compose(
        [
            NormalizeGeomNames(),
            RenameBodies(
                {
                    "arm_bracelet_link": "ee_link",
                }
            ),
            # Menagerie pinch offset along world +Z at qpos0 (ee Z is flipped).
            AddDummyBody(
                parent_path="ee_link",
                name="grasp_point",
                pos=(0.0, 0.0, 0.061525),
                align_to="world",
                marker_size=0.01,
                rgba=(1.0, 0.0, 0.0, 0.6),
            ),
        ]
    )
    return transform.transform(asset)


def _resolve_mjcf(
    local: Path | None,
    *,
    github_url: str,
    vendor_dirname: str,
    relative_xml: str,
    force_download: bool,
) -> Path:
    if local is not None:
        path = local.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local MJCF path not found: {path}")
        return path

    dest = _VENDOR / vendor_dirname
    cached = dest.exists() and any(dest.rglob("*")) and not force_download
    if cached:
        print(f"Using cached vendor asset: {dest}")
    else:
        print(f"Fetching {github_url} -> {dest}")
    download_github_dir(github_url, dest, force=force_download)

    xml_path = dest / relative_xml
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"Expected MJCF at {xml_path} after fetching {github_url}"
        )
    return xml_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a B2 + Kinova Gen3 MJCF asset. "
            "If --b2/--kinova are omitted, download vendor folders into artifacts/vendor/."
        )
    )
    parser.add_argument(
        "--b2",
        type=Path,
        default=None,
        help="Local B2 MJCF file (default: download unitree b2_description_mujoco).",
    )
    parser.add_argument(
        "--kinova",
        type=Path,
        default=None,
        help="Local Kinova Gen3 MJCF file (default: download mujoco_menagerie kinova_gen3).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download vendor assets even if artifacts/vendor/ already has them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "b2_kinova",
        help="Output directory for the assembled model.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after export.",
    )
    args = parser.parse_args()

    b2_xml = _resolve_mjcf(
        args.b2,
        github_url=B2_GITHUB,
        vendor_dirname="b2_description_mujoco",
        relative_xml=B2_FILE,
        force_download=args.force_download,
    )
    kinova_xml = _resolve_mjcf(
        args.kinova,
        github_url=KINOVA_GITHUB,
        vendor_dirname="kinova_gen3",
        relative_xml=KINOVA_FILE,
        force_download=args.force_download,
    )
    print(f"Using B2: {b2_xml}")
    print(f"Using Kinova: {kinova_xml}")

    robot = build_b2_kinova(load_b2(b2_xml), load_kinova(kinova_xml))
    saved = robot.save(args.output)
    print(saved.xml_path)

    if args.no_viewer:
        return

    from assetx import launch_preview

    launch_preview(robot)


if __name__ == "__main__":
    main()
