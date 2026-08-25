from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    AddDummyBody,
    Compose,
    MujocoAsset,
    RenameBodies,
    ReplaceCylinderWithCapsule,
    NormalizeGeomNames,
    assemble,
    asset_builder,
)
from assetx.fetch import download_github_dir, find_mjcf

_ARTIFACTS = Path("artifacts")
_VENDOR = _ARTIFACTS / "vendor"

# Subdirectory-only fetches (no full-repo clone / no git history).
A2_GITHUB = (
    "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a2_description"
)
PIPER_GITHUB = (
    "https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper"
)


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
            NormalizeGeomNames(),
            ReplaceCylinderWithCapsule(),
            RenameBodies(
                {
                    "arm_link7": "gripper_right",
                    "arm_link8": "gripper_left",
                    "arm_link6": "gripper_base",
                }
            ),
            AddDummyBody(
                parent_path="gripper_base",
                name="grasp_point",
                pos=(0.05, 0.0, 0.0),
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
    preferred_names: list[str],
    force_download: bool,
) -> Path:
    if local is not None:
        path = local.expanduser().resolve()
        if path.is_dir():
            return find_mjcf(path, preferred_names=preferred_names)
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
    return find_mjcf(dest, preferred_names=preferred_names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an A2 + Piper MJCF asset. "
            "If --a2/--piper are omitted, download vendor folders into artifacts/vendor/."
        )
    )
    parser.add_argument(
        "--a2",
        type=Path,
        default=None,
        help="Local A2 MJCF file or directory (default: download unitree a2_description).",
    )
    parser.add_argument(
        "--piper",
        type=Path,
        default=None,
        help="Local Piper MJCF file or directory (default: download mujoco_menagerie agilex_piper).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download vendor assets even if artifacts/vendor/ already has them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ARTIFACTS / "a2_piper",
        help="Output directory for the assembled model.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip the MuJoCo viewer after export.",
    )
    args = parser.parse_args()

    a2_xml = _resolve_mjcf(
        args.a2,
        github_url=A2_GITHUB,
        vendor_dirname="a2_description",
        preferred_names=["a2.xml"],
        force_download=args.force_download,
    )
    piper_xml = _resolve_mjcf(
        args.piper,
        github_url=PIPER_GITHUB,
        vendor_dirname="agilex_piper",
        preferred_names=["piper.xml"],
        force_download=args.force_download,
    )
    print(f"Using A2: {a2_xml}")
    print(f"Using Piper: {piper_xml}")

    robot = build_a2_piper(load_a2(a2_xml), load_piper(piper_xml))
    saved = robot.save(args.output)
    print(saved.xml_path)

    if args.no_viewer:
        return

    from assetx import launch_preview

    launch_preview(robot)


if __name__ == "__main__":
    main()
