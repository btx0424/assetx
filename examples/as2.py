from __future__ import annotations

import argparse
from pathlib import Path

from assetx import (
    AddDummyBody,
    ApproximateWithCapsule,
    Compose,
    EditBodies,
    GeomsToBody,
    MujocoAsset,
    NormalizeGeomNames,
    RemoveActuators,
    RemoveSensors,
    RenameBodies,
    ReplaceCylinderWithCapsule,
    assemble,
    asset_builder,
)

from assetx.fetch import download_github_dir

_ARTIFACTS = Path("artifacts")
_VENDOR = _ARTIFACTS / "vendor"

AS2_GITHUB = (
    "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/as2_description"
)

AS2_FILE = "as2.xml"


@asset_builder
def load_as2(xml_path: str | Path) -> MujocoAsset:
    asset = MujocoAsset.from_file(xml_path)
    transform = Compose([
        NormalizeGeomNames(),
        RemoveSensors(names=[".*pos", ".*torque", "imu.*", ".*vel"]),
        RemoveActuators(names=[".*"]),
        GeomsToBody(["FL_calf_collision3", "FL_calf_visual1"], "FL_foot", mass=0.05),
        GeomsToBody(["FR_calf_collision3", "FR_calf_visual1"], "FR_foot", mass=0.05),
        GeomsToBody(["RL_calf_collision3", "RL_calf_visual1"], "RL_foot", mass=0.05),
        GeomsToBody(["RR_calf_collision3", "RR_calf_visual1"], "RR_foot", mass=0.05),
    ])
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
        description="Build an AS2 MJCF asset."
    )
    parser.add_argument("--as2", type=Path, default=None, help="Local AS2 MJCF file (default: download unitree as2_description).")
    parser.add_argument("--output", type=Path, default=_ARTIFACTS / "as2")
    parser.add_argument("--force-download", action="store_true", help="Force download the vendor assets.")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the viewer.")
    args = parser.parse_args()
    
    as2_xml = _resolve_mjcf(
        args.as2,
        github_url=AS2_GITHUB,
        vendor_dirname="as2_description",
        relative_xml=AS2_FILE,
        force_download=args.force_download,
    )
    print(f"Using AS2: {as2_xml}")
    robot = load_as2(as2_xml)
    saved = robot.save(args.output)
    print(saved.xml_path)

    if args.no_viewer:
        return

    from assetx import launch_preview

    launch_preview(robot)

if __name__ == "__main__":
    main()