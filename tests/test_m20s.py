"""M20S example: scene cleanup and wheel cylinders left intact."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import pytest

import assetx
from assetx.fetch import parse_github_dir_url
from assetx.vendor import cache_dir


def _load_example():
    path = Path(assetx.__file__).resolve().parent.parent.parent / "examples" / "m20s.py"
    spec = importlib.util.spec_from_file_location("assetx_example_m20s", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m20s_keeps_wheel_cylinders() -> None:
    m20s = _load_example()
    ref = parse_github_dir_url(m20s.DEEP_M20S)
    vendor = cache_dir() / "vendor" / f"{ref.owner}__{ref.repo}__{ref.ref}" / ref.path
    if not (vendor / m20s.M20S_MJCF).is_file():
        pytest.skip(f"M20S vendor asset not cached at {vendor}")

    asset = m20s.build_m20s(vendor)
    free = [
        joint
        for joint in asset.spec.joints
        if joint.type == mujoco.mjtJoint.mjJNT_FREE
    ]
    assert [joint.name for joint in free] == ["floating_base_joint"]
    assert list(asset.spec.actuators) == []
    assert list(asset.spec.sensors) == []
    assert asset.spec.geom("floor") is None

    for leg in ("fl", "fr", "hl", "hr"):
        wheel = asset.spec.geom(f"{leg}_wheel_collision0")
        assert wheel is not None
        assert wheel.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        hip = asset.spec.geom(f"{leg}_hipx_collision0")
        knee = asset.spec.geom(f"{leg}_knee_collision1")
        assert hip is not None and hip.type == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert knee is not None and knee.type == mujoco.mjtGeom.mjGEOM_CAPSULE
    rail = asset.spec.geom("base_link_collision1")
    assert rail is not None and rail.type == mujoco.mjtGeom.mjGEOM_CAPSULE
