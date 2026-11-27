"""AS2W rounded-wheel metadata from the tire meshes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import pytest

import assetx
from assetx.fetch import parse_github_dir_url
from assetx.vendor import cache_dir


def _load_example():
    path = Path(assetx.__file__).resolve().parent.parent.parent / "examples" / "as2w.py"
    spec = importlib.util.spec_from_file_location("assetx_example_as2w", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_as2w_records_rounded_wheels_and_keeps_sphere_collision() -> None:
    as2w = _load_example()
    ref = parse_github_dir_url(as2w.UNITREE_AS2W)
    vendor = cache_dir() / "vendor" / f"{ref.owner}__{ref.repo}__{ref.ref}" / ref.path
    if not (vendor / as2w.AS2W_URDF).is_file():
        pytest.skip(f"AS2W vendor asset not cached at {vendor}")

    asset = as2w.build_as2w(vendor)
    fits = {fit.body: fit for fit in asset.metadata["rounded_wheels"]}
    assert set(fits) == {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
    for leg in ("FL", "FR", "RL", "RR"):
        body = f"{leg}_foot"
        fit = fits[body]
        assert fit.geoms == (f"{body}_collision0",)
        assert fit.radius == pytest.approx(0.086, abs=1e-4)
        assert fit.half_width == pytest.approx(0.0259, abs=1e-4)
        assert fit.axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
        sign = 1.0 if leg.endswith("L") else -1.0
        assert fit.center[1] == pytest.approx(sign * 0.0546, abs=1e-4)
        assert fit.center[0] == pytest.approx(0.0, abs=1e-5)
        assert fit.center[2] == pytest.approx(0.0, abs=1e-5)
        sphere = asset.spec.geom(f"{body}_collision0")
        assert sphere is not None
        assert int(sphere.type) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
