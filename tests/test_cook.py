"""Tests for recipe registration and ``assetx.cook`` (no network, no USD)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from assetx import MujocoAsset
from assetx import cook as cook_mod
from assetx.recipes import registry

_SHA = "0123456789abcdef0123456789abcdef01234567"
_VENDOR = f"https://github.com/org/repo/tree/{_SHA}/robot"
_XML = """
<mujoco model="toy">
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="leg" pos="0 0 -0.2">
        <joint name="knee" axis="0 1 0"/>
        <geom type="capsule" size="0.02 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def toy_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "toy.xml").write_text(_XML)
    monkeypatch.setattr(registry, "fetch_vendor", lambda url: vendor)
    monkeypatch.setattr(registry, "_RECIPES", dict(registry._RECIPES))

    @registry.recipe("toy", vendor=_VENDOR)
    def toy(vendor_dir: Path) -> MujocoAsset:
        return MujocoAsset.from_file(vendor_dir / "toy.xml")

    return "toy"


def test_recipe_requires_pinned_sha() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        registry.recipe("bad", vendor="https://github.com/org/repo/tree/main/robot")


def test_cook_writes_manifest_and_skips_when_fresh(toy_recipe: str, tmp_path: Path) -> None:
    out = tmp_path / "models" / "toy"
    assert cook_mod.check(toy_recipe, out, usd=False) == "missing"

    cook_mod.cook(toy_recipe, out, usd=False)
    manifest = json.loads((out / cook_mod.MANIFEST).read_text())
    assert manifest["recipe"] == "toy"
    assert manifest["formats"] == ["mjcf", "urdf"]
    assert (out / "model.xml").is_file() and (out / "model.urdf").is_file()
    assert cook_mod.check(toy_recipe, out, usd=False) is None
    assert cook_mod.check(toy_recipe, out, usd=True) == "missing USD"
    assert not list(out.parent.glob(".toy.*/"))

    mtime = (out / "model.xml").stat().st_mtime_ns
    cook_mod.cook(toy_recipe, out, usd=False)
    assert (out / "model.xml").stat().st_mtime_ns == mtime


def test_check_detects_stale_and_foreign_bundles(toy_recipe: str, tmp_path: Path) -> None:
    out = tmp_path / "toy"
    cook_mod.cook(toy_recipe, out, usd=False)
    manifest_path = out / cook_mod.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert cook_mod.check(toy_recipe, out, usd=False).startswith("stale")

    cook_mod.cook(toy_recipe, out, usd=False)
    assert cook_mod.check(toy_recipe, out, usd=False) is None

    manifest_path.unlink()
    assert "no assetx.json" in cook_mod.check(toy_recipe, out, usd=False)


def test_hash_ignores_other_recipes_and_preview() -> None:
    files = {p.relative_to(cook_mod._PACKAGE_ROOT).as_posix() for p in cook_mod._library_files()}
    assert "core/transforms/edit.py" in files
    assert "conversion/usd/newton.py" in files
    assert "recipes/registry.py" in files
    assert "recipes/spot.py" not in files
    assert "core/preview.py" not in files and "cli.py" not in files


def test_usd_export_imports_without_pxr() -> None:
    code = (
        "import sys; sys.modules['pxr'] = None\n"
        "import assetx.cook, assetx.conversion.usd.newton\n"
        "from assetx.conversion.usd import mjcf_to_usd\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
