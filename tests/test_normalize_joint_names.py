"""Tests for ``NormalizeJointNames``."""

from __future__ import annotations

from pathlib import Path

import pytest

from assetx import MujocoAsset, NormalizeJointNames


def _asset(tmp_path: Path, root_joints: str, extra: str = "") -> MujocoAsset:
    xml = tmp_path / "robot.xml"
    xml.write_text(
        f"""<mujoco model="robot">
  <worldbody>
    <body name="base">
      {root_joints}
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="leg" pos="0 0 -0.2">
        <joint name="hip" type="hinge"/>
        <geom type="capsule" size="0.02 0.1"/>
      </body>
    </body>
  </worldbody>
  {extra}
</mujoco>
"""
    )
    return MujocoAsset.from_file(xml)


def test_renames_root_freejoint_and_sensor(tmp_path: Path) -> None:
    asset = _asset(
        tmp_path,
        '<freejoint name="freejoint"/>',
        '<sensor><jointpos name="p" joint="hip"/>'
        '<framepos name="f" objtype="body" objname="base"/></sensor>',
    )
    out = NormalizeJointNames().transform(asset)
    names = [j.name for j in out.spec.joints]
    assert names == ["floating_base_joint", "hip"]


def test_unnamed_freejoint_and_custom_name(tmp_path: Path) -> None:
    asset = _asset(tmp_path, "<freejoint/>")
    out = NormalizeJointNames(root_joint="root").transform(asset)
    assert out.spec.joints[0].name == "root"


def test_fixed_base_unchanged(tmp_path: Path) -> None:
    asset = _asset(tmp_path, "")
    out = NormalizeJointNames().transform(asset)
    assert [j.name for j in out.spec.joints] == ["hip"]


def test_name_collision_raises(tmp_path: Path) -> None:
    asset = _asset(tmp_path, '<freejoint name="fj"/>')
    with pytest.raises(ValueError, match="already exists"):
        NormalizeJointNames(root_joint="hip").transform(asset)
