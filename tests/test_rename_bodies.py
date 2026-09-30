"""Tests for ``RenameBodies`` reference propagation."""

from __future__ import annotations

from pathlib import Path

from assetx import MujocoAsset, RenameBodies

_XML = """<mujoco model="robot">
  <worldbody>
    <body name="body">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <camera name="cam" mode="targetbody" target="leg"/>
      <body name="leg" pos="0 0 -0.2">
        <joint name="hip" type="hinge"/>
        <geom type="capsule" size="0.02 0.1"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="body" body2="leg"/>
  </contact>
  <equality>
    <weld body1="leg" body2="body" active="false"/>
  </equality>
  <sensor>
    <framepos name="leg_pos" objtype="body" objname="leg" reftype="body" refname="body"/>
    <subtreecom name="com" body="body"/>
  </sensor>
</mujoco>
"""


def test_rename_updates_references(tmp_path: Path) -> None:
    xml = tmp_path / "robot.xml"
    xml.write_text(_XML)
    asset = MujocoAsset.from_file(xml)

    out = RenameBodies({"body": "base_link", "leg": "thigh"}).transform(asset)
    spec = out.spec

    assert {b.name for b in spec.bodies} >= {"base_link", "thigh"}
    ex = spec.excludes[0]
    assert (ex.bodyname1, ex.bodyname2) == ("base_link", "thigh")
    eq = spec.equalities[0]
    assert (eq.name1, eq.name2) == ("thigh", "base_link")
    framepos, subtreecom = spec.sensors
    assert (framepos.objname, framepos.refname) == ("thigh", "base_link")
    assert subtreecom.objname == "base_link"
    assert spec.cameras[0].targetbody == "thigh"
