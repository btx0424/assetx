"""Tests for ``MujocoAsset.save(mesh_format=...)``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from assetx import MujocoAsset

_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


def _write_asset(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "meshes").mkdir(parents=True)
    (src / "meshes" / "tet.obj").write_text(_OBJ)
    xml = src / "robot.xml"
    xml.write_text(
        """<mujoco model="robot">
  <compiler meshdir="meshes"/>
  <asset><mesh name="tet" file="tet.obj" content_type="model/obj"/></asset>
  <worldbody>
    <body name="base">
      <geom name="vis" type="mesh" mesh="tet" contype="0" conaffinity="0" group="2"/>
      <geom name="col" type="mesh" mesh="tet" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return xml


def test_save_converts_meshes_to_stl(tmp_path: Path) -> None:
    asset = MujocoAsset.from_file(_write_asset(tmp_path))
    ref_vert = mujoco.MjModel.from_xml_path(str(asset.xml_path)).mesh_vert.copy()

    saved = asset.save(tmp_path / "out", mesh_format="stl")

    stl_files = list((tmp_path / "out" / "meshes").rglob("*.stl"))
    assert [p.name for p in stl_files] == ["tet.stl"]
    assert not list((tmp_path / "out" / "meshes").rglob("*.obj"))
    assert trimesh.load(stl_files[0]).faces.shape == (4, 3)

    xml_text = saved.xml_path.read_text()
    assert "tet.stl" in xml_text and "model/obj" not in xml_text
    urdf_text = saved.xml_path.with_suffix(".urdf").read_text()
    assert "tet.stl" in urdf_text and "tet.obj" not in urdf_text

    new_vert = mujoco.MjModel.from_xml_path(str(saved.xml_path)).mesh_vert
    np.testing.assert_allclose(
        np.sort(new_vert, axis=0), np.sort(ref_vert, axis=0), atol=1e-6
    )


def test_urdf_visual_materials(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    xml = src / "robot.xml"
    xml.write_text(
        """<mujoco model="robot">
  <asset><material name="orange" rgba="1 0.5 0 1"/></asset>
  <worldbody>
    <body name="base">
      <geom name="a" type="box" size="0.1 0.1 0.1" material="orange"
            contype="0" conaffinity="0"/>
      <geom name="b" type="sphere" size="0.1" rgba="0 0 1 1"
            contype="0" conaffinity="0"/>
      <geom name="c" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    saved = MujocoAsset.from_file(xml).save(tmp_path / "out")
    root = ET.parse(saved.xml_path.with_suffix(".urdf")).getroot()

    defs = {m.get("name"): m.find("color").get("rgba") for m in root.findall("material")}
    assert defs == {"orange": "1 0.5 0 1", "rgba_0_0_255_255": "0 0 1 1"}
    visual_mats = [v.find("material").get("name") for v in root.iter("visual")]
    assert visual_mats == ["orange", "rgba_0_0_255_255"]
    assert all(c.find("material") is None for c in root.iter("collision"))


def test_save_mesh_format_requires_copy(tmp_path: Path) -> None:
    asset = MujocoAsset.from_file(_write_asset(tmp_path))
    with pytest.raises(ValueError, match="copy_meshes"):
        asset.save(tmp_path / "out", mesh_format="stl", copy_meshes=False)
