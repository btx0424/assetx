"""Tests for ``RemoveSceneSettings``."""

from __future__ import annotations

from pathlib import Path

import mujoco

from assetx import MujocoAsset, RemoveSceneSettings

_XML = """<mujoco model="robot">
  <option impratio="100" integrator="implicitfast" cone="elliptic" timestep="0.004"/>
  <visual><global ellipsoidinertia="true"/></visual>
  <worldbody>
    <light name="sun" pos="0 0 3"/>
    <body name="base">
      <light name="lamp" pos="0 0 1"/>
      <geom name="g" type="box" size="0.1 0.1 0.1" solref="0.004"/>
    </body>
  </worldbody>
</mujoco>
"""


def _asset(tmp_path: Path) -> MujocoAsset:
    xml = tmp_path / "robot.xml"
    xml.write_text(_XML)
    return MujocoAsset.from_file(xml)


def test_removes_lights_and_option(tmp_path: Path) -> None:
    out = RemoveSceneSettings().transform(_asset(tmp_path))
    xml = out.spec.to_xml()

    assert "<light" not in xml
    assert "<option" not in xml
    assert "ellipsoidinertia" in xml
    assert 'solref="0.004"' in xml

    ref = mujoco.MjSpec().compile().opt
    opt = out.spec.compile().opt
    assert opt.integrator == ref.integrator
    assert opt.cone == ref.cone
    assert opt.impratio == ref.impratio
    assert opt.timestep == ref.timestep


def test_visual_and_selective(tmp_path: Path) -> None:
    asset = _asset(tmp_path)

    xml = RemoveSceneSettings(visual=True).transform(asset).spec.to_xml()
    assert "ellipsoidinertia" not in xml

    xml = RemoveSceneSettings(option=False).transform(asset).spec.to_xml()
    assert "<light" not in xml
    assert 'impratio="100"' in xml
