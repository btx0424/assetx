"""``MujocoAsset.save`` must not turn class-inherited capsules into spheres."""

from __future__ import annotations

from pathlib import Path

import mujoco

from assetx import MujocoAsset

_XML = """\
<mujoco model="shapes">
  <default>
    <geom type="box"/>
    <default class="collision">
      <geom type="capsule"/>
      <default class="foot_capsule">
        <geom size="0.01"/>
      </default>
    </default>
  </default>
  <worldbody>
    <body name="base">
      <geom name="block" size="0.1 0.2 0.3"/>
      <geom name="ball" class="collision" type="sphere" size="0.06"/>
      <geom name="limb" class="collision" size="0.04" fromto="0 0 0 0 0 0.2"/>
      <geom name="foot" class="foot_capsule" fromto="0 0 0 0.1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_save_keeps_inherited_capsules(tmp_path: Path) -> None:
    xml = tmp_path / "src" / "shapes.xml"
    xml.parent.mkdir()
    xml.write_text(_XML)
    saved = MujocoAsset.from_file(xml).save(tmp_path / "out", save_urdf=False)

    model = mujoco.MjModel.from_xml_path(str(saved.xml_path))
    types = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): int(model.geom_type[i])
        for i in range(model.ngeom)
    }
    assert types["block"] == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert types["ball"] == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert types["limb"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert types["foot"] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    # Half-length survives the fromto -> pos/quat rewrite (0.2 / 2).
    limb = model.geom(model.geom("limb").id)
    assert abs(float(limb.size[1]) - 0.1) < 1e-6
