"""URDF inertias that MuJoCo rejects are repaired before compile."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco

from assetx.conversion.urdf2mjcf import prepare_urdf_for_mujoco


def _link(name: str, inertia: str, mass: str = "1") -> str:
    return f"""
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia {inertia}/>
    </inertial>
  </link>
"""


def _robot(*links: str) -> str:
    return '<?xml version="1.0"?><robot name="r">' + "".join(links) + "</robot>"


def _inertia(xml: str, link: str) -> ET.Element:
    root = ET.fromstring(xml)
    found = next(elem for elem in root.iter("link") if elem.get("name") == link)
    inertia = found.find("inertial/inertia")
    assert inertia is not None
    return inertia


def test_tiny_inertia_compiles() -> None:
    raw = _robot(
        _link(
            "lidar_link",
            'ixx="1.68E-27" ixy="-3.2E-45" ixz="7.68E-44" '
            'iyy="1.68E-27" iyz="-9.26E-47" izz="1.68E-27"',
            mass="4.19E-15",
        )
    )
    prepared = prepare_urdf_for_mujoco(raw)
    inertia = _inertia(prepared, "lidar_link")
    assert float(inertia.get("ixx")) >= 1e-14
    assert float(inertia.get("iyy")) >= 1e-14
    assert float(inertia.get("izz")) >= 1e-14
    mujoco.MjSpec.from_string(prepared).compile()


def test_triangle_inequality_compiles() -> None:
    raw = _robot(
        _link(
            "bad",
            'ixx="0.01" ixy="0.009" ixz="0" iyy="0.01" iyz="0" izz="0.01"',
        )
    )
    prepared = prepare_urdf_for_mujoco(raw)
    mujoco.MjSpec.from_string(prepared).compile()


def test_valid_inertia_is_unchanged() -> None:
    attrs = 'ixx="0.27466" ixy="-0.000622" ixz="0.00315" iyy="1.0618" iyz="-0.00139" izz="1.1825"'
    raw = _robot(_link("base_link", attrs, mass="35.86"))
    inertia = _inertia(prepare_urdf_for_mujoco(raw), "base_link")
    assert inertia.get("ixx") == "0.27466"
    assert inertia.get("ixy") == "-0.000622"
    assert inertia.get("izz") == "1.1825"
