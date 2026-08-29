"""Tests for EditBodies."""

from __future__ import annotations

from pathlib import Path

import mujoco

from assetx import EditBodies, MujocoAsset


def test_edit_bodies_clears_gravcomp() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="arm_link1", gravcomp=1)
    spec.worldbody.add_body(name="base_link", gravcomp=1)
    asset = MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))

    out = EditBodies("arm_link.*", gravcomp=0).transform(asset)

    assert float(out.spec.body("arm_link1").gravcomp) == 0.0
    assert float(out.spec.body("base_link").gravcomp) == 1.0
    xml = out.spec.to_xml()
    assert 'name="arm_link1"' in xml
    assert 'name="arm_link1" gravcomp' not in xml
    assert 'name="base_link" gravcomp="1"' in xml


def test_edit_bodies_requires_attribute() -> None:
    try:
        EditBodies("arm_link.*")
    except ValueError as exc:
        assert "attribute" in str(exc)
    else:
        raise AssertionError("expected ValueError when no attrs given")
