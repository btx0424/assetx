"""Tests for preview ground / float helpers and Viser rig joints."""

from __future__ import annotations

import math

import mujoco
import pytest

from assetx.core.preview import (
    _body_frame_pose,
    _float_above_ground,
    _geom_rgba_u8,
    _lowest_non_plane_z,
    apply_rig_joint_values,
    compile_for_preview,
    list_rig_joints,
)


def _make_floating_box_spec(*, z: float = 0.5) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base", pos=(0.0, 0.0, z))
    body.add_freejoint(name="freejoint")
    geom = body.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 0.0, 0.0),
    )
    del geom
    return spec


def _make_arm_spec() -> mujoco.MjSpec:
    # MjSpec joint ranges default to degrees unless compiler.angle is set.
    spec = mujoco.MjSpec()
    base = spec.worldbody.add_body(name="base")
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.05))
    link = base.add_body(name="link", pos=(0.0, 0.0, 0.1))
    joint = link.add_joint(
        name="shoulder",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0.0, 1.0, 0.0),
        range=(-90.0, 90.0),
    )
    joint.limited = True
    link.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.02, 0.08),
        pos=(0.0, 0.0, 0.08),
    )
    return spec


def test_compile_for_preview_ground_and_lighting() -> None:
    model = compile_for_preview(
        _make_floating_box_spec(),
        lighting=True,
        ground=True,
    )
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "preview_floor")
    assert floor_id >= 0
    assert model.nlight >= 1


def test_float_above_ground_lifts_freejoint() -> None:
    clearance = 0.02
    model = compile_for_preview(
        _make_floating_box_spec(z=0.05),
        lighting=False,
        ground=True,
    )
    data = mujoco.MjData(model)
    _float_above_ground(model, data, clearance=clearance)
    z_min = _lowest_non_plane_z(model, data)
    assert abs(z_min - clearance) < 1e-5


def test_float_above_ground_moves_floor_for_fixed_base() -> None:
    clearance = 0.015
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base", pos=(0.0, 0.0, 0.2))
    body.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
    )
    model = compile_for_preview(spec, lighting=False, ground=True)
    data = mujoco.MjData(model)
    _float_above_ground(model, data, clearance=clearance)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "preview_floor")
    z_min = _lowest_non_plane_z(model, data)
    floor_z = float(model.geom_pos[floor_id, 2])
    assert abs((z_min - floor_z) - clearance) < 1e-5
    assert model.nq == 0


def test_list_rig_joints_skips_freejoint() -> None:
    model = compile_for_preview(_make_floating_box_spec(), lighting=False, ground=False)
    assert list_rig_joints(model) == []


def test_list_and_apply_rig_joints() -> None:
    model = compile_for_preview(_make_arm_spec(), lighting=False, ground=False)
    joints = list_rig_joints(model)
    assert len(joints) == 1
    assert joints[0].name == "shoulder"
    assert joints[0].kind == "hinge"
    assert joints[0].lower == pytest.approx(-0.5 * math.pi)
    assert joints[0].upper == pytest.approx(0.5 * math.pi)

    data = mujoco.MjData(model)
    apply_rig_joint_values(model, data, {"shoulder": 0.5}, joints)
    assert float(data.qpos[joints[0].qpos_adr]) == pytest.approx(0.5)

    # Out-of-range values are clipped.
    apply_rig_joint_values(model, data, {"shoulder": 5.0}, joints)
    assert float(data.qpos[joints[0].qpos_adr]) == pytest.approx(joints[0].upper)


def test_body_frame_pose_link_vs_com() -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base", pos=(1.0, 0.0, 0.5))
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.1, 0.1, 0.1), pos=(0.2, 0.0, 0.0)
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    link_pos, link_quat = _body_frame_pose(data, 1, "link")
    com_pos, _ = _body_frame_pose(data, 1, "com")
    assert link_pos == pytest.approx([1.0, 0.0, 0.5])
    assert link_quat == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert com_pos == pytest.approx([1.2, 0.0, 0.5])


def test_geom_rgba_prefers_material() -> None:
    spec = mujoco.MjSpec()
    mat = spec.add_material(name="red")
    mat.rgba = (1.0, 0.0, 0.0, 1.0)
    body = spec.worldbody.add_body(name="base")
    body.add_geom(name="with_mat", size=(0.1, 0, 0), material="red")
    body.add_geom(name="plain", size=(0.1, 0, 0), rgba=(0.0, 0.0, 1.0, 0.5))
    model = spec.compile()

    assert _geom_rgba_u8(model, 0) == (255, 0, 0, 1.0)
    assert _geom_rgba_u8(model, 1) == (0, 0, 255, 0.5)
