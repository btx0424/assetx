"""Tests for DAE→USDA visual bake and postprocess swap."""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from assetx.conversion.urdf2mjcf import urdf_to_mjcf
from assetx.conversion.usd.postprocess import prefer_visual_meshes
from assetx.conversion.usd.visual_meshes import (
    bake_visual_mesh_sources,
    load_visual_mesh_sources,
    sources_path_for_mjcf,
    write_dae_as_usda,
    write_visual_mesh_sources,
)


def _tiny_dae(path: Path, *, rgba=(10, 20, 30, 255)) -> None:
    r, g, b, a = (c / 255.0 for c in rgba)
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="fx"><profile_COMMON>
      <technique sid="common"><phong>
        <diffuse><color>{r} {g} {b} {a}</color></diffuse>
      </phong></technique>
    </profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="mat" name="mat"><instance_effect url="#fx"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="geom" name="geom"><mesh>
      <source id="pos"><float_array id="pos-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
        <technique_common><accessor source="#pos-array" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="verts"><input semantic="POSITION" source="#pos"/></vertices>
      <triangles count="1" material="mat">
        <input semantic="VERTEX" source="#verts" offset="0"/>
        <p>0 1 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="node" name="node">
        <instance_geometry url="#geom"><bind_material>
          <technique_common>
            <instance_material symbol="mat" target="#mat"/>
          </technique_common>
        </bind_material></instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
""",
        encoding="utf-8",
    )


def test_write_and_load_sources(tmp_path: Path) -> None:
    dae = tmp_path / "part.dae"
    dae.write_text("<collada/>", encoding="utf-8")
    stl = tmp_path / "part.stl"
    path = tmp_path / "visual_mesh_sources.json"
    write_visual_mesh_sources(path, {"part": dae, "other": stl})
    loaded = load_visual_mesh_sources(path)
    assert set(loaded) == {"part"}
    assert loaded["part"] == dae.resolve()


def test_bake_dae_preserves_display_color(tmp_path: Path) -> None:
    dae = tmp_path / "link.dae"
    _tiny_dae(dae, rgba=(51, 102, 153, 255))
    usda = tmp_path / "link.usda"
    write_dae_as_usda(dae, usda, root_name="link")
    stage = Usd.Stage.Open(str(usda))
    assert stage is not None
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    color = meshes[0].GetAttribute("primvars:displayColor").Get()
    assert color is not None
    assert abs(color[0][0] - 0.2) < 1e-3
    assert abs(color[0][1] - 0.4) < 1e-3
    assert abs(color[0][2] - 0.6) < 1e-3


def test_prefer_visual_meshes_swaps_under_visuals(tmp_path: Path) -> None:
    dae = tmp_path / "calf.dae"
    _tiny_dae(dae, rgba=(0, 0, 0, 255))
    mesh_dir = tmp_path / "visual_meshes"
    bake_visual_mesh_sources({"calf": dae}, mesh_dir)

    usd = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(usd))
    robot = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(robot.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/robot/FL_calf")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdGeom.Xform.Define(stage, "/robot/FL_calf/visuals")
    mesh = UsdGeom.Mesh.Define(stage, "/robot/FL_calf/visuals/calf")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.0, 0.0))
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    mesh.GetDisplayColorAttr().Set([(1.0, 1.0, 1.0)])
    stage.GetRootLayer().Save()
    del stage

    assert prefer_visual_meshes(usd, mesh_dir) == 1
    stage = Usd.Stage.Open(str(usd))
    prim = stage.GetPrimAtPath("/robot/FL_calf/visuals/calf")
    assert prim.GetTypeName() == "Xform"
    assert not prim.IsA(UsdGeom.Mesh)
    refs = prim.GetMetadata("references")
    assert refs is not None
    translate = UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()
    assert Gf.IsClose(translate, Gf.Vec3d(0.1, 0.0, 0.0), 1e-6)
    composed = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    assert composed
    dc = composed[0].GetAttribute("primvars:displayColor").Get()
    assert dc is not None
    assert dc[0][0] == 0.0


def _box_dae(path: Path) -> None:
    import trimesh

    mesh = trimesh.creation.box((0.1, 0.1, 0.1))
    mesh.visual.face_colors = [10, 20, 30, 255]
    mesh.export(path)


def _visual_names(spec) -> dict[str, str]:
    names: dict[str, str] = {}
    for body in spec.bodies:
        if body.name == "world":
            continue
        for geom in body.geoms:
            if int(geom.contype) or int(geom.conaffinity) or not geom.meshname:
                continue
            names[body.name] = geom.name or geom.meshname
    return names


def test_dae_sources_follow_visual_prim_names(tmp_path: Path) -> None:
    thigh = tmp_path / "thigh.dae"
    calf = tmp_path / "calf.dae"
    _box_dae(thigh)
    _box_dae(calf)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="r">
  <link name="thigh">
    <visual name="calf_vis">
      <geometry><mesh filename="{thigh}"/>
      </geometry>
    </visual>
  </link>
  <link name="calf">
    <visual name="visual">
      <geometry><mesh filename="{calf}"/>
      </geometry>
    </visual>
  </link>
  <joint name="knee" type="fixed">
    <parent link="thigh"/>
    <child link="calf"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    spec, _model, xml_out = urdf_to_mjcf(urdf)
    sources = load_visual_mesh_sources(sources_path_for_mjcf(xml_out))
    names = _visual_names(spec)
    assert names["thigh"] == "calf_vis"
    assert names["calf"] == "visual"
    assert sources["calf_vis"].name == "thigh.dae"
    assert sources["visual"].name == "calf.dae"


def test_shared_mesh_visual_does_not_take_collision_prim_name(tmp_path: Path) -> None:
    dae = tmp_path / "calf.dae"
    _box_dae(dae)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        f"""<?xml version="1.0"?>
<robot name="r">
  <link name="base">
    <visual>
      <geometry><mesh filename="{dae}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{dae}"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    spec, _model, xml_out = urdf_to_mjcf(urdf)
    sources = load_visual_mesh_sources(sources_path_for_mjcf(xml_out))
    visual = _visual_names(spec)["base"]
    assert visual == "calf_visual"
    assert sources[visual].name == "calf.dae"
    collision_names = [
        geom.name or geom.meshname
        for geom in spec.body("base").geoms
        if int(geom.contype) or int(geom.conaffinity)
    ]
    assert "calf" in collision_names
    assert visual not in collision_names
