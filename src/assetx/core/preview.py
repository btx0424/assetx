"""MuJoCo / Viser viewer helpers for interactive previews (never written to disk)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import mujoco.viewer
import numpy as np
import trimesh

if TYPE_CHECKING:
    from assetx.core.asset import MujocoAsset

_DEFAULT_FLOAT_CLEARANCE = 0.01
_DEFAULT_VISER_PORT = 8080
_UNLIMITED_HINGE_RANGE = (-np.pi, np.pi)
_UNLIMITED_SLIDE_RANGE = (-1.0, 1.0)


def add_preview_light(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add a directional key light to ``spec`` (mutates and returns it)."""
    light = spec.worldbody.add_light()
    light.name = "preview_key"
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = (0.0, 0.0, 3.0)
    light.dir = (0.25, 0.25, -1.0)
    light.diffuse = (0.9, 0.9, 0.9)
    light.specular = (0.3, 0.3, 0.3)
    light.castshadow = True
    return spec


def add_preview_ground(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add a checkerboard ground plane to ``spec`` (mutates and returns it)."""
    if not any(t.name == "preview_skybox" for t in spec.textures):
        sky = spec.add_texture(name="preview_skybox")
        sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1 = (0.3, 0.5, 0.7)
        sky.rgb2 = (0.0, 0.0, 0.0)
        sky.width = 512
        sky.height = 3072

    if not any(t.name == "preview_groundplane" for t in spec.textures):
        tex = spec.add_texture(name="preview_groundplane")
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.mark = mujoco.mjtMark.mjMARK_EDGE
        tex.rgb1 = (0.2, 0.3, 0.4)
        tex.rgb2 = (0.1, 0.2, 0.3)
        tex.markrgb = (0.8, 0.8, 0.8)
        tex.width = 300
        tex.height = 300

    if not any(m.name == "preview_groundplane" for m in spec.materials):
        mat = spec.add_material(name="preview_groundplane")
        mat.textures[0] = "preview_groundplane"
        mat.texuniform = True
        mat.texrepeat = (5.0, 5.0)
        mat.reflectance = 0.2

    floor = spec.worldbody.add_geom()
    floor.name = "preview_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = (0.0, 0.0, 0.05)
    floor.material = "preview_groundplane"
    return spec


def compile_for_preview(
    spec: mujoco.MjSpec,
    *,
    lighting: bool = True,
    ground: bool = True,
) -> mujoco.MjModel:
    """Compile a copy of ``spec`` with optional preview lighting / ground.

    The original ``spec`` is left unchanged.
    """
    preview = spec.copy()
    if lighting:
        add_preview_light(preview)
    if ground:
        add_preview_ground(preview)
    return preview.compile()


def _lowest_non_plane_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Conservative world-frame lowest bound of non-plane geoms."""
    z_min = float("inf")
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        z_min = min(z_min, float(data.geom_xpos[gid, 2] - model.geom_rbound[gid]))
    if not np.isfinite(z_min):
        raise ValueError("preview model has no non-plane geoms to measure height from")
    return z_min


def _freejoint_qposadr(model: mujoco.MjModel) -> int | None:
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid])
    return None


def _float_above_ground(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    clearance: float,
) -> None:
    """Lift the freejoint (or drop the floor) so the robot floats by ``clearance``."""
    mujoco.mj_forward(model, data)
    z_min = _lowest_non_plane_z(model, data)
    delta = float(clearance) - z_min
    if abs(delta) < 1e-9:
        return

    free_adr = _freejoint_qposadr(model)
    if free_adr is not None:
        data.qpos[free_adr + 2] = float(data.qpos[free_adr + 2]) + delta
        mujoco.mj_forward(model, data)
        return

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "preview_floor")
    if floor_id < 0:
        return
    # Fixed-base robots: keep the robot pose and place the floor under it.
    model.geom_pos[floor_id, 2] = float(model.geom_pos[floor_id, 2]) - delta
    mujoco.mj_forward(model, data)


@dataclass(frozen=True)
class RigJoint:
    """Scalar joint exposed as a Viser slider."""

    joint_id: int
    name: str
    qpos_adr: int
    lower: float
    upper: float
    step: float
    kind: Literal["hinge", "slide"]


def list_rig_joints(model: mujoco.MjModel) -> list[RigJoint]:
    """Return hinge/slide joints suitable for kinematic sliders."""
    joints: list[RigJoint] = []
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        if jtype == int(mujoco.mjtJoint.mjJNT_HINGE):
            kind: Literal["hinge", "slide"] = "hinge"
            default_lo, default_hi = _UNLIMITED_HINGE_RANGE
            step = 0.01
        elif jtype == int(mujoco.mjtJoint.mjJNT_SLIDE):
            kind = "slide"
            default_lo, default_hi = _UNLIMITED_SLIDE_RANGE
            step = 0.001
        else:
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        if bool(model.jnt_limited[jid]):
            lower, upper = (float(x) for x in model.jnt_range[jid])
        else:
            lower, upper = float(default_lo), float(default_hi)
        if upper <= lower:
            upper = lower + 1e-3
        joints.append(
            RigJoint(
                joint_id=jid,
                name=name,
                qpos_adr=int(model.jnt_qposadr[jid]),
                lower=lower,
                upper=upper,
                step=step,
                kind=kind,
            )
        )
    return joints


def apply_rig_joint_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    values: dict[str, float],
    joints: list[RigJoint] | None = None,
) -> None:
    """Write slider values into ``data.qpos`` and run ``mj_forward``."""
    joint_list = joints if joints is not None else list_rig_joints(model)
    by_name = {j.name: j for j in joint_list}
    for name, value in values.items():
        joint = by_name.get(name)
        if joint is None:
            raise KeyError(f"unknown rig joint {name!r}")
        data.qpos[joint.qpos_adr] = float(np.clip(value, joint.lower, joint.upper))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _geom_wxyz(data: mujoco.MjData, geom_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[geom_id])
    return quat


def _body_frame_pose(
    data: mujoco.MjData, body_id: int, kind: Literal["link", "com"]
) -> tuple[np.ndarray, np.ndarray]:
    """World position and wxyz of a body's link frame or COM (principal inertia) frame."""
    if kind == "link":
        return np.array(data.xpos[body_id], dtype=float), np.array(
            data.xquat[body_id], dtype=float
        )
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.ximat[body_id])
    return np.array(data.xipos[body_id], dtype=float), quat


def _geom_rgba_u8(model: mujoco.MjModel, geom_id: int) -> tuple[int, int, int, float]:
    # Like MuJoCo's renderer, a geom's material color overrides geom_rgba.
    matid = int(model.geom_matid[geom_id])
    src = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[geom_id]
    rgba = np.asarray(src, dtype=float)
    rgb = tuple(int(np.clip(c * 255.0, 0, 255)) for c in rgba[:3])
    alpha = float(np.clip(rgba[3], 0.0, 1.0))
    return rgb[0], rgb[1], rgb[2], alpha


def _mesh_trimesh(model: mujoco.MjModel, mesh_id: int) -> trimesh.Trimesh:
    vadr = int(model.mesh_vertadr[mesh_id])
    vnum = int(model.mesh_vertnum[mesh_id])
    fadr = int(model.mesh_faceadr[mesh_id])
    fnum = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=float)
    vertices = vertices * np.asarray(model.mesh_scale[mesh_id], dtype=float)
    faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=int)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _geom_local_trimesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    """Build a local-frame trimesh for a geom, or ``None`` to skip."""
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        return _mesh_trimesh(model, mesh_id)
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return trimesh.creation.box(extents=2.0 * size)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return trimesh.creation.icosphere(radius=float(size[0]), subdivisions=3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return trimesh.creation.capsule(
            radius=float(size[0]),
            height=max(2.0 * float(size[1]), 1e-6),
        )
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return trimesh.creation.cylinder(
            radius=float(size[0]),
            height=max(2.0 * float(size[1]), 1e-6),
        )
    if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=3)
        sphere.apply_scale(size)
        return sphere
    return None


def _is_collision_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    return int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0


@dataclass
class _ViserGeom:
    geom_id: int
    handle: Any
    is_collision: bool


def _resolve_source_spec(source: "MujocoAsset | mujoco.MjSpec") -> mujoco.MjSpec:
    from assetx.core.asset import MujocoAsset

    if isinstance(source, MujocoAsset):
        return source.spec
    if isinstance(source, mujoco.MjSpec):
        return source
    raise TypeError(
        f"launch_preview expected MujocoAsset or MjSpec, got {type(source)!r}"
    )


def _launch_mujoco_preview(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


def _launch_viser_preview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    port: int,
) -> None:
    import viser

    mujoco.mj_forward(model, data)
    joints = list_rig_joints(model)
    qpos0 = np.array(data.qpos, dtype=float, copy=True)

    server = viser.ViserServer(host="0.0.0.0", port=port, label="assetx rig")
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.25)

    geom_nodes: list[_ViserGeom] = []
    for gid in range(model.ngeom):
        local = _geom_local_trimesh(model, gid)
        if local is None or local.vertices.size == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        body_id = int(model.geom_bodyid[gid])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        )
        r, g, b, alpha = _geom_rgba_u8(model, gid)
        is_collision = _is_collision_geom(model, gid)
        # Collision-only geoms: slightly transparent so visuals stay readable.
        opacity = min(alpha, 0.35) if is_collision and alpha >= 0.99 else alpha
        opacity_arg = None if opacity >= 0.999 else max(opacity, 0.05)

        handle = server.scene.add_mesh_simple(
            f"/robot/{body_name}/{name}",
            vertices=np.asarray(local.vertices, dtype=np.float32),
            faces=np.asarray(local.faces, dtype=np.int32),
            color=(r, g, b),
            opacity=opacity_arg,
            position=np.asarray(data.geom_xpos[gid], dtype=float),
            wxyz=_geom_wxyz(data, gid),
        )
        geom_nodes.append(_ViserGeom(geom_id=gid, handle=handle, is_collision=is_collision))

    axes_length = 0.08 * float(model.stat.extent)
    frame_style = {
        "link": {"axes_length": axes_length, "origin_color": (236, 236, 0)},
        "com": {"axes_length": 0.6 * axes_length, "origin_color": (230, 0, 230)},
    }
    frame_nodes: dict[str, list[tuple[int, Any]]] = {"link": [], "com": []}
    for body_id in range(1, model.nbody):
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        )
        for kind, style in frame_style.items():
            pos, wxyz = _body_frame_pose(data, body_id, kind)  # type: ignore[arg-type]
            handle = server.scene.add_frame(
                f"/frames/{kind}/{body_name}",
                axes_length=style["axes_length"],
                axes_radius=0.05 * style["axes_length"],
                origin_radius=0.12 * style["axes_length"],
                origin_color=style["origin_color"],
                position=pos,
                wxyz=wxyz,
                visible=False,
            )
            frame_nodes[kind].append((body_id, handle))

    dirty = {"flag": True}
    sliders: dict[str, Any] = {}

    with server.gui.add_folder("Joints", expand_by_default=True):
        if not joints:
            server.gui.add_markdown("_No hinge/slide joints to rig._")
        for joint in joints:
            slider = server.gui.add_slider(
                joint.name,
                min=joint.lower,
                max=joint.upper,
                step=joint.step,
                initial_value=float(
                    np.clip(data.qpos[joint.qpos_adr], joint.lower, joint.upper)
                ),
                hint=joint.kind,
            )

            def _on_update(_event: Any, *, _name: str = joint.name) -> None:
                dirty["flag"] = True

            slider.on_update(_on_update)
            sliders[joint.name] = slider

    show_collision = server.gui.add_checkbox("Show collision geoms", initial_value=True)
    show_link = server.gui.add_checkbox(
        "Show link frames", initial_value=False, hint="Body frames (yellow origin)"
    )
    show_com = server.gui.add_checkbox(
        "Show COM frames",
        initial_value=False,
        hint="Center of mass, principal inertia axes (magenta origin)",
    )
    reset_btn = server.gui.add_button("Reset qpos0")

    @show_collision.on_update
    def _toggle_collision(_event: Any) -> None:
        visible = bool(show_collision.value)
        for node in geom_nodes:
            if node.is_collision:
                node.handle.visible = visible

    def _bind_frame_toggle(checkbox: Any, kind: str) -> None:
        @checkbox.on_update
        def _toggle(_event: Any) -> None:
            for _, handle in frame_nodes[kind]:
                handle.visible = bool(checkbox.value)

    _bind_frame_toggle(show_link, "link")
    _bind_frame_toggle(show_com, "com")

    @reset_btn.on_click
    def _reset(_event: Any) -> None:
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for joint in joints:
            sliders[joint.name].value = float(
                np.clip(data.qpos[joint.qpos_adr], joint.lower, joint.upper)
            )
        dirty["flag"] = True

    def _sync_scene() -> None:
        values = {name: float(slider.value) for name, slider in sliders.items()}
        if values:
            apply_rig_joint_values(model, data, values, joints)
        else:
            mujoco.mj_forward(model, data)
        for node in geom_nodes:
            node.handle.position = np.asarray(data.geom_xpos[node.geom_id], dtype=float)
            node.handle.wxyz = _geom_wxyz(data, node.geom_id)
        for kind, nodes in frame_nodes.items():
            for body_id, handle in nodes:
                pos, wxyz = _body_frame_pose(data, body_id, kind)  # type: ignore[arg-type]
                handle.position = pos
                handle.wxyz = wxyz

    print(f"Viser rig preview: http://localhost:{port}  (Ctrl+C to exit)")
    try:
        while True:
            if dirty["flag"]:
                _sync_scene()
                dirty["flag"] = False
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nStopping Viser preview.")
    finally:
        server.stop()


def launch_preview(
    source: "MujocoAsset | mujoco.MjSpec",
    *,
    ground: bool = True,
    lighting: bool = True,
    float_clearance: float = _DEFAULT_FLOAT_CLEARANCE,
    viewer: Literal["mujoco", "viser"] = "mujoco",
    port: int = _DEFAULT_VISER_PORT,
) -> None:
    """Open an interactive preview with optional ground and lighting.

    Ground / lighting are applied only to a temporary ``MjSpec`` copy so saved
    assets and conversion outputs are unaffected. When ``ground`` is enabled,
    the robot is automatically floated slightly above the plane (via freejoint
    lift, or by dropping the floor for fixed-base models).

    Parameters
    ----------
    viewer:
        ``"mujoco"`` — native passive MuJoCo viewer.
        ``"viser"`` — browser UI with per-joint kinematic sliders (best for
        posing / inspecting articulation).
    port:
        Viser HTTP port when ``viewer="viser"``.
    """
    if float_clearance < 0.0:
        raise ValueError(f"float_clearance must be >= 0, got {float_clearance}")
    if viewer not in ("mujoco", "viser"):
        raise ValueError(f"viewer must be 'mujoco' or 'viser', got {viewer!r}")

    spec = _resolve_source_spec(source)
    model = compile_for_preview(spec, lighting=lighting, ground=ground)
    data = mujoco.MjData(model)
    if ground:
        _float_above_ground(model, data, clearance=float_clearance)

    if viewer == "viser":
        _launch_viser_preview(model, data, port=port)
    else:
        _launch_mujoco_preview(model, data)
