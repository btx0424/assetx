"""MuJoCo viewer helpers for interactive previews (never written to disk)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer

if TYPE_CHECKING:
    from assetx.core.asset import MujocoAsset


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


def compile_for_preview(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile a copy of ``spec`` with preview lighting (original unchanged)."""
    preview = add_preview_light(spec.copy())
    return preview.compile()


def launch_preview(source: "MujocoAsset | mujoco.MjSpec") -> None:
    """Open a passive MuJoCo viewer with preview lighting.

    Lighting is applied only to a temporary ``MjSpec`` copy so saved assets /
    conversion outputs are unaffected.
    """
    from assetx.core.asset import MujocoAsset

    if isinstance(source, MujocoAsset):
        spec = source.spec
    elif isinstance(source, mujoco.MjSpec):
        spec = source
    else:
        raise TypeError(
            f"launch_preview expected MujocoAsset or MjSpec, got {type(source)!r}"
        )

    model = compile_for_preview(spec)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
