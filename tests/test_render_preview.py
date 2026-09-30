"""Offscreen visual / collision preview images."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from assetx.core.preview import _write_png, render_preview


def test_write_png_roundtrip(tmp_path) -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[1, 2] = (0, 0, 255)
    path = tmp_path / "nested" / "img.png"
    _write_png(path, rgb)

    raw = path.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert raw.endswith(b"IEND\xaeB`\x82")


def test_render_preview_distinguishes_collision(tmp_path) -> None:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="link")
    body.add_geom(
        name="visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.05),
        contype=0,
        conaffinity=0,
        rgba=(0.9, 0.2, 0.1, 1.0),
    )
    body.add_geom(
        name="collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.04, 0.12),
        pos=(0.0, 0.0, 0.1),
        contype=1,
        conaffinity=1,
        rgba=(0.2, 0.2, 0.2, 1.0),
    )

    try:
        written = render_preview(spec, tmp_path, ground=False, width=80, height=60)
    except Exception as exc:
        pytest.skip(f"offscreen rendering unavailable: {exc}")

    visual = written["visual"].read_bytes()
    collision = written["collision"].read_bytes()
    assert visual.startswith(b"\x89PNG")
    assert collision.startswith(b"\x89PNG")
    assert visual != collision
