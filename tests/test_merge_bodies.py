"""Merged inertia stays finite when a body has no inertial element."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assetx import MergeBodies, MujocoAsset


def _asset_with_bodies(masses: tuple[float, float]) -> MujocoAsset:
    import mujoco

    spec = mujoco.MjSpec()
    parent = spec.worldbody.add_body(name="base")
    for name, mass, pos in (("a", masses[0], (0.0, 0.0, 0.0)), ("b", masses[1], (0.0, 0.0, 1.0))):
        body = parent.add_body(name=name, pos=pos)
        if mass > 0.0:
            body.mass = mass
            body.ipos = (0.0, 0.0, 0.1)
            body.inertia = (0.01, 0.02, 0.03)
            body.iquat = (1.0, 0.0, 0.0, 0.0)
            body.explicitinertial = 1
        else:
            # MuJoCo's sentinel for an omitted <inertial> center of mass.
            body.ipos = (float("nan"), 0.0, 0.0)
    return MujocoAsset(Path("/tmp/model.xml"), spec, Path("."))


def test_merge_skips_unset_inertial() -> None:
    out = MergeBodies(["a", "b"]).transform(_asset_with_bodies((2.0, 0.0)))
    body = out.spec.body("a")
    assert abs(float(body.mass) - 2.0) < 1e-9
    assert np.all(np.isfinite(body.ipos))
    assert np.all(np.isfinite(body.inertia))
    assert abs(float(body.ipos[2]) - 0.1) < 1e-6


def test_merge_massless_bodies_stays_finite() -> None:
    out = MergeBodies(["a", "b"]).transform(_asset_with_bodies((0.0, 0.0)))
    body = out.spec.body("a")
    assert float(body.mass) == 0.0
    assert np.all(np.isfinite(body.ipos))
    assert np.all(np.isfinite(body.inertia))
