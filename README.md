# assetx

**Composable, reproducible robot descriptions from base assets and recipes.**

Assemble MJCF building blocks (base, arm, gripper), apply transforms, and export a
deterministic robot model — instead of hand-editing monolithic XML copies.

## Motivation

Robot descriptions are often single, hand-edited files. Reusing a sub-assembly
means copying XML, fixing mesh paths, and hoping nothing breaks. Sharing a
variant usually means sharing another full copy, so provenance is unclear.

**assetx** makes composition explicit:

1. **Base assets** — Canonical MJCF blocks (quadruped, arm, gripper). Versioned
   sources of truth; not edited per-robot.
2. **Recipes** — Python functions that **assemble** assets (mount child on
   parent link + pose) and **transform** the result (rename bodies, strip
   actuators/sensors, fit collision shapes, add grasp frames, …).
3. **Final robot** — Fully determined by `base assets + recipe`. Same inputs
   always yield the same output.

That gives reproducibility, reuse across bases (e.g. one “mount arm” recipe on
different quadrupeds), and clear provenance.

## Installation

Requires Python ≥ 3.10 and a working MuJoCo install.

```bash
cd aa-projects/assetx
pip install -e .
```

Core dependencies (from `pyproject.toml`): `mujoco`, `scipy`, `trimesh`, `viser`,
`usd-core`.

Optional research extras:

```bash
pip install -e ".[research]"   # adds mujoco-warp
```

## Quick start

Recipes are plain Python functions returning `MujocoAsset`. Use `@asset_builder`
only if you want optional registration.

```python
from assetx import (
    Compose,
    MujocoAsset,
    NormalizeGeomNames,
    RenameBodies,
    ReplaceCylinderWithCapsule,
    assemble,
    asset_builder,
)

@asset_builder
def load_base(path) -> MujocoAsset:
    return MujocoAsset.from_file(path)

@asset_builder
def load_arm(path) -> MujocoAsset:
    return MujocoAsset.from_file(path)

@asset_builder
def build_robot(base: MujocoAsset, arm: MujocoAsset) -> MujocoAsset:
    robot = assemble(
        parent=base,
        child=arm,
        parent_link="base_link",
        child_prefix="arm_",
        translation=(0.05, 0.0, 0.10),
    )
    return Compose([
        NormalizeGeomNames(),
        ReplaceCylinderWithCapsule(),
        RenameBodies({"arm_link6": "gripper_base"}),
    ]).transform(robot)
```

- **Transforms** are unary (`asset → asset`).
- **Assembly** is multi-input (`parent + child → asset`).
- Recipes are normal call graphs that IDEs can navigate.

## Examples

Run from the `assetx` package root (or any cwd; examples write under
`artifacts/`). Vendor MJCF is fetched automatically into `artifacts/vendor/`
when local paths are omitted.

| Example | What it builds | Command |
|--------|----------------|---------|
| [`examples/a2_piper.py`](examples/a2_piper.py) | Unitree A2 + AgileX Piper | `python examples/a2_piper.py` |
| [`examples/b2_kinova.py`](examples/b2_kinova.py) | Unitree B2 + Kinova Gen3 | `python examples/b2_kinova.py` |
| [`examples/b2z1.py`](examples/b2z1.py) | B2-Z1 cleanup / merge recipe | `python examples/b2z1.py --help` |
| [`examples/rov_arx.py`](examples/rov_arx.py) | BlueROV + ARX X5A (lab paths) | `python examples/rov_arx.py --help` |
| [`examples/g1_inspire_hand.py`](examples/g1_inspire_hand.py) | G1 Inspire finger capsule approx. | `python examples/g1_inspire_hand.py --help` |

Typical flags (A2 + Piper):

```bash
python examples/a2_piper.py                  # fetch vendor MJCF if needed, preview
python examples/a2_piper.py --no-viewer      # export only → artifacts/a2_piper/
python examples/a2_piper.py --force-download # refresh artifacts/vendor/
python examples/a2_piper.py --a2 /path/to/a2.xml --piper /path/to/piper.xml
```

Outputs land in `artifacts/<name>/` (e.g. `model.xml`, `model.urdf`, meshes).

A common recipe pattern: strip vendor sensors/actuators on load (downstream
apps add their own), then assemble and rename EE links / add a `grasp_point`:

```python
from assetx import Compose, RemoveActuators, RemoveSensors

Compose([
    RemoveSensors(names=[".*pos", ".*torque", "imu.*"]),
    RemoveActuators(names=[".*"]),
]).transform(MujocoAsset.from_file("a2.xml"))
```

## Package layout

| Path | Role |
|------|------|
| `assetx` / `assetx.core` | MJCF assemble, transforms, builders, preview |
| `assetx.conversion` | MJCF ↔ URDF, USD helpers |
| `assetx.fetch` | Sparse GitHub directory download for vendor MJCF |
| `examples/` | End-to-end recipes |
| `artifacts/` | Generated robots + cached vendor trees (local) |

## License

MIT — see `pyproject.toml`.
