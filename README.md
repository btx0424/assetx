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

assetx is managed with [uv](https://docs.astral.sh/uv/). From the package root:

```bash
cd aa-projects/assetx
uv sync                    # creates .venv (Python 3.12, see .python-version) + dev deps
uv sync --extra research   # optionally add mujoco-warp
```

Then run anything through the project environment with `uv run` (no activation
needed):

```bash
uv run assetx preview spot
uv run pytest -q
```

Core dependencies (from `pyproject.toml`): `mujoco`, `scipy`, `trimesh`, `viser`,
`usd-core`. `pytest` is in the `dev` dependency group, installed by default.

### Viewer / OpenGL troubleshooting

`pyproject.toml` sets `python-preference = "only-managed"`, so uv always builds
`.venv` on its own standalone CPython, even inside an active conda env. Conda
interpreters bundle an old `libstdc++` that the system Mesa driver cannot load,
which breaks the MuJoCo viewer with:

```text
libGL error: MESA-LOADER: failed to open iris ...
GLFWError: (65543) GLX: Failed to create context
ERROR: could not create window
```

If you see this, the venv was created from a conda Python (check
`readlink -f .venv/bin/python`). Recreate it:

```bash
rm -rf .venv && uv sync
```

Run `LIBGL_DEBUG=verbose uv run assetx preview spot` to see which library fails to
load. On a headless machine, use the Viser preview (`--viser`) or `--no-viewer`.

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

## Recipes and cooking

Robots that downstream projects consume are registered recipes in
[`src/assetx/recipes/`](src/assetx/recipes/). Each recipe pins its vendor
source to a full commit SHA, and `assetx cook` turns it into a bundle:

```bash
uv run assetx list                                 # registered recipes
uv run assetx cook spot spot_arm --out-root DIR    # -> DIR/<name>/{model.xml,model.urdf,meshes/,usd/,assetx.json}
uv run assetx status --out-root DIR                # exit 1 if any bundle is missing or stale
uv run assetx preview spot [--viser]               # build in memory and open a viewer
```

| Recipe | Vendor |
|--------|--------|
| `spot` | Boston Dynamics Spot, feet split into `*_foot` bodies (MuJoCo Menagerie) |
| `spot_arm` | Spot + arm, plus a `grasp_point` body on the wrist (MuJoCo Menagerie) |

- **Staleness.** `assetx.json` records a sha256 over the recipe module, the
  assetx library code that shapes the output (transforms, exporters, USD
  conversion, the recipe registry), the vendor URL and the mesh format.
  `cook` skips bundles whose hash matches (use `--force` to rebuild).
  `status` and `assetx.cook.check()` report a bundle as stale when the hash
  differs, and as foreign when there is no manifest.
- **Vendor cache.** Pinned vendor trees are fetched once into
  `$ASSETX_CACHE_DIR/vendor/` (default `~/.cache/assetx/vendor/`).
- **Atomic.** A cook builds into a temporary sibling directory under a file lock
  and swaps it in, so readers never see a half-written bundle.
- **USD.** On by default (`--no-usd` skips it); see below.

To add one, write a function returning `MujocoAsset` in a module under
`recipes/`, decorate it with `@recipe("name", vendor=<pinned GitHub tree URL>)`,
and import the module in `recipes/__init__.py`.

## Examples

Run from the `assetx` package root (examples write under `artifacts/`, relative
to the cwd). Vendor MJCF is fetched automatically into `artifacts/vendor/` when
local paths are omitted. These are one-off scripts; move a robot into
`recipes/` once another project depends on it.

| Example | What it builds | Command |
|--------|----------------|---------|
| [`examples/a2_piper.py`](examples/a2_piper.py) | Unitree A2 + AgileX Piper | `uv run examples/a2_piper.py` |
| [`examples/as2.py`](examples/as2.py) | Unitree AS2 | `uv run examples/as2.py` |
| [`examples/b2_kinova.py`](examples/b2_kinova.py) | Unitree B2 + Kinova Gen3 | `uv run examples/b2_kinova.py` |
| [`examples/b2z1.py`](examples/b2z1.py) | B2-Z1 cleanup / merge recipe | `uv run examples/b2z1.py --help` |
| [`examples/rov_arx.py`](examples/rov_arx.py) | BlueROV + ARX X5A (lab paths) | `uv run examples/rov_arx.py --help` |
| [`examples/g1_inspire_hand.py`](examples/g1_inspire_hand.py) | G1 Inspire finger capsule approx. | `uv run examples/g1_inspire_hand.py --help` |

Typical flags (A2 + Piper):

```bash
uv run examples/a2_piper.py                  # fetch vendor MJCF if needed, preview
uv run examples/a2_piper.py --no-viewer      # export only → artifacts/a2_piper/
uv run examples/a2_piper.py --force-download # refresh artifacts/vendor/
uv run examples/a2_piper.py --a2 /path/to/a2.xml --piper /path/to/piper.xml
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

## USD export without Isaac Sim

[`tools/mjcf2usd.py`](tools/mjcf2usd.py) converts an exported `model.xml`
with [newton-physics/mujoco-usd-converter](https://github.com/newton-physics/mujoco-usd-converter).
Prefer it over the URDF route: it keeps the floating base, contact excludes
(as `UsdPhysics.FilteredPairsAPI`), sites and MuJoCo contact parameters
(`mjc:*` attributes, ignored by PhysX).

```bash
uv run tools/mjcf2usd.py artifacts/spot/model.xml                   # -> artifacts/spot/usd/spot.usdc
uv run tools/mjcf2usd.py artifacts/spot/model.xml --keep-hierarchy  # nested bodies (Newton/MuJoCo)
uv run tools/mjcf2usd.py artifacts/spot/model.xml --keep-hierarchy --layered  # Atomic Component
```

From a recipe, `robot.save(out_dir, save_usd=True)` runs the same conversion
on the saved `model.xml` (-> `out_dir/usd/<model>.usdc`). The Python API is
`assetx.conversion.usd.mjcf_to_usd` / `urdf_to_usd`.

[`tools/urdf2usd.py`](tools/urdf2usd.py) does the same for URDF-only sources
with [newton-physics/urdf-usd-converter](https://github.com/newton-physics/urdf-usd-converter)
(same flags, plus `--fix-base` and `--package NAME=PATH`).

- **Isaac Lab layout by default.** The converters nest rigid bodies along the
  kinematic tree (`/spot/Geometry/base_link/fl_hip/fl_uleg/...`). PhysX does
  not support nested rigid bodies, and Isaac Lab prim-path regexes such as
  `/Robot/.*_foot` match one level only. The tools therefore write a single
  `.usdc` layer and flatten every link to a direct child of the robot prim
  (`/spot/base_link`, `/spot/fl_uleg`, ...), preserving world poses and
  rewriting joint bodies, filtered pairs and material bindings. Each link's
  geometry is then grouped like the Isaac importer does: shapes with
  `CollisionAPI` go under `{link}/collisions`, the rest under `{link}/visuals`
  (what active-adaptation's camera / mesh registry reads), and every joint
  moves to `/<robot>/joints/<joint>`. Duplicate joint names (the MuJoCo
  converter calls every weld `PhysicsFixedJoint`) become `<link>_<name>`.
  `--keep-hierarchy` skips this (the style Newton and MuJoCo's USD loader expect).
- **Floating base (URDF only).** URDF cannot express one, so the URDF converter
  always welds the root link to the world; `urdf2usd.py` removes that joint
  unless `--fix-base` is passed.
- **No physics scene.** No `UsdPhysics.Scene` is authored unless
  `--physics-scene` is passed; the simulator is expected to provide it.
- **Isolated converters.** Conversion and the flattening pass run in an
  ephemeral `uv run --isolated` environment (pinned converter, Python 3.12)
  with the caller's `VIRTUAL_ENV` stripped, because `usd-exchange` ships its own
  `pxr`, which crashes when mixed with `usd-core`. The calling process
  therefore never imports `pxr` (Isaac Sim's Kit can call it), but `uv` must be
  on `PATH`.
- MJCF keyframes are not converted.

## Using with active-adaptation

assetx produces robot models; [active-adaptation](../../active-adaptation)
(AA) simulates them. AA depends on assetx only to cook and check bundles
(`assetx.cook`); at simulation time the interface is the bundle on disk plus
the naming conventions below.

| Side | Owns |
|------|------|
| assetx | Composition, renames, geom naming, stripping vendor actuators / sensors / scene settings, MJCF / URDF / USD export |
| AA (`active_adaptation/assets/<family>/<robot>.py`) | `AssetSpec` factories: PD actuators, contact sensors, init state, symmetry, `joint_names_simulation` / `body_names_simulation` |

### Workflow

Robots with a registered recipe (currently `spot`, `spot_arm`) are cooked
from AA's environment straight into `ROBOT_MODEL_DIR`
(`active-adaptation/.cache/aa-robot-models/`):

```bash
aa-cook-assets              # every recipe AA uses; skips fresh bundles
aa-cook-assets spot --force
```

AA factories check the bundle with `assetx.cook.check()` when they build their
config and raise with that command if it is missing or stale. They never cook
implicitly. mjlab's `spec_fn` reads `model.xml` and Isaac's `UsdFileCfg` reads
`usd/<model>.usdc`.

Robots that still come from `examples/` are exported and copied by hand:

```bash
cd aa-projects/assetx
uv run examples/a2_piper.py --no-viewer                 # -> artifacts/a2_piper/model.xml + meshes/
uv run tools/mjcf2usd.py artifacts/a2_piper/model.xml   # -> artifacts/a2_piper/usd/<model>.usdc
mkdir -p ../../active-adaptation/.cache/aa-robot-models/a2_piper
cp -r artifacts/a2_piper/* ../../active-adaptation/.cache/aa-robot-models/a2_piper/
```

Never point factories at `artifacts/`; it is a mutable build dir.

### What AA relies on

- **No actuators, sensors, lights or `<option>`** in the saved MJCF
  (`RemoveActuators`, `RemoveSensors`, `RemoveSceneSettings`). AA adds
  actuators and contact sensors per backend, and the simulation settings.
- **Root free joint named `floating_base_joint`** (`NormalizeJointNames`), so
  every robot exposes the same root joint.
- **Collision geoms named `{body}_collision{i}`** (`NormalizeGeomNames`),
  matched by mjlab `CollisionCfg(geom_names_expr=(".*_collision.*",))`.
  `NormalizeGeomNames` only names unnamed geoms; vendor-named colliders (e.g.
  Spot's `FL` foot) keep their name and need their own pattern or a rename.
- **Stable body / joint names.** Take `*_names_simulation`, actuator
  `joint_names_expr` and symmetry mappings from the saved model, not the
  vendor XML. `RenameBodies` / `NormalizeJointNames` rewrite in-model
  references (excludes, equalities, sensors), not AA configs.
- **Flat USD articulation.** Every link is a direct child of the robot prim,
  so Isaac spawns it at `{ENV_REGEX_NS}/Robot/<link>` and
  `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")` matches all links.
  Joints live in `/Robot/joints/`, as with Isaac's importer.
- **`{link}/visuals` and `{link}/collisions` USD groups.** AA's camera sensor,
  mesh registry and Viser viewer read body-local meshes from `{link}/visuals`
  (`envs/backends/isaaclab/meshes.py`); `get_collision_meshes` reads
  `{link}/collisions`. When a link has no `visuals` group, AA falls back to the
  whole link and skips `purpose=guide` / invisible prims (the converters mark
  collision shapes as `guide`).
- **URDF visuals** contain only mesh geoms in MuJoCo's default-visible groups
  (0–2) or visual-only geoms, with material colors, so the Isaac URDF importer
  and the USD converters do not render colliders.

## Package layout

| Path | Role |
|------|------|
| `assetx` / `assetx.core` | MJCF assemble, transforms, builders, preview |
| `assetx.conversion` | MJCF ↔ URDF, USD helpers |
| `assetx.fetch` | Sparse GitHub directory download for vendor MJCF |
| `assetx.recipes` | Registered recipes with pinned vendors (`@recipe`) |
| `assetx.cook` / `assetx` CLI | Cook bundles with a staleness manifest; `list` / `cook` / `status` / `preview` |
| `examples/` | One-off recipe scripts |
| `artifacts/` | Generated robots + cached vendor trees (local) |

## License

MIT — see `pyproject.toml`.
