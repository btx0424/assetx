# Repository Guidelines

## Purpose

**assetx** composes and transforms MuJoCo (MJCF) robot models from base assets + Python recipes (assemble → transform → save). Format bridges (USD / URDF) live under `assetx.conversion`; CLIs in `tools/` stay thin.

## Layout

```
src/assetx/
  core/                 # MJCF-only API
  conversion/           # format bridges
    mjcf2urdf.py
    urdf2mjcf.py
    usd/
      geoms.py          # mesh / collision extraction
      robot.py          # kinematic tree + USD→MJCF
tools/                  # argparse CLIs (+ optional viewer)
tools/research/         # optional experiments ([research] extra)
examples/               # canonical recipes
artifacts/              # generated models (gitignored)
```

### `core/` — MJCF

| Module | Responsibility |
|--------|----------------|
| `asset.py` | `MujocoAsset`, `JointCfg` |
| `assemble.py` | `assemble(parent, child, …)` |
| `transforms.py` | Unary `Transform` subclasses + `Compose` |
| `builders.py` | `@asset_builder` |
| `preview.py` | `launch_preview` (lit viewer copy; not saved) |

Import from `assetx` or `assetx.core`.

### `conversion/` — format bridges

| Module | Responsibility |
|--------|----------------|
| `mjcf2urdf` | `write_urdf`, `mjcf_to_urdf` |
| `urdf2mjcf` | `prepare_urdf_for_mujoco`, `urdf_to_mjcf` |
| `usd.geoms` | `extract_body_geoms`, `extract_meshes`, `export_meshes` |
| `usd.robot` | `convert_usd_to_mjcf`, `build_kinematic_tree`, `build_mjcf` |

Conversion writes beside the input (same directory / stem): `foo.usd` → `foo.xml` + `meshes/`.

Collision geom names from USD→MJCF: `{body}_collision` / `{body}_collision{N}`, feet `{leg}_foot_collision` (matches mjlab `.*_collision.*` / `.*_foot_collision$`).

## Where to Put New Code

| Task | Location |
|------|----------|
| New MJCF transform | `core/transforms.py`; export from `core/__init__.py` and package `__init__.py` |
| Assembly / asset I/O | `core/assemble.py` / `core/asset.py` |
| Recipe example | `examples/` |
| USD geom logic | `conversion/usd/geoms.py` |
| USD→MJCF logic | `conversion/usd/robot.py`; keep `tools/usd2mjcf.py` as CLI only |
| URDF↔MJCF | `conversion/urdf2mjcf.py` or `conversion/mjcf2urdf.py` |
| Research / viz | `tools/research/` |
| Tests | `tests/test_<feature>.py` |

Do **not** put generated outputs under `src/`.

## Build & Run

```bash
# Prefer lab51 conda env when available (pxr + mujoco).
pip install -e .
pip install -e ".[usd]"        # USD tools (+ mujoco-usd-converter)
pip install -e ".[research]"

python examples/a2_piper.py --help
python tools/usd2mjcf.py --help
python tools/urdf2mjcf.py --help
python tools/mjcf2urdf.py --help
python tools/mjcf2usd.py --help
python tools/extract_meshes.py --help
```

```bash
PYTHONPATH=src python tools/usd2mjcf.py -p /path/to/robot.usd --no-viewer
```

## Coding Style

- 4-space indent; type hints on public functions.
- Builders: named `MujocoAsset` params (`base`, `arm`), not `*args`.
- Naming: `build_a2_piper`, `RenameBodies`, modules `lowercase_with_underscores`.
- Pure imports: no viewer launch or file deletion at import time.
- CLI viewers are opt-out via `--no-viewer` where interactive; use `assetx.launch_preview` (adds a temporary key light, never written to disk).
- `MujocoAsset.from_file` requires exactly one body under `worldbody`.
- `assemble()` keeps a `TemporaryDirectory` alive on the returned asset (`_tmpdir`) until the asset is GC'd; call `save()` to persist.

## Testing

No suite yet. Add `tests/test_<feature>.py` for deterministic MJCF ops first (load/save, assemble, transforms, mjcf2urdf). Avoid interactive viewers in tests.

## Commits & PRs

Short imperative subjects. Note API impact, affected paths, verification commands, and sample artifact paths when outputs change.
