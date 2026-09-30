# Repository Guidelines

## Purpose

**assetx** composes and transforms MuJoCo (MJCF) robot models from base assets + Python recipes (assemble → transform → save). Format bridges (USD / URDF) live under `assetx.conversion`; CLIs in `tools/` stay thin.

Outputs are consumed by active-adaptation. Naming and USD-layout changes can break its asset factories, camera and contact sensors; check the contract in `README.md` ("Using with active-adaptation") before changing them.

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
      newton.py         # MJCF/URDF→USD via isolated converter envs
      postprocess.py    # stdlib + pxr only; runs inside the converter env
  recipes/              # registered recipes (@recipe, pinned vendor SHA)
  cook.py               # cook()/check(): bundles + assetx.json staleness hash
  cli.py                # `assetx` list / cook / status / preview
tools/                  # argparse CLIs (+ optional viewer)
tools/research/         # optional experiments ([research] extra)
examples/               # one-off recipe scripts
artifacts/              # generated models (gitignored)
```

### `core/` — MJCF

| Module | Responsibility |
|--------|----------------|
| `asset.py` | `MujocoAsset`, `JointCfg` |
| `assemble.py` | `assemble(parent, child, …)` |
| `transforms/` | Unary `Transform` package (`base`, `topology`, `edit`, `simplification`) |
| `builders.py` | `@asset_builder` |
| `preview.py` | `launch_preview` (lit viewer copy; not saved) |
| `fetch.py` | Download a GitHub subdirectory into `artifacts/` (no full clone) |

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
| New MJCF transform | `core/transforms/` (`base` / `topology` / `edit` / `simplification`); re-export from `transforms/__init__.py`, `core/__init__.py`, and package `__init__.py` |
| Assembly / asset I/O | `core/assemble.py` / `core/asset.py` |
| Robot consumed by another project | `recipes/<family>.py` with `@recipe`, imported in `recipes/__init__.py` |
| One-off recipe script | `examples/` |
| USD geom logic | `conversion/usd/geoms.py` |
| USD→MJCF logic | `conversion/usd/robot.py`; keep `tools/usd2mjcf.py` as CLI only |
| URDF↔MJCF | `conversion/urdf2mjcf.py` or `conversion/mjcf2urdf.py` |
| Research / viz | `tools/research/` |
| Tests | `tests/test_<feature>.py` |
| Collision mesh → capsule | `.agent/skills/collision-simplification/SKILL.md` |

Do **not** put generated outputs under `src/`.

## Build & Run

```bash
uv sync                          # .venv on uv-managed Python 3.12
uv sync --extra research

uv run assetx list
uv run assetx cook spot --out-root /tmp/cook
uv run assetx status --out-root /tmp/cook
uv run examples/a2_piper.py --help
uv run tools/usd2mjcf.py --help
uv run tools/mjcf2usd.py --help
```

Never import `pxr` in code that `cook`/`save(save_usd=True)` runs in the calling process: AA calls it from Isaac venvs where Kit owns `pxr`. USD work goes through `conversion/usd/newton.run_in_converter_env`.

Changing any hashed file (everything under `src/assetx/` except recipe modules other than `registry.py`, `core/preview.py` and `cli.py`) marks every cooked bundle stale, which is intended: downstream users must re-run `aa-cook-assets`.

## Coding Style

- 4-space indent; type hints on public functions.
- Builders: named `MujocoAsset` params (`base`, `arm`), not `*args`.
- Naming: `build_a2_piper`, `RenameBodies`, modules `lowercase_with_underscores`.
- Pure imports: no viewer launch or file deletion at import time.
- CLI viewers are opt-out via `--no-viewer` where interactive; use `assetx.launch_preview` (adds a temporary key light, never written to disk).
- `MujocoAsset.from_file` requires exactly one body under `worldbody`.
- `assemble()` leaves mesh paths absolute into the parent/child source trees (no temp dir). Call `save()` to write a durable artifact with meshes copied under the output directory.

## Testing

`uv run pytest -q`. Add `tests/test_<feature>.py` for deterministic ops (load/save, assemble, transforms, conversion, cook). Avoid interactive viewers and network access in tests (monkeypatch `registry.fetch_vendor`).

## Commits & PRs

Short imperative subjects. Note API impact, affected paths, verification commands, and sample artifact paths when outputs change.
