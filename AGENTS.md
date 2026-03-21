# Repository Guidelines

## Project Structure & Module Organization

Core library code lives in `src/assetx/core/`. Keep stable MJCF APIs there:

- `asset.py`: `MujocoAsset`, `JointCfg`
- `assemble.py`: multi-input MJCF composition
- `transforms.py`: unary MJCF transforms
- `builders.py`: function-based asset builder registration

Top-level `src/assetx/` files are compatibility shims. Prefer importing from `assetx` or `assetx.core`, not from old internal paths.

Use `examples/` for canonical library usage, such as `examples/a2_piper.py`. Use `tools/` for one-off utilities and research workflows (`tools/research/`). Do not put generated outputs in source directories; use `artifacts/` or OS temp directories instead.

## Build, Test, and Development Commands

- `python -m pip install -e .` installs the package in editable mode.
- `python examples/a2_piper.py --help` checks the builder example entrypoint.
- `python tools/urdf2mjcf.py --help` shows the URDF conversion utility.
- `python tools/extract_meshes.py --help` shows the USD mesh extraction utility.
- `pytest` runs tests when a `tests/` tree exists.

## Coding Style & Naming Conventions

Use 4-space indentation and type hints for public functions. Keep APIs explicit: builder functions should use named `MujocoAsset` parameters such as `base` and `arm`, not `*args`.

Naming:

- builder functions: `build_a2_piper`, `load_piper`
- transform classes: `RenameBodies`, `RemoveSubtrees`
- modules: lowercase with underscores only

Prefer pure imports: no file deletion, temp directory cleanup, or viewer launch at import time.

## Testing Guidelines

There is no large test suite yet. When adding stable behavior, add `pytest` tests under `tests/` named `test_<feature>.py`. Focus on deterministic MJCF operations first: load/save, assembly, and transforms. Avoid depending on interactive viewers in tests.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `refactor`, `refine example`, and `readme`. Keep commit titles concise and lowercase unless a proper noun requires otherwise.

PRs should include:

- a short summary of the structural or API change
- affected paths, for example `src/assetx/core/assemble.py`
- verification steps you ran
- sample output or artifact path if behavior changes materially
