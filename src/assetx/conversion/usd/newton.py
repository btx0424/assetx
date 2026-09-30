"""Isaac-free MJCF / URDF -> USD via the newton-physics converters.

- https://github.com/newton-physics/mujoco-usd-converter
- https://github.com/newton-physics/urdf-usd-converter

The converters depend on ``usd-exchange``, which vendors its own ``pxr`` and
would clobber assetx's ``usd-core`` if installed into the same venv, so they
run in isolated ``uv run --no-project`` environments. Post-processing
(:mod:`assetx.conversion.usd.postprocess`) runs in that same environment, so
the host process never needs ``pxr`` (Isaac Sim only provides it once Kit is
running).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from assetx.conversion.usd import postprocess

# The converters' usd-exchange wheels support Python < 3.13.
CONVERTER_PYTHON = "3.12"
MJCF_CONVERTER_PACKAGE = "mujoco-usd-converter"
MJCF_CONVERTER_VERSION = "0.6.0"
URDF_CONVERTER_PACKAGE = "urdf-usd-converter"
URDF_CONVERTER_VERSION = "0.3.3"


def run_in_converter_env(package: str, version: str, cmd: list[str]) -> None:
    """Run ``cmd`` in an ephemeral environment with ``package==version``."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv not found on PATH; install uv (https://docs.astral.sh/uv/)")
    # Without --isolated and a clean env, `uv run --with` layers the packages
    # over the caller's active venv, whose usd-core pxr segfaults usd-exchange.
    env = {k: v for k, v in os.environ.items() if k not in _INHERITED_ENV_VARS}
    subprocess.run(
        [
            uv,
            "run",
            "--isolated",
            "--no-project",
            "--managed-python",
            "--python",
            CONVERTER_PYTHON,
            # tinyobjloader>=2.0.0rc13 is a pre-release requirement of the converters.
            "--prerelease=allow",
            "--with",
            f"{package}=={version}",
            "--",
            *cmd,
        ],
        check=True,
        env=env,
    )


_INHERITED_ENV_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT")


def run_converter(package: str, version: str, executable: str, args: list[str]) -> None:
    run_in_converter_env(package, version, [executable, *args])


def _postprocess(package: str, version: str, usd_path: Path, *flags: str) -> None:
    if flags:
        run_in_converter_env(
            package, version, ["python", str(Path(postprocess.__file__)), str(usd_path), *flags]
        )


def newest_usd_file(out_dir: Path) -> Path:
    candidates = sorted(
        (p for p in out_dir.iterdir() if p.suffix in {".usda", ".usdc", ".usd"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Converter produced no top-level USD file in {out_dir}")
    return candidates[0]


def _prepare(src: str | Path, output_dir: str | Path | None, kind: str) -> tuple[Path, Path]:
    src = Path(src).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"{kind} not found: {src}")
    out = Path(output_dir).resolve() if output_dir is not None else src.parent / "usd"
    out.mkdir(parents=True, exist_ok=True)
    return src, out


def _common_args(layer_structure: bool, physics_scene: bool, comment: str, verbose: bool) -> list[str]:
    args = []
    if not layer_structure:
        args.append("--no-layer-structure")
    if not physics_scene:
        args.append("--no-physics-scene")
    if comment:
        args += ["--comment", comment]
    if verbose:
        args.append("--verbose")
    return args


def mjcf_to_usd(
    mjcf: str | Path,
    output_dir: str | Path | None = None,
    *,
    flatten: bool = True,
    layer_structure: bool = False,
    physics_scene: bool = False,
    comment: str = "",
    verbose: bool = False,
    converter_version: str = MJCF_CONVERTER_VERSION,
) -> Path:
    """Convert ``mjcf`` to USD and return the path of the primary USD file.

    Output goes to ``output_dir`` (default: ``<mjcf dir>/usd``). Converting
    from MJCF keeps the floating base, contact excludes (as
    ``UsdPhysics.FilteredPairsAPI``), sites and ``mjc:*`` contact parameters.
    ``flatten`` (default) rewrites the converter's nested tree into the Isaac
    Lab layout (see :func:`flatten_articulation`). ``layer_structure=True``
    writes an Atomic Component (``<model>.usda`` + payloaded layers) and is
    incompatible with ``flatten``. ``physics_scene`` is off by default since
    the simulator authors its own.
    """
    if flatten and layer_structure:
        raise ValueError("flatten requires a single-layer output (layer_structure=False)")
    mjcf, out = _prepare(mjcf, output_dir, "MJCF")
    args = [str(mjcf), str(out), *_common_args(layer_structure, physics_scene, comment, verbose)]
    run_converter(MJCF_CONVERTER_PACKAGE, converter_version, "mujoco_usd_converter", args)

    usd_path = newest_usd_file(out)
    _postprocess(
        MJCF_CONVERTER_PACKAGE, converter_version, usd_path, *(["--flatten"] if flatten else [])
    )
    return usd_path


def urdf_to_usd(
    urdf: str | Path,
    output_dir: str | Path | None = None,
    *,
    fix_base: bool = False,
    flatten: bool = True,
    layer_structure: bool = False,
    physics_scene: bool = False,
    packages: dict[str, str | Path] | None = None,
    comment: str = "",
    verbose: bool = False,
    converter_version: str = URDF_CONVERTER_VERSION,
) -> Path:
    """Convert ``urdf`` to USD and return the path of the primary USD file.

    Same options as :func:`mjcf_to_usd`. URDF cannot express a floating base,
    so the converter always welds the root link to the world; unless
    ``fix_base``, that joint is removed. ``packages`` maps ROS package names
    to directories for ``package://`` URIs.
    """
    if flatten and layer_structure:
        raise ValueError("flatten requires a single-layer output (layer_structure=False)")
    urdf, out = _prepare(urdf, output_dir, "URDF")
    args = [str(urdf), str(out)]
    for name, path in (packages or {}).items():
        args += ["--package", f"{name}={Path(path).resolve()}"]
    args += _common_args(layer_structure, physics_scene, comment, verbose)
    run_converter(URDF_CONVERTER_PACKAGE, converter_version, "urdf_usd_converter", args)

    usd_path = newest_usd_file(out)
    flags = [] if fix_base else ["--remove-world-fixed-joints"]
    if flatten:
        flags.append("--flatten")
    _postprocess(URDF_CONVERTER_PACKAGE, converter_version, usd_path, *flags)
    return usd_path
