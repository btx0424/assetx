"""Cook recipes into self-describing model bundles.

A bundle is ``save()``'s output (``model.xml``, ``model.urdf``, ``meshes/``,
optionally ``usd/``) plus ``assetx.json``, whose ``hash`` covers every input
that determines the output: the recipe module, the assetx library code
(transforms, conversion, pinned converter versions, ...) and the pinned vendor
URL. :func:`check` compares that hash to decide whether a bundle is stale.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from assetx.recipes import Recipe, get_recipe

MANIFEST = "assetx.json"
MANIFEST_VERSION = 1

_PACKAGE_ROOT = Path(__file__).resolve().parent
# Package code that cannot change a cooked bundle. Recipe modules are hashed
# individually (only the one defining the recipe).
_HASH_EXCLUDE = ("recipes/", "core/preview.py", "cli.py", "fetch.py")
# Other recipe modules are excluded so editing one recipe doesn't stale every bundle,
# but the registry decides the vendor layout, which shows up in mesh paths.
# fetch.py only changes how pinned vendor files are transported, not their content.
_HASH_INCLUDE = ("recipes/registry.py",)


def _library_files() -> list[Path]:
    files = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel in _HASH_INCLUDE or not any(rel == ex or rel.startswith(ex) for ex in _HASH_EXCLUDE):
            files.append(path)
    return files


def recipe_hash(recipe: Recipe) -> str:
    """Hash of everything that determines ``recipe``'s cooked output."""
    h = hashlib.sha256()
    h.update(f"manifest={MANIFEST_VERSION}\n".encode())
    h.update(f"recipe={recipe.name}\nvendor={recipe.vendor}\nmesh_format={recipe.mesh_format}\n".encode())
    # The recipe module may live outside assetx; label it by role, not path.
    h.update(b"file=<recipe>\n" + recipe.source_file.read_bytes())
    for path in _library_files():
        h.update(f"file={path.relative_to(_PACKAGE_ROOT).as_posix()}\n".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def read_manifest(out_dir: str | Path) -> dict | None:
    path = Path(out_dir) / MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, out_dir: str | Path, *, usd: bool = True) -> str | None:
    """Return ``None`` if ``out_dir`` holds an up-to-date bundle, else why not."""
    out_dir = Path(out_dir)
    manifest = read_manifest(out_dir)
    if manifest is None:
        return "missing" if not out_dir.exists() else f"not cooked by assetx (no {MANIFEST})"
    if manifest.get("recipe") != name:
        return f"cooked from recipe {manifest.get('recipe')!r}, not {name!r}"
    if manifest.get("hash") != recipe_hash(get_recipe(name)):
        return "stale (recipe or assetx changed since it was cooked)"
    if usd and "usd" not in manifest.get("formats", []):
        return "missing USD"
    return None


def _assetx_version() -> str:
    try:
        return version("assetx")
    except PackageNotFoundError:
        return "unknown"


def cook(name: str, out_dir: str | Path, *, usd: bool = True, force: bool = False) -> Path:
    """Build recipe ``name`` into ``out_dir`` unless it is already up to date.

    The bundle is written to a temporary sibling directory and swapped in only
    once complete, under a lock, so concurrent cooks and interrupted runs never
    leave a half-written ``out_dir``. An existing ``out_dir`` is replaced.
    """
    from assetx.conversion.usd import newton

    recipe = get_recipe(name)
    out_dir = Path(out_dir).resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / f".{out_dir.name}.cook.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not force and check(name, out_dir, usd=usd) is None:
            return out_dir

        tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
        tmp.chmod(0o755)
        try:
            recipe.build().save(tmp, save_urdf=True, save_usd=usd, mesh_format=recipe.mesh_format)
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "recipe": name,
                "hash": recipe_hash(recipe),
                "formats": ["mjcf", "urdf", *(["usd"] if usd else [])],
                "vendor": recipe.vendor,
                "assetx_version": _assetx_version(),
                "usd_converter": (
                    f"{newton.MJCF_CONVERTER_PACKAGE}=={newton.MJCF_CONVERTER_VERSION}"
                    if usd
                    else None
                ),
                "cooked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
            (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            old = out_dir.with_name(f".{out_dir.name}.old")
            if old.exists():
                shutil.rmtree(old)
            if out_dir.exists():
                out_dir.rename(old)
            tmp.rename(out_dir)
            shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return out_dir
