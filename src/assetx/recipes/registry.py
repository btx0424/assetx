"""Named, reproducible robot recipes (vendor source pinned to a commit)."""

from __future__ import annotations

import fcntl
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from assetx.core.asset import MujocoAsset
from assetx.fetch import download_github_dir, parse_github_dir_url

RecipeFn = Callable[[Path], MujocoAsset]


def cache_dir() -> Path:
    """Root of assetx's user cache (``$ASSETX_CACHE_DIR`` or ``~/.cache/assetx``)."""
    env = os.environ.get("ASSETX_CACHE_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "assetx"


def fetch_vendor(url: str) -> Path:
    """Download a pinned GitHub directory once into the vendor cache.

    ``url`` must reference a commit SHA (``.../tree/<sha>/<dir>``) so the
    cached copy never goes stale. A directory without the fetch metadata
    (interrupted download) is fetched again.
    """
    ref = parse_github_dir_url(url)
    # Keep the vendor directory name as the leaf: save() names mesh folders after it.
    dest = cache_dir() / "vendor" / f"{ref.owner}__{ref.repo}__{ref.ref}" / (ref.path or ref.repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f".{dest.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = (dest / ".assetx_fetch.json").is_file()
        if not complete:
            print(f"Fetching {url} -> {dest}")
        download_github_dir(ref, dest, force=not complete)
    return dest


@dataclass(frozen=True)
class Recipe:
    """A robot build: ``fn(vendor_dir) -> MujocoAsset`` plus save options."""

    name: str
    fn: RecipeFn
    vendor: str
    mesh_format: Literal["stl"] | None = None

    @property
    def source_file(self) -> Path:
        return Path(inspect.getfile(self.fn)).resolve()

    def build(self) -> MujocoAsset:
        return self.fn(fetch_vendor(self.vendor))


_RECIPES: dict[str, Recipe] = {}


def recipe(
    name: str, *, vendor: str, mesh_format: Literal["stl"] | None = None
) -> Callable[[RecipeFn], RecipeFn]:
    """Register ``fn(vendor_dir) -> MujocoAsset`` as recipe ``name``."""
    ref = parse_github_dir_url(vendor)
    if len(ref.ref) != 40 or any(c not in "0123456789abcdef" for c in ref.ref):
        raise ValueError(f"Recipe {name!r}: vendor URL must pin a full commit SHA, got {ref.ref!r}")

    def decorator(fn: RecipeFn) -> RecipeFn:
        if name in _RECIPES:
            raise ValueError(f"Recipe {name!r} is already registered")
        _RECIPES[name] = Recipe(name=name, fn=fn, vendor=vendor, mesh_format=mesh_format)
        return fn

    return decorator


def get_recipe(name: str) -> Recipe:
    try:
        return _RECIPES[name]
    except KeyError:
        raise KeyError(f"Unknown recipe {name!r}; available: {sorted(_RECIPES)}") from None


def list_recipes() -> list[str]:
    return sorted(_RECIPES)
