"""Pinned vendor download cache (used by example scripts and downstream cookers)."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from assetx.fetch import download_github_dir, parse_github_dir_url


def cache_dir() -> Path:
    """Root of assetx's user cache (``$ASSETX_CACHE_DIR`` or ``~/.cache/assetx``)."""
    env = os.environ.get("ASSETX_CACHE_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "assetx"


def fetch_vendor(url: str, *, method: str | None = None) -> Path:
    """Download a pinned GitHub directory once into the vendor cache.

    ``url`` must reference a commit SHA (``.../tree/<sha>/<dir>``).
    ``method`` / ``$ASSETX_FETCH``: ``auto``, ``tarball``, ``http``, or ``git``.
    """
    ref = parse_github_dir_url(url)
    dest = cache_dir() / "vendor" / f"{ref.owner}__{ref.repo}__{ref.ref}" / (ref.path or ref.repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f".{dest.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = (dest / ".assetx_fetch.json").is_file()
        if not complete:
            print(f"Fetching {url} -> {dest}")
        download_github_dir(ref, dest, force=not complete, method=method)
    return dest
