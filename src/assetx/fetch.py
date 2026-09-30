"""Download a single GitHub directory without cloning the full repository."""

from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_GITHUB_TREE_URL = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/tree/(?P<ref>[^/]+)(?:/(?P<path>.*))?$"
)
_USER_AGENT = "assetx-fetch/0.1"


@dataclass(frozen=True)
class GitHubDirRef:
    owner: str
    repo: str
    ref: str
    path: str = ""

    @property
    def cache_key(self) -> str:
        slug = self.path.strip("/").replace("/", "__") or "_root"
        return f"{self.owner}__{self.repo}__{self.ref}__{slug}"

    def page_url(self) -> str:
        base = f"https://github.com/{self.owner}/{self.repo}/tree/{self.ref}"
        path = self.path.strip("/")
        return f"{base}/{path}" if path else base


def parse_github_dir_url(url: str) -> GitHubDirRef:
    """Parse a GitHub directory URL (``.../tree/<ref>/<path>``)."""
    match = _GITHUB_TREE_URL.match(url.strip().rstrip("/"))
    if match is None:
        raise ValueError(
            "Expected a GitHub directory URL like "
            "'https://github.com/org/repo/tree/main/path/to/dir', "
            f"got {url!r}"
        )
    return GitHubDirRef(
        owner=match.group("owner"),
        repo=match.group("repo"),
        ref=match.group("ref"),
        path=(match.group("path") or "").strip("/"),
    )


def _http_get_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API request failed ({exc.code}) for {url}: {detail}"
        ) from exc


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _list_subdir_files(ref: GitHubDirRef) -> list[str]:
    """Return repo-relative file paths under ``ref.path`` via the Trees API."""
    tree_url = (
        f"https://api.github.com/repos/{ref.owner}/{ref.repo}"
        f"/git/trees/{urllib.parse.quote(ref.ref)}?recursive=1"
    )
    payload = _http_get_json(tree_url)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected GitHub tree response for {tree_url}")
    if payload.get("truncated"):
        raise RuntimeError(
            f"GitHub tree listing for {ref.owner}/{ref.repo}@{ref.ref} was truncated; "
            "subdir is too large for a single Trees API response."
        )
    entries = payload.get("tree")
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected GitHub tree payload for {tree_url}")

    prefix = ref.path.strip("/")
    files: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        if prefix:
            if path == prefix or path.startswith(prefix + "/"):
                files.append(path)
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No files found under {ref.page_url()} "
            f"({ref.owner}/{ref.repo}@{ref.ref}:{prefix or '/'})"
        )
    return files


def download_github_dir(
    source: str | GitHubDirRef,
    dest: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Download only one GitHub directory into ``dest`` (no git history).

    ``source`` may be a :class:`GitHubDirRef` or a directory URL such as
    ``https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a2_description``.

    Uses one recursive Trees API listing, then downloads each blob from
    ``raw.githubusercontent.com``. Skips work when ``dest`` already contains
    files unless ``force=True``.
    """
    ref = parse_github_dir_url(source) if isinstance(source, str) else source
    dest_path = Path(dest)
    if dest_path.exists() and any(dest_path.rglob("*")) and not force:
        return dest_path

    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    prefix = ref.path.strip("/")
    files = _list_subdir_files(ref)
    for repo_path in files:
        rel = repo_path[len(prefix) :].lstrip("/") if prefix else repo_path
        if not rel:
            continue
        raw_url = (
            f"https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/"
            f"{urllib.parse.quote(ref.ref)}/{urllib.parse.quote(repo_path, safe='/')}"
        )
        _http_download(raw_url, dest_path / rel)

    meta = {
        "source": ref.page_url(),
        "owner": ref.owner,
        "repo": ref.repo,
        "ref": ref.ref,
        "path": ref.path,
        "files": len(files),
    }
    (dest_path / ".assetx_fetch.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return dest_path
