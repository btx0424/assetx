"""Tests for GitHub subdirectory URL parsing and download helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
from pathlib import Path

import pytest

from assetx import fetch
from assetx.fetch import GitHubDirRef, download_github_dir, parse_fetch_method, parse_github_dir_url


def test_parse_github_dir_url() -> None:
    ref = parse_github_dir_url(
        "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a2_description"
    )
    assert ref.owner == "unitreerobotics"
    assert ref.repo == "unitree_ros"
    assert ref.ref == "master"
    assert ref.path == "robots/a2_description"


def test_parse_fetch_method() -> None:
    assert parse_fetch_method(None) == "auto"
    assert parse_fetch_method("tarball") == "tarball"
    assert parse_fetch_method("raw") == "http"
    with pytest.raises(ValueError, match="Invalid fetch method"):
        parse_fetch_method("ftp")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "origin"
    (repo / "robot" / "meshes").mkdir(parents=True)
    (repo / "robot" / "model.xml").write_text("<mujoco/>")
    (repo / "robot" / "meshes" / "base.obj").write_text("v 0 0 0")
    (repo / "other").mkdir()
    (repo / "other" / "big.bin").write_text("x" * 1000)
    _git("init", "-q", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=repo)
    # GitHub allows partial clone and fetching commits by SHA; a local repo must opt in.
    _git("config", "uploadpack.allowFilter", "true", cwd=repo)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo)
    return repo, _git("rev-parse", "HEAD", cwd=repo)


def _skip_tarball(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(ref, dest):
        raise RuntimeError("tarball unavailable in test")

    monkeypatch.setattr(fetch, "_download_tarball", fail)


def _http_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(ref):
        raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))

    monkeypatch.setattr(fetch, "_list_subdir_files", fail)


def test_falls_back_to_git_sparse_fetch(tmp_path, origin, monkeypatch) -> None:
    repo, sha = origin
    _skip_tarball(monkeypatch)
    _http_down(monkeypatch)
    monkeypatch.setattr(GitHubDirRef, "git_urls", lambda self: [f"file://{tmp_path}/missing", f"file://{repo}"])

    dest = download_github_dir(GitHubDirRef("o", "r", sha, "robot"), tmp_path / "dest", method="auto")

    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert files == [".assetx_fetch.json", "meshes/base.obj", "model.xml"]
    meta = json.loads((dest / ".assetx_fetch.json").read_text())
    assert meta["method"] == "git" and meta["files"] == 2


def test_explicit_git_skips_tarball(tmp_path, origin, monkeypatch) -> None:
    repo, sha = origin

    def fail_tarball(ref, dest):
        raise AssertionError("tarball should not run when method=git")

    monkeypatch.setattr(fetch, "_download_tarball", fail_tarball)
    monkeypatch.setattr(GitHubDirRef, "git_urls", lambda self: [f"file://{repo}"])

    dest = download_github_dir(GitHubDirRef("o", "r", sha, "robot"), tmp_path / "dest", method="git")
    meta = json.loads((dest / ".assetx_fetch.json").read_text())
    assert meta["method"] == "git"


def test_reports_both_failures(tmp_path, monkeypatch) -> None:
    _skip_tarball(monkeypatch)
    _http_down(monkeypatch)
    monkeypatch.setattr(GitHubDirRef, "git_urls", lambda self: [f"file://{tmp_path}/missing"])

    with pytest.raises(RuntimeError, match=r"(?s)tarball: .*HTTP: .*reset by peer.*git: "):
        download_github_dir(GitHubDirRef("o", "r", "0" * 40, "robot"), tmp_path / "dest")


def test_tarball_extracts_subdirectory(tmp_path, origin, monkeypatch) -> None:
    import tarfile

    repo, sha = origin
    archive = tmp_path / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in (repo / "robot").rglob("*"):
            if path.is_file():
                arcname = f"r-{sha}/robot/{path.relative_to(repo / 'robot').as_posix()}"
                tar.add(path, arcname=arcname)

    def fake_http_download(url: str, dest: Path, *, timeout: float = 30.0) -> None:
        if "codeload.github.com" not in url:
            raise RuntimeError(f"unexpected download URL: {url}")
        shutil.copy2(archive, dest)

    monkeypatch.setattr(fetch, "_http_download", fake_http_download)

    dest = download_github_dir(GitHubDirRef("o", "r", sha, "robot"), tmp_path / "dest")
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert files == [".assetx_fetch.json", "meshes/base.obj", "model.xml"]
    meta = json.loads((dest / ".assetx_fetch.json").read_text())
    assert meta["method"] == "tarball" and meta["files"] == 2
