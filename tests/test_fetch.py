"""Tests for GitHub subdirectory URL parsing helpers."""

from __future__ import annotations

from pathlib import Path

from assetx.fetch import find_mjcf, parse_github_dir_url


def test_parse_github_dir_url() -> None:
    ref = parse_github_dir_url(
        "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a2_description"
    )
    assert ref.owner == "unitreerobotics"
    assert ref.repo == "unitree_ros"
    assert ref.ref == "master"
    assert ref.path == "robots/a2_description"


def test_find_mjcf_prefers_named_file(tmp_path: Path) -> None:
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    target = tmp_path / "a2.xml"
    target.write_text("<mujoco/>", encoding="utf-8")
    assert find_mjcf(tmp_path, preferred_names=["a2.xml"]) == target
