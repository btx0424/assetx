"""Tests for GitHub subdirectory URL parsing helpers."""

from __future__ import annotations

from assetx.fetch import parse_github_dir_url


def test_parse_github_dir_url() -> None:
    ref = parse_github_dir_url(
        "https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a2_description"
    )
    assert ref.owner == "unitreerobotics"
    assert ref.repo == "unitree_ros"
    assert ref.ref == "master"
    assert ref.path == "robots/a2_description"
