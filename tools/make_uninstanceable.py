from __future__ import annotations

from pathlib import Path
from typing import Optional

from pxr import Usd


def _make_uninstanceable_from_root(root_prim: Usd.Prim) -> int:
    """Make a prim and all its descendants uninstanceable.

    Returns the number of prims that were modified.
    """
    modified = 0
    queue = [root_prim]

    while queue:
        prim = queue.pop(0)

        if prim.IsInstance():
            prim.SetInstanceable(False)
            modified += 1

        queue.extend(list(prim.GetChildren()))

    return modified


def make_uninstanceable(stage: Usd.Stage, root_prim: Optional[Usd.Prim] = None) -> int:
    """Disable instancing for all instanced prims under the given root.

    Args:
        stage: The USD stage to operate on.
        root_prim: Optional root prim to start from. If not provided, the
            stage's default prim is used when available, otherwise all
            children of the pseudo-root are traversed.

    Returns:
        The number of prims whose `instanceable` flag was turned off.
    """
    if stage is None:
        raise ValueError("Stage must not be None.")

    modified = 0

    if root_prim is not None:
        if not root_prim:
            raise ValueError("Provided root_prim is not valid.")
        return _make_uninstanceable_from_root(root_prim)

    # Prefer the default prim if set.
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        modified += _make_uninstanceable_from_root(default_prim)
    else:
        # Fallback: traverse all top-level prims under the pseudo-root.
        pseudo_root = stage.GetPseudoRoot()
        for child in pseudo_root.GetChildren():
            modified += _make_uninstanceable_from_root(child)

    return modified


def make_uninstanceable_cli(usd_path: str) -> int:
    """Open a USD file, make all instanced prims uninstanceable, and save it.

    Args:
        usd_path: Path to the USD file on disk.

    Returns:
        The number of prims whose `instanceable` flag was turned off.
    """
    path = Path(usd_path)
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage at: {path}")

    modified = make_uninstanceable(stage)
    stage.GetRootLayer().Save()
    return modified


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Make instanced prims in a USD file uninstanceable.")
    parser.add_argument("--usd", type=str, required=True, help="Path to USD file to modify.")
    args = parser.parse_args()

    modified = make_uninstanceable_cli(args.usd)
    print(f"Made {modified} prim(s) uninstanceable in '{args.usd}'.")


if __name__ == "__main__":
    main()
