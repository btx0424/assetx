"""``assetx`` command line: list, cook, check and preview registered recipes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from assetx.cook import check, cook
from assetx.recipes import get_recipe, list_recipes


def _names(names: list[str]) -> list[str]:
    for name in names:
        get_recipe(name)
    return names or list_recipes()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="assetx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered recipes.")

    for cmd, help_text in (
        ("cook", "Build bundles (skips ones that are up to date)."),
        ("status", "Report whether bundles are up to date."),
    ):
        p = sub.add_parser(cmd, help=help_text)
        p.add_argument("names", nargs="*", help="Recipes (default: all).")
        p.add_argument(
            "--out-root",
            type=Path,
            default=Path("artifacts"),
            help="Bundles go to OUT_ROOT/<name> (default: artifacts).",
        )
        p.add_argument("--no-usd", action="store_true", help="Skip the USD export.")
        if cmd == "cook":
            p.add_argument("--force", action="store_true", help="Rebuild even if up to date.")

    p = sub.add_parser("preview", help="Build a recipe in memory and open a viewer.")
    p.add_argument("name")
    p.add_argument("--viser", action="store_true", help="Viser joint-slider rig preview.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-ground", action="store_true")
    p.add_argument("--no-lighting", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list":
        for name in list_recipes():
            print(f"{name}\t{get_recipe(name).vendor}")
    elif args.command == "status":
        stale = False
        for name in _names(args.names):
            reason = check(name, args.out_root / name, usd=not args.no_usd)
            stale |= reason is not None
            print(f"{name}: {reason or 'up to date'}")
        sys.exit(1 if stale else 0)
    elif args.command == "cook":
        for name in _names(args.names):
            out = args.out_root / name
            reason = "forced" if args.force else check(name, out, usd=not args.no_usd)
            if reason is None:
                print(f"{name}: up to date ({out})")
                continue
            print(f"{name}: cooking ({reason}) -> {out}")
            cook(name, out, usd=not args.no_usd, force=args.force)
    elif args.command == "preview":
        from assetx import launch_preview

        launch_preview(
            get_recipe(args.name).build(),
            ground=not args.no_ground,
            lighting=not args.no_lighting,
            viewer="viser" if args.viser else "mujoco",
            port=args.port,
        )


if __name__ == "__main__":
    main()
