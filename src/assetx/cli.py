"""``assetx`` command line: run example scripts under ``examples/``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="assetx",
        description=(
            "MJCF compose/transform toolkit. Build robots with scripts in "
            "examples/ (e.g. examples/spot.py). Downstream projects register "
            "cook recipes in active-adaptation and use aa-cook-assets."
        ),
    )
    parser.parse_args(argv)
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
