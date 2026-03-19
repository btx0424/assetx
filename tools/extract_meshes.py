import argparse
from pathlib import Path

from assetx.utils import extract_meshes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Mesh and Cube prims from a USD file and export as STL or OBJ.",
    )
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the USD file.")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for mesh files.",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="stl",
        choices=["stl", "obj"],
        help="Output mesh format (default: stl).",
    )
    args = parser.parse_args()

    meshes = extract_meshes(args.path)
    if not meshes:
        print("No mesh or cube prims found.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format.lower()
    for name, mesh in meshes.items():
        out_path = out_dir / f"{name}.{ext}"
        mesh.export(str(out_path))
        print(out_path)


if __name__ == "__main__":
    main()
