import mujoco_usd_converter
import usdex.core
import argparse
from pxr import Sdf, Usd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    path = args.path
    if not path.endswith(".xml"):
        raise ValueError("Path must end with .xml")
    output_dir = Path(path).resolve().parent

    converter = mujoco_usd_converter.Converter()
    asset: Sdf.AssetPath = converter.convert(
        input_file=path,
        output_dir=output_dir,
    )
    stage: Usd.Stage = Usd.Stage.Open(asset.path)
    usdex.core.saveStage(stage)


if __name__ == "__main__":
    main()
