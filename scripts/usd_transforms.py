from pxr import Usd, UsdGeom
from assetx.conversion import UsdToMjcf


if __name__ == "__main__":
    path = "/home/btx0424/lab51/open_doors_g1_wholebody/assets/doors/door_fixed_handle.usda"
    stage = Usd.Stage.Open(path)
    transform = UsdToMjcf()
    spec = transform.transform(stage)
    print(spec.to_xml())
