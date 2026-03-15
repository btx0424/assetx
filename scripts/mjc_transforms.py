import mujoco
import mujoco.viewer
from assetx.compose import assemble, JointCfg
from assetx.transform import (
    MujocoAsset,
    Compose,
    ReplaceCylinderWithCapsule,
    MergeBodies,
    RemoveSubtrees,
    RenameBodies,
)

def test_compose_specs():
    parent_xml_path = "/home/btx0424/lab51/unitree_ros/robots/a2_description/a2.xml"
    child_xml_path = "/home/btx0424/lab51/piper_isaac_sim/piper_description/mujoco_model/piper_description.xml"

    parent = MujocoAsset.from_file(parent_xml_path)
    child = MujocoAsset.from_file(child_xml_path)
    print(parent.meshdir, child.meshdir)

    asset = assemble(
        parent,
        child,
        "base_link",
        child_prefix="arm_",
        translation=(0.05, 0.0, 0.10),
        rotation=(0.0, 0.0, 0.0),
        joint_cfg=JointCfg(type="fixed", range=(0, 0)),
    )

    transform = Compose([
        # ReplaceCylinderWithCapsule(),
        # MergeBodies(parent_path="base_link", child_path="arm_base_link"),
        # RenameBodies(body_names={"arm_link7": "gripper_left", "arm_link8": "gripper_right"}),
    ])

    new_asset = transform.transform(asset)
    model = new_asset.spec.compile()
    new_asset.spec.to_file("assembled.xml")

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            viewer.sync()
    
    import mujoco_usd_converter
    import usdex.core
    from pxr import Sdf, Usd

    converter = mujoco_usd_converter.Converter()
    asset: Sdf.AssetPath = converter.convert("assembled.xml", "assembled.usda")
    stage: Usd.Stage = Usd.Stage.Open(asset.path)
    # modify further using Usd or usdex.core functionality
    usdex.core.saveStage(stage)


if __name__ == "__main__":
    test_compose_specs()