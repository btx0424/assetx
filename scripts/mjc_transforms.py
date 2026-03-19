import mujoco
import mujoco.viewer
from assetx.assemble import assemble, JointCfg
from assetx.transform import (
    MujocoAsset,
    Compose,
    ReplaceCylinderWithCapsule,
    MergeBodies,
    RemoveSubtrees,
    RenameBodies,
)

def test_assemble_asset():
    parent_xml_path = "/home/btx0424/lab51/unitree_ros/robots/a2_description/a2.xml"
    child_xml_path = "/home/btx0424/lab51/piper_isaac_sim/piper_description/mujoco_model/piper_description.xml"

    parent = MujocoAsset.from_file(parent_xml_path)
    child = MujocoAsset.from_file(child_xml_path)
    print(parent.meshdir, child.meshdir)

    assembled_asset = assemble(
        parent,
        child,
        "base_link",
        child_prefix="arm_",
        translation=(0.05, 0.0, 0.10),
        rotation=(0.0, 0.0, 0.0),
        joint_cfg=JointCfg(type="fixed", range=(0, 0)),
    )
    print(assembled_asset)

    transform = Compose([
        ReplaceCylinderWithCapsule(),
        RenameBodies(body_names={"arm_link7": "gripper_left", "arm_link8": "gripper_right"}),
        # RemoveSubtrees(subtree_paths=["gripper_left", "gripper_right"]),
    ])

    transformed_asset = transform.transform(assembled_asset)
    
    model = transformed_asset.spec.compile()
    transformed_asset.save("./assembled", copy_meshes=False)

    model = mujoco.MjModel.from_xml_path("./assembled/model.xml")
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            viewer.sync()


if __name__ == "__main__":
    test_assemble_asset()