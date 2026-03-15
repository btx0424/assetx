import assetx
import mujoco
from assetx.transform import MujocoAsset
from assetx.common import JointCfg
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot
import tempfile
import shutil


TMP_DIR = Path(assetx.__path__[0]) / "tmp"
shutil.rmtree(TMP_DIR, ignore_errors=True)


def assemble(
    parent: MujocoAsset,
    child: MujocoAsset,
    parent_link: str,
    child_prefix: str = "child_",
    translation: tuple[float, float, float] = (0, 0, 0),
    rotation: tuple[float, float, float] = (0, 0, 0),
    joint_cfg: JointCfg | None = None
) -> MujocoAsset:

    spec = parent.spec.copy()

    parent_meshdir = (Path(parent.spec.modelfiledir) / parent.meshdir).resolve()
    parent_meshes = [mesh.name for mesh in spec.meshes]
    child_meshdir = (Path(child.spec.modelfiledir) / child.meshdir).resolve()
    child_meshes = [f"{child_prefix}{mesh.name}" for mesh in child.spec.meshes]

    child_root = child.spec.worldbody.first_body()
    frame = spec.body(parent_link).add_frame()
    frame.pos = translation
    frame.quat = sRot.from_euler("xyz", rotation).as_quat(scalar_first=True)
    frame.attach_body(child_root, child_prefix)
    
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
    tmp_path = tempfile.mktemp(dir=tmp_dir, suffix=".xml")
    meshdir = tmp_dir / "meshes"
    meshdir.mkdir(parents=True, exist_ok=True)

    for name in parent_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        mesh.file = str(parent_meshdir / mesh.file)
    for name in child_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        mesh.file = str(child_meshdir / mesh.file)
    
    spec.compile()
    spec.to_file(tmp_path)
    
    spec = mujoco.MjSpec.from_file(tmp_path)
    meshdir = (Path(spec.modelfiledir) / spec.meshdir).resolve()
    (meshdir / parent.spec.modelname).symlink_to(parent_meshdir)
    (meshdir / child.spec.modelname).symlink_to(child_meshdir)

    for name in parent_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        file_name = Path(mesh.file).name
        mesh.file = str(Path(parent.spec.modelname) / file_name)
    
    for name in child_meshes:
        mesh: mujoco.MjsMesh = spec.mesh(name)
        file_name = Path(mesh.file).name
        mesh.file = str(Path(child.spec.modelname) / file_name)
    
    spec.compile()
    spec.to_file(str(tmp_dir / "model.xml"))
    asset = MujocoAsset(str(tmp_dir / "model.xml"), spec, meshdir)
    return asset


if __name__ == "__main__":
    parent_xml_path = "/home/btx0424/lab51/unitree_ros/robots/a2_description/a2.xml"
    child_xml_path = "/home/btx0424/lab51/piper_isaac_sim/piper_description/mujoco_model/piper_description.xml"

    parent = MujocoAsset.from_file(parent_xml_path)
    child = MujocoAsset.from_file(child_xml_path)

    assembled_asset = assemble(
        parent,
        child,
        "base_link",
        child_prefix="arm_",
        translation=(0.05, 0.0, 0.10),
        rotation=(0.0, 0.0, 0.0),
        joint_cfg=None,
    )
    print(assembled_asset)
    assembled_asset.save("./assembled")

    import mujoco.viewer
    model = assembled_asset.spec.compile()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            viewer.sync()

