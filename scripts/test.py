import mujoco
import mujoco.viewer
from pathlib import Path

if __name__ == "__main__":
    parent_xml_path = "/home/btx0424/lab51/unitree_ros/robots/a2_description/a2.xml"
    child_xml_path = "/home/btx0424/lab51/piper_isaac_sim/piper_description/mujoco_model/piper_description.xml"

    parent = mujoco.MjSpec.from_file(parent_xml_path)
    child = mujoco.MjSpec.from_file(child_xml_path)

    spec = parent.copy()
    parent_meshes = list(spec.meshes)
    
    child_root = child.worldbody.first_body()
    handle = spec.worldbody.add_frame().attach_body(child_root, "child_")

    parent_meshdir = (Path(parent.modelfiledir) / parent.meshdir).resolve()
    child_meshdir = (Path(child.modelfiledir) / child.meshdir).resolve()
    print(parent_meshdir, child_meshdir)

    for mesh in parent_meshes:
        mesh: mujoco.MjsMesh
        file_path = parent_meshdir / mesh.file
        file_path = file_path.resolve()
        mesh.file = str(file_path)
        print(mesh.file)
    
    for mesh in child.meshes:
        mesh: mujoco.MjsMesh
        file_path = child_meshdir / mesh.file
        file_path = file_path.resolve()
        mesh.file = str(file_path)
        print(mesh.file)
    
    spec.compile()
    
    file_path = "model.xml"
    spec.to_file(file_path)
    
    mujoco.MjSpec.from_file(file_path).compile()

