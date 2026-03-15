import numpy as np
import mujoco
from assetx.transform import Transform
from scipy.spatial.transform import Rotation as sRot

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    raise ImportError("Use `pip install usd-core` to install the USD library.")


class UsdToMjcf(Transform):
    def __init__(self) -> None:
        pass

    def transform(self, stage: Usd.Stage) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        time = Usd.TimeCode.Default()
        root = stage.GetDefaultPrim()
        if not root:
            return spec

        # Step 1: Find all bodies (prims with RigidBodyAPI) and joints (type name contains "Joint")
        bodies: list[Usd.Prim] = []
        joints: list[Usd.Prim] = []
        queue = list(root.GetChildren())
        while queue:
            prim = queue.pop(0)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(prim)
            if "Joint" in prim.GetTypeName():
                joints.append(prim)
            queue.extend(prim.GetChildren())

        # Step 2: Build the kinematic tree from joint body0/body1 relationships
        # Edges (parent_path, child_path); parent_map[child_path] = parent_path
        kinematic_edges: list[tuple[Sdf.Path, Sdf.Path]] = []
        parent_map: dict[Sdf.Path, Sdf.Path] = {}
        for joint in joints:
            rel_body0 = joint.GetRelationship("physics:body0")
            rel_body1 = joint.GetRelationship("physics:body1")
            if not rel_body0 or not rel_body1:
                continue
            targets0 = rel_body0.GetTargets()
            targets1 = rel_body1.GetTargets()
            if len(targets0) == 0 or len(targets1) == 0:
                continue
            path_body0 = targets0[0]
            path_body1 = targets1[0]
            kinematic_edges.append((path_body0, path_body1))
            parent_map[path_body1] = path_body0

        # bodies, kinematic_edges, parent_map ready for building MjSpec body tree
        root_path = set(body.GetPath() for body in bodies) - set(parent_map.keys())
        assert len(root_path) == 1
        
        root = stage.GetPrimAtPath(root_path.pop())
        print(root)

        # root_body = spec.worldbody.add_body()
        # root_body.name = ...
        # root_body.mass = ...

        return spec
