import argparse
import logging
import time
from typing import NamedTuple, Union

import mujoco
import mujoco.viewer
import mujoco_warp  # kept for future batched GPU-based IK
import numpy as np
import trimesh
import viser

logging.basicConfig(level=logging.INFO)


class MeshObject(NamedTuple):
    geom_id: int
    body_name: str
    geom_name: str
    trimesh: trimesh.Trimesh
    handle: viser.MeshHandle


def _ik_single_position(
    model: mujoco.MjModel,
    eef_body_id: int,
    target_pos: np.ndarray,
    *,
    max_iters: int = 64,
    tol: float = 1e-3,
    step_size: float = 0.5,
) -> tuple[np.ndarray, bool]:
    data = mujoco.MjData(model)
    q = np.zeros(model.nv, dtype=np.float64)
    data.qpos[: model.nv] = q
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    lambda_sq = 1e-4

    for _ in range(max_iters):
        mujoco.mj_jacBody(model, data, jacp, jacr, eef_body_id)
        eef_pos = data.xpos[eef_body_id]
        err = target_pos - eef_pos
        if np.linalg.norm(err) < tol:
            return data.qpos.copy(), True

        jjt = jacp @ jacp.T
        dq = step_size * (jacp.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), err))
        data.qpos[: model.nv] += dq
        mujoco.mj_forward(model, data)

    return data.qpos.copy(), False


def find_workspace(
    spec_or_path: Union[mujoco.MjSpec, str],
    eef_name: str,
    origin_name: str | None = None,
    *,
    radius: float = 0.5,
    num_points_per_axis: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(spec_or_path, str):
        model = mujoco.MjModel.from_xml_path(spec_or_path)
    elif isinstance(spec_or_path, mujoco.MjSpec):
        model = spec_or_path.compile()
    else:
        raise TypeError("spec_or_path must be a path or mujoco.MjSpec")

    if origin_name is None:
        origin_body_id = 0
    else:
        origin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, origin_name)

    eef_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_name)
    logging.info("Origin body ID: %d, EEF body ID: %d", origin_body_id, eef_body_id)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin_pos = data.xpos[origin_body_id].copy()

    axis = np.linspace(-radius, radius, num_points_per_axis, dtype=np.float32)
    x_grid, y_grid, z_grid = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    candidate_positions = origin_pos + offsets

    reachable_mask = np.zeros(len(candidate_positions), dtype=bool)
    configurations = np.zeros((len(candidate_positions), model.nq), dtype=np.float64)
    collision_data = mujoco.MjData(model)
    penetration_eps = 1e-6

    for i, pos in enumerate(candidate_positions):
        qpos, success = _ik_single_position(model, eef_body_id, pos.astype(np.float64))
        if not success:
            continue

        collision_data.qpos[:] = qpos
        collision_data.qvel[:] = 0.0
        mujoco.mj_forward(model, collision_data)
        mujoco.mj_collision(model, collision_data)

        in_collision = False
        for c_idx in range(collision_data.ncon):
            if collision_data.contact[c_idx].dist < -penetration_eps:
                in_collision = True
                break

        if in_collision:
            continue

        reachable_mask[i] = True
        configurations[i] = qpos

    workspace_positions = candidate_positions[reachable_mask]
    workspace_configurations = configurations[reachable_mask]
    logging.info(
        "Workspace computation: %d/%d positions reachable",
        workspace_positions.shape[0],
        candidate_positions.shape[0],
    )
    return workspace_positions, workspace_configurations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--eef", type=str, required=True)
    parser.add_argument("--origin", type=str, default=None)
    args = parser.parse_args()

    spec = mujoco.MjSpec.from_file(str(args.xml_path))
    reachable_positions, _ = find_workspace(spec, args.eef, args.origin)

    model = spec.compile()
    data = mujoco.MjData(model)

    logging.info("Launching viser to visualize %d reachable poses", reachable_positions.shape[0])

    server = viser.ViserServer(port=8080, label="Workspace Visualization")
    if reachable_positions.size > 0:
        pts = reachable_positions.astype(float)
        min_pos = pts.min(axis=0, keepdims=True)
        max_pos = pts.max(axis=0, keepdims=True)
        denom = np.maximum(max_pos - min_pos, 1e-6)
        colors = (pts - min_pos) / denom

        server.scene.add_point_cloud(
            "/workspace",
            points=pts,
            colors=colors,
            point_size=0.01,
        )

        mujoco.mj_forward(model, data)
        meshes: list[MeshObject] = []
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue

            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"

            mesh_id = int(model.geom_dataid[geom_id])
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            fadr = int(model.mesh_faceadr[mesh_id])
            fnum = int(model.mesh_facenum[mesh_id])

            vertices = model.mesh_vert[vadr : (vadr + vnum)] * model.mesh_scale[mesh_id]
            faces = model.mesh_face[fadr : (fadr + fnum)]
            geom_pos = data.geom_xpos[geom_id]
            geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
            world_vertices = (geom_mat @ vertices.T).T + geom_pos

            mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces)
            handle = server.scene.add_mesh_trimesh(
                f"/bodies/{body_name}/{geom_name}",
                mesh=mesh,
            )
            meshes.append(
                MeshObject(
                    geom_id=geom_id,
                    body_name=body_name,
                    geom_name=geom_name,
                    trimesh=mesh,
                    handle=handle,
                )
            )

    print("Viser server running on port 8080. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
