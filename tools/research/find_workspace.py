import argparse
import logging
import time
from typing import NamedTuple, Union

import mujoco
import mujoco.viewer
import mujoco_warp  # kept for future batched GPU-based IK
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
import trimesh
import viser

logging.basicConfig(level=logging.INFO)


class MeshObject(NamedTuple):
    geom_id: int
    body_name: str
    geom_name: str
    trimesh: trimesh.Trimesh
    handle: viser.MeshHandle


class WorkspaceResult(NamedTuple):
    positions: np.ndarray
    configurations: np.ndarray
    graph_points: np.ndarray
    graph_edges: np.ndarray
    graph_configurations: np.ndarray


def _apply_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if not model.jnt_limited[joint_id]:
            continue

        qpos_adr = model.jnt_qposadr[joint_id]
        lower, upper = model.jnt_range[joint_id]
        qpos[qpos_adr] = np.clip(qpos[qpos_adr], lower, upper)


def _is_in_collision(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    penetration_eps: float,
) -> bool:
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    for contact_idx in range(data.ncon):
        if data.contact[contact_idx].dist < -penetration_eps:
            return True
    return False


def _configuration_delta(
    model: mujoco.MjModel,
    qpos_from: np.ndarray,
    qpos_to: np.ndarray,
) -> np.ndarray:
    qvel = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel, 1.0, qpos_from, qpos_to)
    return qvel


def _configuration_distance(
    model: mujoco.MjModel,
    qpos_from: np.ndarray,
    qpos_to: np.ndarray,
) -> float:
    return float(np.linalg.norm(_configuration_delta(model, qpos_from, qpos_to)))


def _edge_is_collision_free(
    model: mujoco.MjModel,
    qpos_start: np.ndarray,
    qpos_goal: np.ndarray,
    *,
    max_step_size: float,
    penetration_eps: float,
) -> bool:
    delta = _configuration_delta(model, qpos_start, qpos_goal)
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_start
        return not _is_in_collision(model, data, penetration_eps=penetration_eps)

    steps = max(1, int(np.ceil(distance / max_step_size)))
    data = mujoco.MjData(model)

    for step_idx in range(steps + 1):
        alpha = step_idx / steps
        qpos = qpos_start.copy()
        mujoco.mj_integratePos(model, qpos, delta, alpha)
        _apply_joint_limits(model, qpos)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        if _is_in_collision(model, data, penetration_eps=penetration_eps):
            return False

    return True


def _geom_wxyz(data: mujoco.MjData, geom_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.geom_xmat[geom_id])
    return quat


def _update_mesh_handles(
    data: mujoco.MjData,
    meshes: list[MeshObject],
) -> None:
    for mesh_obj in meshes:
        mesh_obj.handle.position = data.geom_xpos[mesh_obj.geom_id]
        mesh_obj.handle.wxyz = _geom_wxyz(data, mesh_obj.geom_id)


def _sample_graph_path(
    num_nodes: int,
    graph_edges: np.ndarray,
    rng: np.random.Generator,
    *,
    max_nodes: int = 8,
) -> list[int]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src_idx, dst_idx in graph_edges:
        adjacency[int(src_idx)].append(int(dst_idx))
        adjacency[int(dst_idx)].append(int(src_idx))

    if num_nodes == 0:
        return []
    if num_nodes == 1:
        return [0]

    path = [0]
    prev_idx = -1
    while len(path) < max_nodes:
        neighbors = adjacency[path[-1]]
        if not neighbors:
            break

        candidates = [idx for idx in neighbors if idx != prev_idx]
        if not candidates:
            candidates = neighbors
        next_idx = int(rng.choice(candidates))
        path.append(next_idx)
        prev_idx = path[-2]

    return path


def _ik_single_position(
    model: mujoco.MjModel,
    eef_body_id: int,
    target_pos: np.ndarray,
    *,
    initial_qpos: np.ndarray | None = None,
    max_iters: int = 64,
    tol: float = 1e-3,
    step_size: float = 0.5,
) -> tuple[np.ndarray, bool]:
    data = mujoco.MjData(model)
    if initial_qpos is None:
        data.qpos[:] = model.qpos0
    else:
        data.qpos[:] = initial_qpos
    _apply_joint_limits(model, data.qpos)
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
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
        _apply_joint_limits(model, data.qpos)
        mujoco.mj_forward(model, data)

    return data.qpos.copy(), False


def _find_workspace_result(
    spec_or_path: Union[mujoco.MjSpec, str],
    eef_name: str,
    origin_name: str | None = None,
    *,
    radius: float = 0.5,
    num_points_per_axis: int = 10,
    connectivity_neighbors: int = 12,
    edge_step_size: float = 0.1,
    max_neighbor_distance: float | None = None,
) -> WorkspaceResult:
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
    home_eef_pos = data.xpos[eef_body_id].copy()

    axis = np.linspace(-radius, radius, num_points_per_axis, dtype=np.float32)
    x_grid, y_grid, z_grid = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    candidate_positions = origin_pos + offsets

    penetration_eps = 1e-6
    home_qpos = model.qpos0.copy()
    reachable_mask = np.zeros(len(candidate_positions), dtype=bool)
    configurations = np.zeros((len(candidate_positions), model.nq), dtype=np.float64)

    ik_progress = tqdm(candidate_positions, desc="IK samples", unit="sample")
    for i, pos in enumerate(ik_progress):
        qpos, success = _ik_single_position(
            model,
            eef_body_id,
            pos.astype(np.float64),
            initial_qpos=home_qpos,
        )
        if not success:
            continue

        collision_data = mujoco.MjData(model)
        collision_data.qpos[:] = qpos
        collision_data.qvel[:] = 0.0
        if _is_in_collision(model, collision_data, penetration_eps=penetration_eps):
            continue

        reachable_mask[i] = True
        configurations[i] = qpos

    workspace_positions = candidate_positions[reachable_mask]
    workspace_configurations = configurations[reachable_mask]
    if workspace_positions.size == 0:
        logging.info("Workspace computation: 0/%d positions reachable", candidate_positions.shape[0])
        return WorkspaceResult(
            positions=workspace_positions,
            configurations=workspace_configurations,
            graph_points=home_eef_pos[None, :],
            graph_edges=np.zeros((0, 2), dtype=np.int32),
            graph_configurations=home_qpos[None, :],
        )

    graph_configs = np.vstack([home_qpos[None, :], workspace_configurations])
    neighbor_count = min(connectivity_neighbors + 1, graph_configs.shape[0])
    kd_tree = cKDTree(graph_configs)
    _, neighbor_indices = kd_tree.query(graph_configs, k=neighbor_count)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    adjacency: list[set[int]] = [set() for _ in range(graph_configs.shape[0])]
    checked_edges: set[tuple[int, int]] = set()
    graph_positions = np.vstack([home_eef_pos[None, :], workspace_positions])

    edge_progress = tqdm(neighbor_indices, desc="Graph edges", unit="node")
    for node_idx, neighbors in enumerate(edge_progress):
        for neighbor_idx in neighbors:
            if neighbor_idx == node_idx:
                continue

            edge = (min(node_idx, int(neighbor_idx)), max(node_idx, int(neighbor_idx)))
            if edge in checked_edges:
                continue
            checked_edges.add(edge)

            qpos_a = graph_configs[edge[0]]
            qpos_b = graph_configs[edge[1]]
            distance = _configuration_distance(model, qpos_a, qpos_b)
            if max_neighbor_distance is not None and distance > max_neighbor_distance:
                continue
            if not _edge_is_collision_free(
                model,
                qpos_a,
                qpos_b,
                max_step_size=edge_step_size,
                penetration_eps=penetration_eps,
            ):
                continue

            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])

    connected_to_home = np.zeros(graph_configs.shape[0], dtype=bool)
    stack = [0]
    connected_to_home[0] = True
    while stack:
        node_idx = stack.pop()
        for neighbor_idx in adjacency[node_idx]:
            if connected_to_home[neighbor_idx]:
                continue
            connected_to_home[neighbor_idx] = True
            stack.append(neighbor_idx)

    workspace_connected_mask = connected_to_home[1:]
    workspace_positions = workspace_positions[workspace_connected_mask]
    workspace_configurations = workspace_configurations[workspace_connected_mask]
    node_id_map = -np.ones(graph_configs.shape[0], dtype=np.int32)
    node_id_map[np.flatnonzero(connected_to_home)] = np.arange(np.count_nonzero(connected_to_home), dtype=np.int32)

    graph_edge_list: list[tuple[int, int]] = []
    for src_idx, neighbors in enumerate(adjacency):
        if not connected_to_home[src_idx]:
            continue
        for dst_idx in neighbors:
            if src_idx >= dst_idx or not connected_to_home[dst_idx]:
                continue
            graph_edge_list.append((int(node_id_map[src_idx]), int(node_id_map[dst_idx])))

    graph_points = graph_positions[connected_to_home]
    graph_edges = (
        np.asarray(graph_edge_list, dtype=np.int32)
        if graph_edge_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    logging.info(
        "Workspace computation: %d/%d positions reachable and connected",
        workspace_positions.shape[0],
        candidate_positions.shape[0],
    )
    return WorkspaceResult(
        positions=workspace_positions,
        configurations=workspace_configurations,
        graph_points=graph_points,
        graph_edges=graph_edges,
        graph_configurations=graph_configs[connected_to_home],
    )


def find_workspace(
    spec_or_path: Union[mujoco.MjSpec, str],
    eef_name: str,
    origin_name: str | None = None,
    *,
    radius: float = 0.5,
    num_points_per_axis: int = 10,
    connectivity_neighbors: int = 12,
    edge_step_size: float = 0.1,
    max_neighbor_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = _find_workspace_result(
        spec_or_path,
        eef_name,
        origin_name,
        radius=radius,
        num_points_per_axis=num_points_per_axis,
        connectivity_neighbors=connectivity_neighbors,
        edge_step_size=edge_step_size,
        max_neighbor_distance=max_neighbor_distance,
    )
    return result.positions, result.configurations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--eef", type=str, required=True)
    parser.add_argument("--origin", type=str, default=None)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--num_points_per_axis", type=int, default=10)
    parser.add_argument("--connectivity_neighbors", type=int, default=12)
    parser.add_argument("--edge_step_size", type=float, default=0.1)
    parser.add_argument("--max_neighbor_distance", type=float, default=None)
    args = parser.parse_args()

    spec = mujoco.MjSpec.from_file(str(args.xml_path))
    workspace_result = _find_workspace_result(
        spec,
        args.eef,
        args.origin,
        radius=args.radius,
        num_points_per_axis=args.num_points_per_axis,
        connectivity_neighbors=args.connectivity_neighbors,
        edge_step_size=args.edge_step_size,
        max_neighbor_distance=args.max_neighbor_distance,
    )
    reachable_positions = workspace_result.positions

    model = spec.compile()
    data = mujoco.MjData(model)

    logging.info("Launching viser to visualize %d reachable poses", reachable_positions.shape[0])

    server = viser.ViserServer(port=8080, label="Workspace Visualization")
    meshes: list[MeshObject] = []
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
        if workspace_result.graph_edges.size > 0:
            edge_points = workspace_result.graph_points[workspace_result.graph_edges]
            server.scene.add_line_segments(
                "/workspace_graph",
                points=edge_points.astype(np.float32),
                colors=np.array([30, 30, 30], dtype=np.uint8),
                line_width=2.0,
            )
        if workspace_result.graph_points.size > 0:
            server.scene.add_point_cloud(
                "/workspace_graph_nodes",
                points=workspace_result.graph_points.astype(np.float32),
                colors=np.tile(np.array([[255, 160, 0]], dtype=np.uint8), (workspace_result.graph_points.shape[0], 1)),
                point_size=0.012,
            )

        mujoco.mj_forward(model, data)
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

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            handle = server.scene.add_mesh_trimesh(
                f"/bodies/{body_name}/{geom_name}",
                mesh=mesh,
                position=data.geom_xpos[geom_id],
                wxyz=_geom_wxyz(data, geom_id),
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
        _update_mesh_handles(data, meshes)

    print("Viser server running on port 8080. Press Ctrl+C to exit.")
    rng = np.random.default_rng()
    active_path_handle: viser.LineSegmentsHandle | None = None
    try:
        while True:
            if workspace_result.graph_configurations.shape[0] <= 1 or not meshes:
                time.sleep(1)
                continue

            node_path = _sample_graph_path(
                workspace_result.graph_configurations.shape[0],
                workspace_result.graph_edges,
                rng,
            )
            if len(node_path) >= 2:
                path_points = workspace_result.graph_points[np.asarray(node_path, dtype=np.int32)]
                if active_path_handle is not None:
                    active_path_handle.remove()
                active_path_handle = server.scene.add_line_segments(
                    "/active_path",
                    points=np.stack([path_points[:-1], path_points[1:]], axis=1).astype(np.float32),
                    colors=np.array([220, 80, 60], dtype=np.uint8),
                    line_width=5.0,
                )

            for src_idx, dst_idx in zip(node_path[:-1], node_path[1:]):
                qpos_start = workspace_result.graph_configurations[src_idx]
                qpos_goal = workspace_result.graph_configurations[dst_idx]
                delta = _configuration_delta(model, qpos_start, qpos_goal)
                distance = float(np.linalg.norm(delta))
                steps = max(2, int(np.ceil(distance / 0.05)))

                for step_idx in range(steps + 1):
                    alpha = step_idx / steps
                    qpos = qpos_start.copy()
                    mujoco.mj_integratePos(model, qpos, delta, alpha)
                    data.qpos[:] = qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    _update_mesh_handles(data, meshes)
                    time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
