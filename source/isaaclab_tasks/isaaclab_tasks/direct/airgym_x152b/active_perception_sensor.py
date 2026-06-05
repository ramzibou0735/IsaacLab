# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

wp.init()

NO_HIT_RAY_VAL = wp.constant(1000.0)


@wp.kernel(enable_backward=False)
def clear_occupancy_map_kernel(occupancy_map: wp.array4d(dtype=wp.float32), env_ids: wp.array(dtype=wp.int32)):
    env_slot, ix, iy, iz = wp.tid()
    env_id = env_ids[env_slot]
    occupancy_map[env_id, ix, iy, iz] = 0.0


@wp.kernel(enable_backward=False)
def clear_depth_kernel(depth: wp.array4d(dtype=wp.float32)):
    env_id, cam_id, y, x = wp.tid()
    depth[env_id, cam_id, y, x] = 0.0


@wp.kernel(enable_backward=False)
def draw_depth_range_mesh_occupancy_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    cam_poss: wp.array2d(dtype=wp.vec3),
    cam_quats: wp.array2d(dtype=wp.quat),
    env_origins: wp.array(dtype=wp.vec3),
    k_inv: wp.mat44,
    far_plane: float,
    pixels: wp.array4d(dtype=wp.float32),
    c_x: int,
    c_y: int,
    calculate_depth: bool,
    occupancy_map: wp.array4d(dtype=wp.float32),
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
):
    env_id, cam_id, x, y = wp.tid()
    mesh = mesh_ids[env_id]
    cam_pos = cam_poss[env_id, cam_id]
    cam_quat = cam_quats[env_id, cam_id]

    cam_coords = wp.vec3(float(x), float(y), 1.0)
    cam_coords_principal = wp.vec3(float(c_x), float(c_y), 1.0)
    uv = wp.transform_vector(k_inv, cam_coords)
    uv_principal = wp.transform_vector(k_inv, cam_coords_principal)

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_principal = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    multiplier = 1.0
    if calculate_depth:
        multiplier = wp.dot(rd, rd_principal)

    dist = NO_HIT_RAY_VAL
    if wp.mesh_query_ray(mesh, ro, rd, far_plane / multiplier, t, u, v, sign, n, f):
        dist = multiplier * t
    pixels[env_id, cam_id, y, x] = dist

    dist_pointcloud = NO_HIT_RAY_VAL
    if wp.mesh_query_ray(mesh, ro, rd, far_plane, t, u, v, sign, n, f):
        dist_pointcloud = t

    min_corner = wp.vec3f(-6.0, -4.0, -3.0)
    grid_size_x_occupancy = float(grid_size_x - 1)
    grid_size_y_occupancy = float(grid_size_y - 1)
    grid_size_z_occupancy = float(grid_size_z - 1)

    distance = dist_pointcloud
    if dist_pointcloud > 5.0:
        distance = 5.0

    free = float(0.0)
    env_origin = env_origins[env_id]
    for _ in range(101):
        point_local = ro + distance * rd * free - env_origin
        voxel_indices_pointcloud = point_local - min_corner
        free += 0.01
        voxel_indices_pointcloud[0] = wp.clamp(
            voxel_indices_pointcloud[0] / 12.0 * grid_size_x_occupancy, 0.0, grid_size_x_occupancy
        )
        voxel_indices_pointcloud[1] = wp.clamp(
            voxel_indices_pointcloud[1] / 8.0 * grid_size_y_occupancy, 0.0, grid_size_y_occupancy
        )
        voxel_indices_pointcloud[2] = wp.clamp(
            voxel_indices_pointcloud[2] / 6.0 * grid_size_z_occupancy, 0.0, grid_size_z_occupancy
        )
        wp.atomic_max(
            occupancy_map,
            env_id,
            int(voxel_indices_pointcloud[0]),
            int(voxel_indices_pointcloud[1]),
            int(voxel_indices_pointcloud[2]),
            1.0,
        )

    point_local = ro + dist_pointcloud * rd - env_origin
    voxel_indices_pointcloud = point_local - min_corner
    voxel_indices_pointcloud[0] = wp.clamp(
        voxel_indices_pointcloud[0] / 12.0 * grid_size_x_occupancy, 0.0, grid_size_x_occupancy
    )
    voxel_indices_pointcloud[1] = wp.clamp(
        voxel_indices_pointcloud[1] / 8.0 * grid_size_y_occupancy, 0.0, grid_size_y_occupancy
    )
    voxel_indices_pointcloud[2] = wp.clamp(
        voxel_indices_pointcloud[2] / 6.0 * grid_size_z_occupancy, 0.0, grid_size_z_occupancy
    )
    wp.atomic_max(
        occupancy_map,
        env_id,
        int(voxel_indices_pointcloud[0]),
        int(voxel_indices_pointcloud[1]),
        int(voxel_indices_pointcloud[2]),
        2.0,
    )


def _resolve_device(device) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _cuboids_to_mesh(
    centers: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_corners = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    base_faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )

    vertices = []
    faces = []
    for box_id in range(centers.shape[0]):
        offset = 8 * box_id
        vertices.append(centers[box_id] + base_corners * half_extents[box_id])
        faces.append(base_faces + offset)
    return np.concatenate(vertices, axis=0).astype(np.float32), np.concatenate(faces, axis=0).astype(np.int32)


class ActivePerceptionWarpSensor:
    """Active-perception-style mesh depth sensor with a persistent 3D occupancy map."""

    def __init__(
        self,
        num_envs: int,
        num_boxes: int,
        *,
        height: int,
        width: int,
        horizontal_fov_deg: float,
        max_range: float,
        map_shape: tuple[int, int, int] = (121, 81, 61),
        device=None,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if int(num_boxes) <= 0:
            raise ValueError(f"num_boxes must be positive, got {num_boxes}")
        self.num_envs = int(num_envs)
        self.num_boxes = int(num_boxes)
        self.height = int(height)
        self.width = int(width)
        self.max_range = float(max_range)
        self.map_shape = tuple(int(v) for v in map_shape)
        self._torch_device = _resolve_device(device)
        self._wp_device = wp.device_from_torch(self._torch_device)
        self._is_cuda = self._torch_device.type == "cuda"

        u_0 = self.width / 2.0
        v_0 = self.height / 2.0
        horizontal_fov = math.radians(horizontal_fov_deg)
        focal = self.width / (2.0 * math.tan(horizontal_fov / 2.0))
        vertical_fov = 2.0 * math.atan(self.height / (2.0 * focal))
        alpha_u = u_0 / math.tan(horizontal_fov / 2.0)
        alpha_v = v_0 / math.tan(vertical_fov / 2.0)
        k = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self._k_inv = wp.inverse(k)
        self._c_x = int(u_0)
        self._c_y = int(v_0)

        dev = self._torch_device
        self._depth = torch.zeros((self.num_envs, 1, self.height, self.width), device=dev, dtype=torch.float32)
        self._occupancy_map = torch.zeros((self.num_envs, *self.map_shape), device=dev, dtype=torch.float32)
        self._cam_pos = torch.zeros((self.num_envs, 1, 3), device=dev, dtype=torch.float32)
        self._cam_quat = torch.zeros((self.num_envs, 1, 4), device=dev, dtype=torch.float32)
        self._cam_quat[..., 3] = 1.0
        self._env_origins = torch.zeros((self.num_envs, 3), device=dev, dtype=torch.float32)
        self._box_centers = torch.zeros((self.num_envs, self.num_boxes, 3), device=dev, dtype=torch.float32)
        self._box_half_extents = torch.zeros((self.num_envs, self.num_boxes, 3), device=dev, dtype=torch.float32)

        self._wp_depth = wp.from_torch(self._depth, dtype=wp.float32)
        self._wp_occupancy_map = wp.from_torch(self._occupancy_map, dtype=wp.float32)
        self._wp_cam_pos = wp.from_torch(self._cam_pos, dtype=wp.vec3)
        self._wp_cam_quat = wp.from_torch(self._cam_quat, dtype=wp.quat)
        self._wp_env_origins = wp.from_torch(self._env_origins, dtype=wp.vec3)

        self._meshes: list[wp.Mesh | None] = [None] * self.num_envs
        self._mesh_ids_host = np.zeros((self.num_envs,), dtype=np.uint64)
        self._mesh_ids = wp.array(self._mesh_ids_host, dtype=wp.uint64, device=self._wp_device)

        self._reset_env_ids_host = np.arange(self.num_envs, dtype=np.int32)
        self._reset_env_ids = wp.array(self._reset_env_ids_host, dtype=wp.int32, device=self._wp_device)

    def set_env_origins(self, env_origins: torch.Tensor) -> None:
        self._env_origins.copy_(env_origins.to(self._torch_device, torch.float32))

    def set_box_half_extents(self, box_half_extents: torch.Tensor) -> None:
        box_half_extents = box_half_extents.to(self._torch_device, torch.float32)
        if box_half_extents.dim() == 2:
            box_half_extents = box_half_extents.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._box_half_extents.copy_(box_half_extents)

    def set_box_transforms(
        self,
        box_center_w: torch.Tensor,
        box_quat_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del box_quat_w
        box_center_w = box_center_w.to(self._torch_device, torch.float32)
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self._torch_device, dtype=torch.long)
            self._box_centers.copy_(box_center_w)
        else:
            env_ids_t = env_ids.to(self._torch_device, dtype=torch.long)
            self._box_centers[env_ids_t] = box_center_w
        self.rebuild_meshes(env_ids_t)

    def rebuild_meshes(self, env_ids: torch.Tensor) -> None:
        env_ids_cpu = env_ids.detach().to("cpu", dtype=torch.long).tolist()
        centers_cpu = self._box_centers.detach().to("cpu").numpy()
        half_cpu = self._box_half_extents.detach().to("cpu").numpy()
        for env_id in env_ids_cpu:
            vertices, faces = _cuboids_to_mesh(centers_cpu[env_id], half_cpu[env_id])
            mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self._wp_device),
                indices=wp.array(faces.flatten(), dtype=wp.int32, device=self._wp_device),
            )
            self._meshes[env_id] = mesh
            self._mesh_ids_host[env_id] = mesh.id
        self._mesh_ids = wp.array(self._mesh_ids_host, dtype=wp.uint64, device=self._wp_device)
        self.reset_maps(env_ids)

    def reset_maps(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._torch_device, dtype=torch.long)
        env_ids_cpu = env_ids.detach().to("cpu", dtype=torch.int32).numpy()
        self._reset_env_ids_host = env_ids_cpu
        self._reset_env_ids = wp.array(env_ids_cpu, dtype=wp.int32, device=self._wp_device)
        dim = (len(env_ids_cpu), *self.map_shape)
        if self._is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
            wp.launch(clear_occupancy_map_kernel, dim=dim, inputs=[self._wp_occupancy_map, self._reset_env_ids], stream=stream)
        else:
            wp.launch(
                clear_occupancy_map_kernel,
                dim=dim,
                inputs=[self._wp_occupancy_map, self._reset_env_ids],
                device=self._wp_device,
            )

    def compute(self, cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor, *_, **__) -> tuple[torch.Tensor, torch.Tensor]:
        if any(mesh is None for mesh in self._meshes):
            missing = [str(i) for i, mesh in enumerate(self._meshes) if mesh is None]
            raise RuntimeError(f"ActivePerceptionWarpSensor meshes have not been built for envs: {', '.join(missing)}")
        self._cam_pos[:, 0].copy_(cam_pos_w.to(self._torch_device, torch.float32))
        self._cam_quat[:, 0].copy_(cam_quat_w.to(self._torch_device, torch.float32)[:, [1, 2, 3, 0]])

        dim = (self.num_envs, 1, self.width, self.height)
        inputs = [
            self._mesh_ids,
            self._wp_cam_pos,
            self._wp_cam_quat,
            self._wp_env_origins,
            self._k_inv,
            float(self.max_range),
            self._wp_depth,
            self._c_x,
            self._c_y,
            True,
            self._wp_occupancy_map,
            self.map_shape[0],
            self.map_shape[1],
            self.map_shape[2],
        ]
        if self._is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(self._torch_device))
            wp.launch(clear_depth_kernel, dim=(self.num_envs, 1, self.height, self.width), inputs=[self._wp_depth], stream=stream)
            wp.launch(draw_depth_range_mesh_occupancy_kernel, dim=dim, inputs=inputs, stream=stream)
        else:
            wp.launch(clear_depth_kernel, dim=(self.num_envs, 1, self.height, self.width), inputs=[self._wp_depth], device=self._wp_device)
            wp.launch(draw_depth_range_mesh_occupancy_kernel, dim=dim, inputs=inputs, device=self._wp_device)
        return self.depth, self.occupancy_map

    @property
    def depth(self) -> torch.Tensor:
        return self._depth[:, 0]

    @property
    def occupancy_map(self) -> torch.Tensor:
        return self._occupancy_map


__all__ = [
    "ActivePerceptionWarpSensor",
    "clear_occupancy_map_kernel",
    "draw_depth_range_mesh_occupancy_kernel",
]
