# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from isaaclab.utils.occupancy import (
    depth_to_point_cloud,
    inflate_occupancy_grid,
    pointcloud_to_occupancy_bev,
    subsample_depth_and_intrinsics,
)


def _sample_uniform_xy(
    num_envs: int,
    device: torch.device | str,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> torch.Tensor:
    """Sample batched x-y positions uniformly inside a rectangular region."""
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("Expected strictly increasing sampling limits.")

    xy = torch.rand((num_envs, 2), device=device, dtype=torch.float32)
    xy[:, 0] = x_min + (x_max - x_min) * xy[:, 0]
    xy[:, 1] = y_min + (y_max - y_min) * xy[:, 1]
    return xy


def sample_spawn_positions(
    num_envs: int,
    device: torch.device | str,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_height: float,
    xy_margin: float,
    z_jitter: float = 0.0,
) -> torch.Tensor:
    """Sample spawn positions inside the room with a wall margin."""
    xy = _sample_uniform_xy(
        num_envs,
        device,
        x_limits=(x_limits[0] + xy_margin, x_limits[1] - xy_margin),
        y_limits=(y_limits[0] + xy_margin, y_limits[1] - xy_margin),
    )
    z = torch.full((num_envs, 1), z_height, device=device, dtype=torch.float32)
    if z_jitter > 0.0:
        z += z_jitter * (2.0 * torch.rand_like(z) - 1.0)
    return torch.cat((xy, z), dim=-1)


def sample_yaws(num_envs: int, device: torch.device | str) -> torch.Tensor:
    """Sample batched yaw angles in ``[-pi, pi]``."""
    return (2.0 * math.pi) * torch.rand((num_envs,), device=device, dtype=torch.float32) - math.pi


def sample_pillar_positions(
    num_envs: int,
    num_pillars: int,
    device: torch.device | str,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    spawn_xy: torch.Tensor,
    wall_margin: float,
    pillar_spacing: float,
    spawn_clearance: float,
    max_tries: int = 64,
) -> torch.Tensor:
    """Sample pillar positions with simple rejection to preserve spacing."""
    if num_pillars < 0:
        raise ValueError(f"Expected num_pillars >= 0. Received: {num_pillars}.")
    if spawn_xy.shape != (num_envs, 2):
        raise ValueError(f"Expected spawn_xy shape ({num_envs}, 2). Got: {spawn_xy.shape}.")

    if num_pillars == 0:
        return torch.zeros((num_envs, 0, 2), device=device, dtype=torch.float32)

    centers = torch.zeros((num_envs, num_pillars, 2), device=device, dtype=torch.float32)
    inner_x_limits = (x_limits[0] + wall_margin, x_limits[1] - wall_margin)
    inner_y_limits = (y_limits[0] + wall_margin, y_limits[1] - wall_margin)

    for pillar_id in range(num_pillars):
        assigned = torch.zeros((num_envs,), device=device, dtype=torch.bool)
        fallback_candidate = None

        for _ in range(max_tries):
            candidate = _sample_uniform_xy(num_envs, device, x_limits=inner_x_limits, y_limits=inner_y_limits)
            fallback_candidate = candidate

            valid = torch.linalg.vector_norm(candidate - spawn_xy, dim=-1) >= spawn_clearance
            if pillar_id > 0:
                previous = centers[:, :pillar_id]
                separation = torch.linalg.vector_norm(candidate.unsqueeze(1) - previous, dim=-1).amin(dim=-1)
                valid &= separation >= pillar_spacing

            accepted = (~assigned) & valid
            if torch.any(accepted):
                centers[accepted, pillar_id] = candidate[accepted]
                assigned |= accepted
            if bool(torch.all(assigned)):
                break

        if not bool(torch.all(assigned)):
            if fallback_candidate is None:
                fallback_candidate = _sample_uniform_xy(
                    num_envs, device, x_limits=inner_x_limits, y_limits=inner_y_limits
                )
            centers[~assigned, pillar_id] = fallback_candidate[~assigned]

    return centers


def build_room_object_state(
    wall_xy: torch.Tensor,
    pillar_xy: torch.Tensor,
    *,
    wall_height: float,
    pillar_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose local object states for fixed walls and randomized pillars."""
    if wall_xy.dim() != 2 or wall_xy.shape[-1] != 2:
        raise ValueError(f"Expected wall_xy shape (W, 2). Got: {wall_xy.shape}.")
    if pillar_xy.dim() != 3 or pillar_xy.shape[-1] != 2:
        raise ValueError(f"Expected pillar_xy shape (N, P, 2). Got: {pillar_xy.shape}.")

    num_envs = pillar_xy.shape[0]
    num_walls = wall_xy.shape[0]
    num_pillars = pillar_xy.shape[1]
    device = pillar_xy.device

    pos = torch.zeros((num_envs, num_walls + num_pillars, 3), device=device, dtype=torch.float32)
    quat = torch.zeros((num_envs, num_walls + num_pillars, 4), device=device, dtype=torch.float32)
    quat[..., 0] = 1.0

    pos[:, :num_walls, :2] = wall_xy.unsqueeze(0)
    pos[:, :num_walls, 2] = 0.5 * wall_height
    pos[:, num_walls:, :2] = pillar_xy
    pos[:, num_walls:, 2] = 0.5 * pillar_height
    return pos, quat


def build_critic_grid_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    camera_pos_w: torch.Tensor,
    camera_quat_ros: torch.Tensor,
    root_pos_w: torch.Tensor,
    world_to_local: torch.Tensor,
    camera_max_distance: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    stride: int = 1,
    near_clip: float = 0.05,
    hit_depth_margin: float = 1.0e-3,
    free_samples_per_ray: int | None = None,
    inflation_radius: int = 1,
) -> torch.Tensor:
    """Build a two-channel critic occupancy grid from metric depth and camera pose data."""
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth_image = depth[:, 0]
    elif depth.dim() == 4 and depth.shape[-1] == 1:
        depth_image = depth[..., 0]
    elif depth.dim() == 3:
        depth_image = depth
    else:
        raise ValueError(f"Expected depth with shape (N, 1, H, W), (N, H, W, 1), or (N, H, W). Got: {depth.shape}.")

    depth_sub, _ = subsample_depth_and_intrinsics(depth_image, intrinsics, stride=stride)
    if depth_sub.dim() == 4 and depth_sub.shape[-1] == 1:
        depth_sub = depth_sub[..., 0]
    ray_depth = depth_sub.transpose(1, 2).reshape(depth_sub.shape[0], -1)

    free_mask = torch.isfinite(ray_depth) & (ray_depth > near_clip)
    occupied_mask = free_mask & (ray_depth < camera_max_distance - hit_depth_margin)

    points_w = depth_to_point_cloud(
        depth_image,
        intrinsics,
        stride=stride,
        is_ortho=True,
        position=camera_pos_w,
        orientation=camera_quat_ros,
    )
    points_local = torch.einsum("bij,bpj->bpi", world_to_local, points_w - root_pos_w.unsqueeze(1))
    camera_pos_local = torch.einsum("bij,bj->bi", world_to_local, camera_pos_w - root_pos_w)

    occupied = pointcloud_to_occupancy_bev(
        points_local.masked_fill(~occupied_mask.unsqueeze(-1), float("nan")),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
    )

    if free_samples_per_ray is None:
        max_range = max(
            abs(x_limits[0]),
            abs(x_limits[1]),
            abs(y_limits[0]),
            abs(y_limits[1]),
            abs(z_limits[0]),
            abs(z_limits[1]),
        )
        free_samples_per_ray = max(2, int(math.ceil(max_range / cell_size)))
    if free_samples_per_ray < 2:
        free_samples_per_ray = 2

    t = torch.linspace(0.0, 1.0, free_samples_per_ray + 1, device=points_local.device, dtype=points_local.dtype)[1:-1]
    rays = points_local - camera_pos_local.unsqueeze(1)
    free_points = camera_pos_local.unsqueeze(1).unsqueeze(2) + rays.unsqueeze(2) * t.view(1, 1, -1, 1)
    free_points = free_points.masked_fill(~free_mask.unsqueeze(-1).unsqueeze(-1), float("nan"))
    free = pointcloud_to_occupancy_bev(
        free_points.reshape(points_local.shape[0], -1, 3),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
    )
    free = torch.where(occupied > 0, torch.zeros_like(free), free)
    grid = torch.stack((free, occupied), dim=1)

    if inflation_radius > 0:
        grid[:, 1] = inflate_occupancy_grid(grid[:, 1], radius=inflation_radius)
        grid[:, 0] = torch.where(grid[:, 1] > 0, torch.zeros_like(grid[:, 0]), grid[:, 0])

    return grid


def build_room_visibility_grid_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    camera_pos_w: torch.Tensor,
    camera_quat_ros: torch.Tensor,
    env_origins: torch.Tensor,
    camera_max_distance: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    stride: int = 1,
    near_clip: float = 0.05,
    hit_depth_margin: float = 1.0e-3,
    free_samples_per_ray: int | None = None,
) -> torch.Tensor:
    """Build a two-channel env-local room visibility grid from metric depth data."""
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth_image = depth[:, 0]
    elif depth.dim() == 4 and depth.shape[-1] == 1:
        depth_image = depth[..., 0]
    elif depth.dim() == 3:
        depth_image = depth
    else:
        raise ValueError(f"Expected depth with shape (N, 1, H, W), (N, H, W, 1), or (N, H, W). Got: {depth.shape}.")

    depth_sub, _ = subsample_depth_and_intrinsics(depth_image, intrinsics, stride=stride)
    if depth_sub.dim() == 4 and depth_sub.shape[-1] == 1:
        depth_sub = depth_sub[..., 0]
    ray_depth = depth_sub.transpose(1, 2).reshape(depth_sub.shape[0], -1)

    free_mask = torch.isfinite(ray_depth) & (ray_depth > near_clip)
    occupied_mask = free_mask & (ray_depth < camera_max_distance - hit_depth_margin)

    points_w = depth_to_point_cloud(
        depth_image,
        intrinsics,
        stride=stride,
        is_ortho=True,
        position=camera_pos_w,
        orientation=camera_quat_ros,
    )
    points_room = points_w - env_origins.unsqueeze(1)
    camera_pos_room = camera_pos_w - env_origins

    occupied = pointcloud_to_occupancy_bev(
        points_room.masked_fill(~occupied_mask.unsqueeze(-1), float("nan")),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
    )

    if free_samples_per_ray is None:
        free_samples_per_ray = max(2, int(math.ceil(camera_max_distance / cell_size)))
    if free_samples_per_ray < 2:
        free_samples_per_ray = 2

    t = torch.linspace(0.0, 1.0, free_samples_per_ray + 1, device=points_room.device, dtype=points_room.dtype)[1:-1]
    rays = points_room - camera_pos_room.unsqueeze(1)
    free_points = camera_pos_room.unsqueeze(1).unsqueeze(2) + rays.unsqueeze(2) * t.view(1, 1, -1, 1)
    free_points = free_points.masked_fill(~free_mask.unsqueeze(-1).unsqueeze(-1), float("nan"))
    free = pointcloud_to_occupancy_bev(
        free_points.reshape(points_room.shape[0], -1, 3),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
    )
    free = torch.where(occupied > 0, torch.zeros_like(free), free)

    return torch.stack((free, occupied), dim=1)


__all__ = [
    "build_critic_grid_from_depth",
    "build_room_object_state",
    "build_room_visibility_grid_from_depth",
    "sample_pillar_positions",
    "sample_spawn_positions",
    "sample_yaws",
]
