# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply


def build_local_map_bases(
    num_envs: int,
    crop_size: int,
    cell_size: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build persistent index and local-position bases for a centered 3D crop."""
    indices = torch.arange(crop_size, device=device, dtype=torch.long)
    index_base = torch.cartesian_prod(indices, indices, indices).unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    center = torch.floor(torch.full((1, 3), 0.5 * crop_size, device=device, dtype=torch.float32))
    position_base = (index_base.to(dtype=torch.float32) - center) * float(cell_size)
    return index_base, position_base


def extract_centered_map(
    occupancy_map: torch.Tensor,
    crop_size: int,
    position: torch.Tensor,
    orientation: torch.Tensor,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    index_base: torch.Tensor,
    position_base: torch.Tensor,
) -> torch.Tensor:
    """Extract a robot-centered 3D crop using full quaternion orientation.

    This mirrors active-perception's ``extract_centered_tensor`` semantics:
    local crop sample points are rotated by the robot quaternion, rounded into
    the global occupancy map, clamped to map bounds, and scattered back into the
    crop grid at ``index_base``.
    """
    num_envs = occupancy_map.shape[0]
    flat_count = crop_size**3
    if position_base.shape[:2] != (num_envs, flat_count):
        position_base = position_base[:num_envs]
    if index_base.shape[:2] != (num_envs, flat_count):
        index_base = index_base[:num_envs]

    quat = orientation.unsqueeze(1).expand(-1, flat_count, -1).reshape(-1, 4)
    local_points = position_base.reshape(-1, 3)
    position_world = position.unsqueeze(1) + quat_apply(quat, local_points).view(num_envs, flat_count, 3)

    mins = torch.tensor(bounds_min, device=occupancy_map.device, dtype=occupancy_map.dtype)
    maxs = torch.tensor(bounds_max, device=occupancy_map.device, dtype=occupancy_map.dtype)
    dims = torch.tensor(occupancy_map.shape[1:], device=occupancy_map.device, dtype=occupancy_map.dtype)
    voxel = ((position_world - mins) / (maxs - mins) * (dims - 1.0)).round().long()
    voxel[..., 0].clamp_(0, occupancy_map.shape[1] - 1)
    voxel[..., 1].clamp_(0, occupancy_map.shape[2] - 1)
    voxel[..., 2].clamp_(0, occupancy_map.shape[3] - 1)

    env_index = torch.arange(num_envs, device=occupancy_map.device).unsqueeze(1)
    values = occupancy_map[env_index, voxel[..., 0], voxel[..., 1], voxel[..., 2]]

    local_map = torch.zeros(
        (num_envs, crop_size, crop_size, crop_size),
        device=occupancy_map.device,
        dtype=occupancy_map.dtype,
    )
    local_map[env_index, index_base[..., 0], index_base[..., 1], index_base[..., 2]] = values
    return local_map


def project_occupancy_to_bev(
    occupancy_map: torch.Tensor,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    map_bounds_min: tuple[float, float, float],
    map_bounds_max: tuple[float, float, float],
) -> torch.Tensor:
    """Project 3D occupancy values into a two-channel BEV known-space grid."""
    num_envs = occupancy_map.shape[0]
    height = int((x_limits[1] - x_limits[0]) / cell_size)
    width = int((y_limits[1] - y_limits[0]) / cell_size)
    device = occupancy_map.device

    x_centers = x_limits[0] + cell_size * (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
    y_centers = y_limits[0] + cell_size * (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
    x_valid = (x_centers >= map_bounds_min[0]) & (x_centers <= map_bounds_max[0])
    y_valid = (y_centers >= map_bounds_min[1]) & (y_centers <= map_bounds_max[1])
    z_mask = _axis_center_mask(
        occupancy_map.shape[3],
        map_bounds_min[2],
        map_bounds_max[2],
        z_limits[0],
        z_limits[1],
        device,
    )
    ix = _centers_to_indices(x_centers, map_bounds_min[0], map_bounds_max[0], occupancy_map.shape[1])
    iy = _centers_to_indices(y_centers, map_bounds_min[1], map_bounds_max[1], occupancy_map.shape[2])

    projected = occupancy_map[:, ix][:, :, iy]
    projected = projected[..., z_mask]
    occupied = (projected == 2.0).any(dim=-1).to(dtype=occupancy_map.dtype)
    free = ((projected == 1.0).any(dim=-1) & (occupied == 0)).to(dtype=occupancy_map.dtype)
    valid_xy = (x_valid[:, None] & y_valid[None, :]).to(dtype=occupancy_map.dtype)
    occupied = occupied * valid_xy
    free = free * valid_xy
    return torch.stack((free, occupied), dim=1).view(num_envs, 2, height, width)


def _centers_to_indices(centers: torch.Tensor, min_value: float, max_value: float, dim: int) -> torch.Tensor:
    return ((centers - min_value) / (max_value - min_value) * (dim - 1)).round().long().clamp(0, dim - 1)


def _axis_center_mask(
    dim: int,
    min_value: float,
    max_value: float,
    lower: float,
    upper: float,
    device: torch.device | str,
) -> torch.Tensor:
    centers = min_value + (max_value - min_value) * torch.arange(dim, device=device, dtype=torch.float32) / (dim - 1)
    return (centers >= lower) & (centers <= upper)


__all__ = [
    "build_local_map_bases",
    "extract_centered_map",
    "project_occupancy_to_bev",
]
