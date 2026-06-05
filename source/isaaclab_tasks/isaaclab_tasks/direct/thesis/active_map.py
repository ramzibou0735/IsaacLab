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
    """Build persistent index and local-position bases for a centered 3-D crop."""
    indices = torch.arange(crop_size, device=device, dtype=torch.long)
    index_base = torch.cartesian_prod(indices, indices, indices).unsqueeze(0).expand(num_envs, -1, -1).contiguous()
    center = torch.floor(torch.full((1, 3), 0.5 * crop_size, device=device, dtype=torch.float32))
    position_base = (index_base.to(dtype=torch.float32) - center) * float(cell_size)
    return index_base, position_base


def reset_thesis_map_tensors(
    known_free: torch.Tensor,
    known_occupied: torch.Tensor,
    recency: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    """Reset the persistent 3-D map tensors for selected environments."""
    known_free[env_ids] = False
    known_occupied[env_ids] = False
    recency[env_ids] = 0.0


def update_thesis_map_tensors(
    occupancy_map: torch.Tensor,
    known_free: torch.Tensor,
    known_occupied: torch.Tensor,
    recency: torch.Tensor,
    env_ids: torch.Tensor | None = None,
    *,
    recency_decay: float = 0.96,
    observed_recency: float = 0.25,
    new_info_recency: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update persistent map channels from the accumulated Warp occupancy map.

    Occupancy values use the active-perception convention: ``0`` unknown,
    ``1`` known free, and ``2`` known occupied. The free/occupied maps are
    mutually exclusive, and occupied observations overwrite prior free labels.

    Returns:
        Newly observed free-cell, occupied-cell, and total-cell counts per env.
    """
    if env_ids is None:
        env_ids = torch.arange(occupancy_map.shape[0], device=occupancy_map.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=occupancy_map.device, dtype=torch.long)

    observed_free = occupancy_map[env_ids] == 1.0
    observed_occupied = occupancy_map[env_ids] == 2.0
    observed_free &= ~observed_occupied
    previously_known = known_free[env_ids] | known_occupied[env_ids]

    new_occupied = observed_occupied & ~previously_known
    new_free = observed_free & ~previously_known
    new_free &= ~observed_occupied

    known_occupied[env_ids] |= observed_occupied
    known_free[env_ids] |= observed_free
    known_free[env_ids] &= ~known_occupied[env_ids]

    observed = observed_free | observed_occupied
    new_info = new_free | new_occupied
    recency[env_ids] *= float(recency_decay)
    recency[env_ids] = torch.where(
        observed,
        torch.maximum(
            recency[env_ids],
            torch.full_like(recency[env_ids], float(observed_recency)),
        ),
        recency[env_ids],
    )
    recency[env_ids] = torch.where(
        new_info,
            torch.full_like(recency[env_ids], float(new_info_recency)),
        recency[env_ids],
    )
    recency[env_ids].clamp_(0.0, 1.0)

    return (
        new_free.flatten(1).sum(dim=-1),
        new_occupied.flatten(1).sum(dim=-1),
        new_info.flatten(1).sum(dim=-1),
    )


def stamp_robot_recency(
    recency: torch.Tensor,
    positions: torch.Tensor,
    env_ids: torch.Tensor,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    radius: float,
    value: float = 1.0,
) -> None:
    """Stamp a robot-centered spherical path marker into the recency map."""
    if radius <= 0.0 or env_ids.numel() == 0:
        return
    env_ids = env_ids.to(device=recency.device, dtype=torch.long)
    positions = positions.to(device=recency.device, dtype=recency.dtype)
    dims = torch.tensor(recency.shape[1:], device=recency.device, dtype=recency.dtype)
    mins = torch.tensor(bounds_min, device=recency.device, dtype=recency.dtype)
    maxs = torch.tensor(bounds_max, device=recency.device, dtype=recency.dtype)
    voxel_size = (maxs - mins) / (dims - 1.0)
    radius_cells = torch.ceil(torch.as_tensor(radius, device=recency.device) / voxel_size).long()

    x_offsets = torch.arange(-radius_cells[0], radius_cells[0] + 1, device=recency.device, dtype=torch.long)
    y_offsets = torch.arange(-radius_cells[1], radius_cells[1] + 1, device=recency.device, dtype=torch.long)
    z_offsets = torch.arange(-radius_cells[2], radius_cells[2] + 1, device=recency.device, dtype=torch.long)
    offsets = torch.cartesian_prod(x_offsets, y_offsets, z_offsets)
    offset_metric = offsets.to(dtype=recency.dtype) * voxel_size
    offsets = offsets[torch.linalg.vector_norm(offset_metric, dim=-1) <= radius]
    if offsets.numel() == 0:
        return

    center = ((positions - mins) / (maxs - mins) * (dims - 1.0)).round().long()
    indices = center.unsqueeze(1) + offsets.unsqueeze(0)
    indices[..., 0].clamp_(0, recency.shape[1] - 1)
    indices[..., 1].clamp_(0, recency.shape[2] - 1)
    indices[..., 2].clamp_(0, recency.shape[3] - 1)

    batch_ids = env_ids.view(-1, 1).expand(-1, offsets.shape[0])
    recency[batch_ids, indices[..., 0], indices[..., 1], indices[..., 2]] = float(value)


def compose_thesis_map_channels(
    known_free: torch.Tensor,
    known_occupied: torch.Tensor,
    recency: torch.Tensor,
) -> torch.Tensor:
    """Compose ``free/occupied/unknown/recency`` channels for the thesis policy."""
    known_free = known_free & ~known_occupied
    unknown = ~(known_free | known_occupied)
    return torch.stack(
        (
            known_free.to(dtype=recency.dtype),
            known_occupied.to(dtype=recency.dtype),
            unknown.to(dtype=recency.dtype),
            recency.clamp(0.0, 1.0),
        ),
        dim=1,
    )


def project_thesis_map_to_bev(
    known_free: torch.Tensor,
    known_occupied: torch.Tensor,
    recency: torch.Tensor,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    map_bounds_min: tuple[float, float, float],
    map_bounds_max: tuple[float, float, float],
) -> torch.Tensor:
    """Project persistent 3-D thesis maps into a four-channel 2-D BEV grid."""
    num_envs = known_free.shape[0]
    height = int((x_limits[1] - x_limits[0]) / cell_size)
    width = int((y_limits[1] - y_limits[0]) / cell_size)
    device = known_free.device

    x_centers = x_limits[0] + cell_size * (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
    y_centers = y_limits[0] + cell_size * (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
    x_valid = (x_centers >= map_bounds_min[0]) & (x_centers <= map_bounds_max[0])
    y_valid = (y_centers >= map_bounds_min[1]) & (y_centers <= map_bounds_max[1])
    z_mask = _axis_center_mask(
        known_free.shape[3],
        map_bounds_min[2],
        map_bounds_max[2],
        z_limits[0],
        z_limits[1],
        device,
    )
    ix = _centers_to_indices(x_centers, map_bounds_min[0], map_bounds_max[0], known_free.shape[1])
    iy = _centers_to_indices(y_centers, map_bounds_min[1], map_bounds_max[1], known_free.shape[2])

    free_3d = known_free[:, ix][:, :, iy][..., z_mask]
    occupied_3d = known_occupied[:, ix][:, :, iy][..., z_mask]
    recency_3d = recency[:, ix][:, :, iy][..., z_mask]
    occupied = occupied_3d.any(dim=-1)
    free = free_3d.any(dim=-1) & ~occupied
    unknown = ~(free | occupied)
    recency_bev = recency_3d.amax(dim=-1)
    valid_xy = x_valid[:, None] & y_valid[None, :]

    channels = torch.stack(
        (
            free & valid_xy,
            occupied & valid_xy,
            unknown & valid_xy,
            recency_bev * valid_xy.to(dtype=recency.dtype),
        ),
        dim=1,
    )
    channels[:, :3] = channels[:, :3].to(dtype=recency.dtype)
    return channels.view(num_envs, 4, height, width)


def extract_centered_map_channels(
    map_channels: torch.Tensor,
    crop_size: int,
    position: torch.Tensor,
    orientation: torch.Tensor,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    index_base: torch.Tensor,
    position_base: torch.Tensor,
) -> torch.Tensor:
    """Extract an orientation-aware robot-centered crop from channelized 3-D maps."""
    if map_channels.dim() != 5:
        raise ValueError(f"Expected map_channels shape (N, C, X, Y, Z), got {tuple(map_channels.shape)}")
    num_envs = map_channels.shape[0]
    num_channels = map_channels.shape[1]
    flat_count = crop_size**3
    if position_base.shape[:2] != (num_envs, flat_count):
        position_base = position_base[:num_envs]
    if index_base.shape[:2] != (num_envs, flat_count):
        index_base = index_base[:num_envs]

    quat = orientation.unsqueeze(1).expand(-1, flat_count, -1).reshape(-1, 4)
    local_points = position_base.reshape(-1, 3)
    position_world = position.unsqueeze(1) + quat_apply(quat, local_points).view(num_envs, flat_count, 3)

    mins = torch.tensor(bounds_min, device=map_channels.device, dtype=map_channels.dtype)
    maxs = torch.tensor(bounds_max, device=map_channels.device, dtype=map_channels.dtype)
    dims = torch.tensor(map_channels.shape[2:], device=map_channels.device, dtype=map_channels.dtype)
    voxel = ((position_world - mins) / (maxs - mins) * (dims - 1.0)).round().long()
    voxel[..., 0].clamp_(0, map_channels.shape[2] - 1)
    voxel[..., 1].clamp_(0, map_channels.shape[3] - 1)
    voxel[..., 2].clamp_(0, map_channels.shape[4] - 1)

    spatial_shape = map_channels.shape[2:]
    flat_voxel = voxel[..., 0] * spatial_shape[1] * spatial_shape[2] + voxel[..., 1] * spatial_shape[2] + voxel[..., 2]
    map_flat = map_channels.reshape(num_envs, num_channels, -1)
    values = torch.gather(map_flat, 2, flat_voxel.unsqueeze(1).expand(-1, num_channels, -1))

    local_map = torch.zeros(
        (num_envs, num_channels, crop_size, crop_size, crop_size),
        device=map_channels.device,
        dtype=map_channels.dtype,
    )
    env_index = torch.arange(num_envs, device=map_channels.device).view(num_envs, 1, 1).expand(-1, num_channels, flat_count)
    channel_index = torch.arange(num_channels, device=map_channels.device).view(1, num_channels, 1).expand(num_envs, -1, flat_count)
    crop_x = index_base[..., 0].unsqueeze(1).expand(-1, num_channels, -1)
    crop_y = index_base[..., 1].unsqueeze(1).expand(-1, num_channels, -1)
    crop_z = index_base[..., 2].unsqueeze(1).expand(-1, num_channels, -1)
    local_map[env_index, channel_index, crop_x, crop_y, crop_z] = values
    return local_map


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
    "compose_thesis_map_channels",
    "extract_centered_map_channels",
    "project_thesis_map_to_bev",
    "reset_thesis_map_tensors",
    "stamp_robot_recency",
    "update_thesis_map_tensors",
]
