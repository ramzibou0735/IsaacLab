# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for building small occupancy grids from depth images."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .math import transform_points, unproject_depth


def _is_batched_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> bool:
    """Infer whether a 3D depth tensor should be interpreted as batched."""
    return depth.dim() == 3 and intrinsics.dim() == 3 and depth.shape[0] == intrinsics.shape[0]


def subsample_depth_and_intrinsics(
    depth: torch.Tensor, intrinsics: torch.Tensor, stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample depth images on a regular grid and scale camera intrinsics accordingly.

    The intended use is to reduce the effective number of rays before calling
    :func:`isaaclab.utils.math.unproject_depth`.

    Args:
        depth: Depth image of shape ``(H, W)``, ``(H, W, 1)``, ``(N, H, W)``, or ``(N, H, W, 1)``.
        intrinsics: Camera intrinsics of shape ``(3, 3)`` or ``(N, 3, 3)``.
        stride: Pixel stride used for regular subsampling. Must be >= 1.

    Returns:
        A tuple ``(depth_sub, intrinsics_sub)`` with the same batch/channel structure as the input depth.

    Raises:
        ValueError: If the input shapes are invalid or the stride is smaller than 1.
    """
    if stride < 1:
        raise ValueError(f"Expected stride >= 1. Received: {stride}.")
    if intrinsics.dim() not in (2, 3):
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3). Got: {intrinsics.shape}.")

    if depth.dim() == 2:
        depth_sub = depth[::stride, ::stride]
    elif depth.dim() == 3:
        if depth.shape[-1] == 1 and not _is_batched_depth(depth, intrinsics):
            depth_sub = depth[::stride, ::stride, :]
        else:
            depth_sub = depth[:, ::stride, ::stride]
    elif depth.dim() == 4:
        if depth.shape[-1] != 1:
            raise ValueError(f"Expected singleton channel dimension in depth image. Got: {depth.shape}.")
        depth_sub = depth[:, ::stride, ::stride, :]
    else:
        raise ValueError(
            "Expected depth to have shape (H, W), (H, W, 1), (N, H, W), or (N, H, W, 1). "
            f"Got: {depth.shape}."
        )

    intrinsics_sub = intrinsics.clone()
    scale = float(stride)
    intrinsics_sub[..., 0, 0] /= scale
    intrinsics_sub[..., 1, 1] /= scale
    intrinsics_sub[..., 0, 2] /= scale
    intrinsics_sub[..., 1, 2] /= scale
    return depth_sub, intrinsics_sub


def depth_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    stride: int = 1,
    is_ortho: bool = True,
    position: torch.Tensor | None = None,
    orientation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a depth image to a point cloud, optionally transformed into another frame.

    Args:
        depth: Depth image of shape ``(H, W)``, ``(H, W, 1)``, ``(N, H, W)``, or ``(N, H, W, 1)``.
        intrinsics: Camera intrinsics of shape ``(3, 3)`` or ``(N, 3, 3)``.
        stride: Optional stride used to subsample the depth image before unprojection.
        is_ortho: Passed through to :func:`isaaclab.utils.math.unproject_depth`.
        position: Optional translation applied with :func:`isaaclab.utils.math.transform_points`.
        orientation: Optional quaternion applied with :func:`isaaclab.utils.math.transform_points`.

    Returns:
        A point cloud of shape ``(P, 3)`` or ``(N, P, 3)``.
    """
    depth_sub, intrinsics_sub = subsample_depth_and_intrinsics(depth, intrinsics, stride=stride)
    if _is_batched_depth(depth_sub, intrinsics_sub) and depth_sub.shape[-1] == 1:
        depth_sub = depth_sub.unsqueeze(-1)
    points = unproject_depth(depth_sub, intrinsics_sub, is_ortho=is_ortho)
    if position is not None or orientation is not None:
        points = transform_points(points, pos=position, quat=orientation)
    return points


def pointcloud_to_occupancy_bev(
    points: torch.Tensor,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Voxelize a point cloud into a binary bird's-eye-view occupancy grid.

    The returned grid uses the x-axis as the first spatial dimension and the y-axis as the
    second spatial dimension. Thus, the output shape is ``(num_x_cells, num_y_cells)`` for
    a single point cloud and ``(N, num_x_cells, num_y_cells)`` for a batch.

    Args:
        points: Point cloud of shape ``(P, 3)`` or ``(N, P, 3)``.
        x_limits: Inclusive-exclusive x-axis limits ``(min_x, max_x)``.
        y_limits: Inclusive-exclusive y-axis limits ``(min_y, max_y)``.
        z_limits: Inclusive-exclusive z-axis limits ``(min_z, max_z)``.
        cell_size: Occupancy cell size in meters. Must be > 0.
        dtype: Output tensor dtype.

    Returns:
        Binary occupancy grid of shape ``(num_x_cells, num_y_cells)`` or ``(N, num_x_cells, num_y_cells)``.

    Raises:
        ValueError: If the input shapes or spatial limits are invalid.
    """
    if cell_size <= 0.0:
        raise ValueError(f"Expected cell_size > 0. Received: {cell_size}.")
    if points.dim() not in (2, 3) or points.shape[-1] != 3:
        raise ValueError(f"Expected points to have shape (P, 3) or (N, P, 3). Got: {points.shape}.")

    x_min, x_max = x_limits
    y_min, y_max = y_limits
    z_min, z_max = z_limits
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError("Expected strictly increasing occupancy limits.")

    is_batched = points.dim() == 3
    if not is_batched:
        points_batch = points.unsqueeze(0)
    else:
        points_batch = points

    num_batches = points_batch.shape[0]
    num_x_cells = math.ceil((x_max - x_min) / cell_size)
    num_y_cells = math.ceil((y_max - y_min) / cell_size)

    occupancy = torch.zeros((num_batches, num_x_cells, num_y_cells), device=points_batch.device, dtype=dtype)

    valid = torch.isfinite(points_batch).all(dim=-1)
    valid &= (points_batch[..., 0] >= x_min) & (points_batch[..., 0] < x_max)
    valid &= (points_batch[..., 1] >= y_min) & (points_batch[..., 1] < y_max)
    valid &= (points_batch[..., 2] >= z_min) & (points_batch[..., 2] < z_max)

    if valid.any():
        ix = torch.floor((points_batch[..., 0] - x_min) / cell_size).long()
        iy = torch.floor((points_batch[..., 1] - y_min) / cell_size).long()

        batch_ids = torch.arange(num_batches, device=points_batch.device, dtype=torch.long)[:, None]
        batch_ids = batch_ids.expand_as(ix)
        flat_ids = batch_ids[valid] * (num_x_cells * num_y_cells) + ix[valid] * num_y_cells + iy[valid]
        occupancy.view(-1)[flat_ids] = torch.as_tensor(1.0, device=occupancy.device, dtype=occupancy.dtype)

    if not is_batched:
        occupancy = occupancy.squeeze(0)
    return occupancy


def inflate_occupancy_grid(occupancy: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Inflate a binary occupancy grid with max-pooling.

    Args:
        occupancy: Occupancy grid of shape ``(H, W)`` or ``(N, H, W)``.
        radius: Inflation radius in cells. Must be >= 0.

    Returns:
        Inflated occupancy grid with the same shape as the input.
    """
    if radius < 0:
        raise ValueError(f"Expected radius >= 0. Received: {radius}.")
    if occupancy.dim() not in (2, 3):
        raise ValueError(f"Expected occupancy to have shape (H, W) or (N, H, W). Got: {occupancy.shape}.")
    if radius == 0:
        return occupancy.clone()

    is_batched = occupancy.dim() == 3
    if not is_batched:
        occupancy_batch = occupancy.unsqueeze(0)
    else:
        occupancy_batch = occupancy

    kernel_size = 2 * radius + 1
    inflated = F.max_pool2d(
        occupancy_batch.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=radius
    ).squeeze(1)

    if not is_batched:
        inflated = inflated.squeeze(0)
    return inflated


def pointcloud_to_free_space_occupancy_bev(
    points: torch.Tensor,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    free_samples_per_ray: int | None = None,
    free_mask: torch.Tensor | None = None,
    occupied_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert ray endpoints into a two-channel BEV grid of free and occupied space.

    The input point cloud is interpreted as a set of ray endpoints emitted from the origin of the current frame.
    Free-space is approximated by sampling points uniformly along each ray segment, excluding the endpoint.

    Args:
        points: Point cloud of shape ``(P, 3)`` or ``(N, P, 3)`` containing ray endpoints.
        x_limits: Inclusive-exclusive x-axis limits ``(min_x, max_x)``.
        y_limits: Inclusive-exclusive y-axis limits ``(min_y, max_y)``.
        z_limits: Inclusive-exclusive z-axis limits ``(min_z, max_z)``.
        cell_size: Occupancy cell size in meters. Must be > 0.
        free_samples_per_ray: Number of samples used to approximate free-space along each ray.
        free_mask: Optional boolean mask over ray endpoints that contribute to the free-space channel.
            Shape must be ``(P,)`` or ``(N, P)``.
        occupied_mask: Optional boolean mask over ray endpoints that contribute to the occupied-space channel.
            Shape must be ``(P,)`` or ``(N, P)``.
        dtype: Output tensor dtype.

    Returns:
        A tensor of shape ``(2, H, W)`` or ``(N, 2, H, W)`` where channel 0 is free-space and
        channel 1 is occupied-space.
    """
    if points.dim() not in (2, 3) or points.shape[-1] != 3:
        raise ValueError(f"Expected points to have shape (P, 3) or (N, P, 3). Got: {points.shape}.")

    is_batched = points.dim() == 3
    if not is_batched:
        points_batch = points.unsqueeze(0)
    else:
        points_batch = points

    num_batches, num_points = points_batch.shape[:2]

    def _resolve_mask(mask: torch.Tensor | None, name: str) -> torch.Tensor:
        if mask is None:
            return torch.ones((num_batches, num_points), device=points_batch.device, dtype=torch.bool)
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.dim() == 1:
            if is_batched:
                if mask.shape[0] != num_points:
                    raise ValueError(f"Expected {name} to have shape ({num_points},). Got: {mask.shape}.")
                mask = mask.unsqueeze(0).expand(num_batches, -1)
            else:
                if mask.shape[0] != num_points:
                    raise ValueError(f"Expected {name} to have shape ({num_points},). Got: {mask.shape}.")
                mask = mask.unsqueeze(0)
        elif mask.dim() == 2:
            if mask.shape != (num_batches, num_points):
                raise ValueError(
                    f"Expected {name} to have shape ({num_batches}, {num_points}). Got: {mask.shape}."
                )
        else:
            raise ValueError(f"Expected {name} to have shape (P,) or (N, P). Got: {mask.shape}.")
        return mask.to(device=points_batch.device, dtype=torch.bool)

    free_mask_batch = _resolve_mask(free_mask, "free_mask")
    occupied_mask_batch = _resolve_mask(occupied_mask, "occupied_mask")

    occupied = pointcloud_to_occupancy_bev(
        points_batch.masked_fill(~occupied_mask_batch.unsqueeze(-1), float("nan")),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
        dtype=dtype,
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

    occupied_batch = occupied

    t = torch.linspace(0.0, 1.0, free_samples_per_ray + 1, device=points_batch.device, dtype=points_batch.dtype)[1:-1]
    free_points = points_batch.masked_fill(~free_mask_batch.unsqueeze(-1), float("nan")).unsqueeze(2) * t.view(1, 1, -1, 1)
    free = pointcloud_to_occupancy_bev(
        free_points.reshape(points_batch.shape[0], -1, 3),
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
        dtype=dtype,
    )
    free = torch.where(occupied_batch > 0, torch.zeros_like(free), free)

    grid = torch.stack((free, occupied_batch), dim=1)
    if not is_batched:
        grid = grid.squeeze(0)
    return grid


def depth_to_occupancy_bev(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    cell_size: float,
    stride: int = 1,
    is_ortho: bool = True,
    position: torch.Tensor | None = None,
    orientation: torch.Tensor | None = None,
    inflation_radius: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a depth image directly into a small BEV occupancy grid.

    Args:
        depth: Depth image of shape ``(H, W)``, ``(H, W, 1)``, ``(N, H, W)``, or ``(N, H, W, 1)``.
        intrinsics: Camera intrinsics of shape ``(3, 3)`` or ``(N, 3, 3)``.
        x_limits: Inclusive-exclusive x-axis limits ``(min_x, max_x)``.
        y_limits: Inclusive-exclusive y-axis limits ``(min_y, max_y)``.
        z_limits: Inclusive-exclusive z-axis limits ``(min_z, max_z)``.
        cell_size: Occupancy cell size in meters. Must be > 0.
        stride: Optional stride used to subsample the depth image before unprojection.
        is_ortho: Passed through to :func:`isaaclab.utils.math.unproject_depth`.
        position: Optional translation applied after unprojection.
        orientation: Optional quaternion applied after unprojection.
        inflation_radius: Optional inflation radius in cells.
        dtype: Output tensor dtype.

    Returns:
        Binary occupancy grid of shape ``(H, W)`` or ``(N, H, W)``.
    """
    points = depth_to_point_cloud(
        depth,
        intrinsics,
        stride=stride,
        is_ortho=is_ortho,
        position=position,
        orientation=orientation,
    )
    occupancy = pointcloud_to_occupancy_bev(
        points,
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        cell_size=cell_size,
        dtype=dtype,
    )
    if inflation_radius > 0:
        occupancy = inflate_occupancy_grid(occupancy, radius=inflation_radius)
    return occupancy


__all__ = [
    "depth_to_occupancy_bev",
    "depth_to_point_cloud",
    "inflate_occupancy_grid",
    "pointcloud_to_free_space_occupancy_bev",
    "pointcloud_to_occupancy_bev",
    "subsample_depth_and_intrinsics",
]
