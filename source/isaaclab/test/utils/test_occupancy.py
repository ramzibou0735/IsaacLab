# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for occupancy-grid utility helpers."""

import pytest
import torch

import isaaclab.utils.occupancy as occupancy_utils


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_subsample_depth_and_intrinsics(device):
    """Test regular depth subsampling and matching intrinsic scaling."""
    depth = torch.arange(16, device=device, dtype=torch.float32).reshape(1, 4, 4)
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 6.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32
    )

    depth_sub, intrinsics_sub = occupancy_utils.subsample_depth_and_intrinsics(depth, intrinsics, stride=2)

    expected_depth = torch.tensor([[[0.0, 2.0], [8.0, 10.0]]], device=device)
    expected_intrinsics = torch.tensor(
        [[[2.0, 0.0, 1.0], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]]], device=device
    )

    torch.testing.assert_close(depth_sub, expected_depth)
    torch.testing.assert_close(intrinsics_sub, expected_intrinsics)


@pytest.mark.parametrize("device", DEVICES)
def test_depth_to_point_cloud_with_stride(device):
    """Test point-cloud construction from a subsampled orthogonal depth image."""
    depth = torch.ones((1, 4, 4), device=device, dtype=torch.float32)
    intrinsics = torch.tensor(
        [[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32
    )

    points = occupancy_utils.depth_to_point_cloud(depth, intrinsics, stride=2, is_ortho=True)

    expected_points = torch.tensor(
        [[[-0.5, -0.5, 1.0], [-0.5, 0.0, 1.0], [0.0, -0.5, 1.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(points, expected_points)


@pytest.mark.parametrize("device", DEVICES)
def test_pointcloud_to_occupancy_bev_filters_invalid_points(device):
    """Test that out-of-bounds and invalid points are ignored during voxelization."""
    points = torch.tensor(
        [[[0.2, 0.2, 1.0], [float("nan"), 0.2, 1.0], [0.8, 0.8, 3.0], [3.0, 0.2, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    occupancy = occupancy_utils.pointcloud_to_occupancy_bev(
        points,
        x_limits=(0.0, 2.0),
        y_limits=(0.0, 2.0),
        z_limits=(0.5, 1.5),
        cell_size=1.0,
    )

    expected = torch.zeros((1, 2, 2), device=device, dtype=torch.float32)
    expected[0, 0, 0] = 1.0
    torch.testing.assert_close(occupancy, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_inflate_occupancy_grid(device):
    """Test max-pooling based inflation."""
    occupancy = torch.zeros((5, 5), device=device, dtype=torch.float32)
    occupancy[2, 2] = 1.0

    inflated = occupancy_utils.inflate_occupancy_grid(occupancy, radius=1)

    expected = torch.zeros((5, 5), device=device, dtype=torch.float32)
    expected[1:4, 1:4] = 1.0
    torch.testing.assert_close(inflated, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_depth_to_occupancy_bev_end_to_end(device):
    """Test end-to-end depth-to-BEV occupancy with a simple translation."""
    depth = torch.tensor([[[2.0]]], device=device, dtype=torch.float32)
    intrinsics = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
    position = torch.tensor([[1.0, 2.0, 0.0]], device=device, dtype=torch.float32)

    occupancy = occupancy_utils.depth_to_occupancy_bev(
        depth,
        intrinsics,
        x_limits=(0.0, 2.0),
        y_limits=(1.0, 3.0),
        z_limits=(1.5, 2.5),
        cell_size=1.0,
        position=position,
        inflation_radius=0,
    )

    expected = torch.zeros((1, 2, 2), device=device, dtype=torch.float32)
    expected[0, 1, 1] = 1.0
    torch.testing.assert_close(occupancy, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_pointcloud_to_free_space_occupancy_bev(device):
    """Test free-space and occupied-space channel generation from ray endpoints."""
    points = torch.tensor([[[2.0, 0.0, 0.0]]], device=device, dtype=torch.float32)

    grid = occupancy_utils.pointcloud_to_free_space_occupancy_bev(
        points,
        x_limits=(0.0, 3.0),
        y_limits=(-1.0, 1.0),
        z_limits=(-1.0, 1.0),
        cell_size=1.0,
        free_samples_per_ray=3,
    )

    expected = torch.zeros((1, 2, 3, 2), device=device, dtype=torch.float32)
    expected[0, 0, 1, 1] = 1.0
    expected[0, 1, 2, 1] = 1.0
    torch.testing.assert_close(grid, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_pointcloud_to_free_space_occupancy_bev_respects_occupied_mask(device):
    """Test that masked endpoints contribute free-space without creating false occupied cells."""
    points = torch.tensor([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], device=device, dtype=torch.float32)
    free_mask = torch.tensor([[True, True]], device=device, dtype=torch.bool)
    occupied_mask = torch.tensor([[True, False]], device=device, dtype=torch.bool)

    grid = occupancy_utils.pointcloud_to_free_space_occupancy_bev(
        points,
        x_limits=(0.0, 4.0),
        y_limits=(-1.0, 1.0),
        z_limits=(-1.0, 1.0),
        cell_size=1.0,
        free_samples_per_ray=4,
        free_mask=free_mask,
        occupied_mask=occupied_mask,
    )

    expected = torch.zeros((1, 2, 4, 2), device=device, dtype=torch.float32)
    expected[0, 0, 1, 1] = 1.0
    expected[0, 0, 2, 1] = 1.0
    expected[0, 1, 2, 1] = 1.0
    torch.testing.assert_close(grid, expected)
