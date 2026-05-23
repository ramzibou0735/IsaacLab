# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the 2D coverage-grid helper."""

import pytest
import torch

from isaaclab.utils.coverage import CoverageGrid2D, CoverageGrid2DCfg


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_xy_to_grid_and_bounds(device):
    """Test local x-y to grid conversion and bounds masking."""
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(-1.0, 3.0), cell_size=1.0),
        num_envs=4,
        device=device,
    )
    xy = torch.tensor(
        [[0.1, -0.5], [3.9, 2.9], [-0.1, 0.0], [1.0, 3.0]],
        device=device,
        dtype=torch.float32,
    )

    cells, in_bounds = grid.xy_to_grid(xy)

    expected_cells = torch.tensor([[0, 0], [3, 3], [0, 1], [1, 3]], device=device, dtype=torch.long)
    expected_in_bounds = torch.tensor([True, True, False, False], device=device, dtype=torch.bool)
    torch.testing.assert_close(cells, expected_cells)
    torch.testing.assert_close(in_bounds, expected_in_bounds)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_marks_spawn(device):
    """Test spawn initialization and coverage count at reset."""
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0), cell_size=1.0),
        num_envs=2,
        device=device,
    )
    spawn_xy = torch.tensor([[0.1, 0.1], [2.4, 1.7]], device=device, dtype=torch.float32)

    grid.reset(None, spawn_xy)

    expected_spawn_cells = torch.tensor([[0, 0], [2, 1]], device=device, dtype=torch.long)
    torch.testing.assert_close(grid.spawn_cell, expected_spawn_cells)
    torch.testing.assert_close(grid.current_cell, expected_spawn_cells)
    torch.testing.assert_close(grid.previous_cell, expected_spawn_cells)
    torch.testing.assert_close(grid.coverage_count, torch.tensor([1, 1], device=device, dtype=torch.long))
    assert bool(grid.visited[0, 0, 0].item())
    assert bool(grid.visited[1, 2, 1].item())


@pytest.mark.parametrize("device", DEVICES)
def test_update_marks_path_cells(device):
    """Test that intermediate path cells are marked and revisits do not pay twice."""
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(0.0, 2.0), cell_size=1.0, mark_path=True),
        num_envs=1,
        device=device,
    )
    grid.reset(None, torch.tensor([[0.1, 0.1]], device=device))

    new_visit_count, in_bounds = grid.update(None, torch.tensor([[2.9, 0.1]], device=device))
    torch.testing.assert_close(new_visit_count, torch.tensor([2], device=device, dtype=torch.long))
    torch.testing.assert_close(in_bounds, torch.tensor([True], device=device))
    assert bool(grid.visited[0, 0, 0].item())
    assert bool(grid.visited[0, 1, 0].item())
    assert bool(grid.visited[0, 2, 0].item())
    torch.testing.assert_close(grid.coverage_count, torch.tensor([3], device=device, dtype=torch.long))

    revisit_count, _ = grid.update(None, torch.tensor([[2.2, 0.1]], device=device))
    torch.testing.assert_close(revisit_count, torch.tensor([0], device=device, dtype=torch.long))
    torch.testing.assert_close(grid.coverage_count, torch.tensor([3], device=device, dtype=torch.long))


@pytest.mark.parametrize("device", DEVICES)
def test_update_out_of_bounds_marks_only_valid_cells(device):
    """Test that out-of-bounds motion sets the flag but still preserves valid in-bounds path coverage."""
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(0.0, 2.0), cell_size=1.0, mark_path=True),
        num_envs=1,
        device=device,
    )
    grid.reset(None, torch.tensor([[0.1, 0.1]], device=device))

    new_visit_count, in_bounds = grid.update(None, torch.tensor([[4.5, 0.1]], device=device))

    torch.testing.assert_close(new_visit_count, torch.tensor([3], device=device, dtype=torch.long))
    torch.testing.assert_close(in_bounds, torch.tensor([False], device=device))
    torch.testing.assert_close(grid.out_of_bounds, torch.tensor([True], device=device))
    torch.testing.assert_close(grid.coverage_count, torch.tensor([4], device=device, dtype=torch.long))
    for x_idx in range(4):
        assert bool(grid.visited[0, x_idx, 0].item())


@pytest.mark.parametrize("device", DEVICES)
def test_partial_reset_preserves_other_envs(device):
    """Test that partial resets only affect selected environments."""
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0), cell_size=1.0),
        num_envs=2,
        device=device,
    )
    spawn_xy = torch.tensor([[0.1, 0.1], [1.1, 0.1]], device=device, dtype=torch.float32)
    grid.reset(None, spawn_xy)
    grid.update(torch.tensor([0], device=device), torch.tensor([[2.1, 0.1]], device=device))

    grid.reset(torch.tensor([0], device=device), torch.tensor([[0.1, 1.1]], device=device))

    torch.testing.assert_close(grid.coverage_count, torch.tensor([1, 1], device=device, dtype=torch.long))
    assert bool(grid.visited[0, 0, 1].item())
    assert not bool(grid.visited[0, 2, 0].item())
    assert bool(grid.visited[1, 1, 0].item())
