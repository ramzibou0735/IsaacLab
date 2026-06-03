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


def _reference_mark_visited(visited, coverage_count, grid_w, env_ids, cells, valid):
    """Original per-env loop implementation, used as a parity reference."""
    flat = visited.view(visited.shape[0], -1)
    new_visit_count = torch.zeros((env_ids.shape[0],), device=visited.device, dtype=torch.long)
    for batch_id, env_id in enumerate(env_ids.tolist()):
        valid_mask = valid[batch_id]
        if not torch.any(valid_mask):
            continue
        cell_batch = cells[batch_id, valid_mask]
        flat_idx = cell_batch[:, 0] * grid_w + cell_batch[:, 1]
        flat_idx = torch.unique(flat_idx)
        was_unvisited = ~flat[env_id, flat_idx]
        count = was_unvisited.sum()
        if count > 0:
            flat[env_id, flat_idx] = True
            coverage_count[env_id] += count
            new_visit_count[batch_id] = count
    return new_visit_count


@pytest.mark.parametrize("device", DEVICES)
def test_mark_visited_matches_reference_loop(device):
    """The vectorized _mark_visited must match the original per-env loop output."""
    torch.manual_seed(3)
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0), cell_size=1.0),
        num_envs=5,
        device=device,
    )
    env_ids = torch.arange(5, device=device, dtype=torch.long)
    steps = 7
    cells = torch.stack(
        (
            torch.randint(0, grid.grid_h, (5, steps), device=device),
            torch.randint(0, grid.grid_w, (5, steps), device=device),
        ),
        dim=-1,
    )
    valid = torch.rand((5, steps), device=device) > 0.3

    ref_visited = grid.visited.clone()
    ref_count = grid.coverage_count.clone()
    ref_new = _reference_mark_visited(ref_visited, ref_count, grid.grid_w, env_ids, cells, valid)

    new_new = grid._mark_visited(env_ids, cells, valid)

    torch.testing.assert_close(new_new, ref_new)
    torch.testing.assert_close(grid.visited, ref_visited)
    torch.testing.assert_close(grid.coverage_count, ref_count)


@pytest.mark.parametrize("device", DEVICES)
def test_cumulative_new_visits_equals_coverage_count(device):
    """Over a random walk, summed new visits (+ spawn) equal the coverage count."""
    torch.manual_seed(11)
    grid = CoverageGrid2D(
        CoverageGrid2DCfg(x_limits=(-5.0, 5.0), y_limits=(-5.0, 5.0), cell_size=0.5),
        num_envs=8,
        device=device,
    )
    pos = torch.zeros((8, 2), device=device)
    grid.reset(None, pos)
    total_new = torch.ones((8,), device=device, dtype=torch.long)  # spawn counts as 1
    for _ in range(40):
        pos = (pos + 0.4 * (2.0 * torch.rand((8, 2), device=device) - 1.0)).clamp(-4.9, 4.9)
        new_visit_count, _ = grid.update(None, pos)
        total_new += new_visit_count
    torch.testing.assert_close(total_new, grid.coverage_count)
    torch.testing.assert_close(grid.coverage_count, grid.visited.flatten(1).sum(dim=-1))


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
