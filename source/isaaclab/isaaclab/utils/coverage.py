# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for tracking 2D coverage in batched environments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CoverageGrid2DCfg:
    """Configuration for a 2D env-local coverage grid."""

    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    cell_size: float
    mark_path: bool = True
    count_spawn_as_visited: bool = True
    max_segment_cells: int = 16
    """Fixed upper bound on cells sampled per path segment in a single update.

    Using a fixed bound (instead of the data-dependent maximum) keeps
    :meth:`CoverageGrid2D.update` free of GPU->CPU synchronizations. It must be
    larger than the most cells any single-step motion can cross
    (``max_speed * step_dt / cell_size``)."""


class CoverageGrid2D:
    """Tracks visited cells on a 2D grid for multiple environments in parallel.

    The grid is defined in env-local coordinates over the x-y plane. Cells are marked as visited
    when the agent enters them. When ``mark_path`` is enabled, intermediate cells between consecutive
    positions are also marked by linearly sampling the traveled segment.
    """

    def __init__(self, cfg: CoverageGrid2DCfg, num_envs: int, device: str | torch.device):
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0. Received: {num_envs}.")
        if cfg.cell_size <= 0.0:
            raise ValueError(f"Expected cell_size > 0. Received: {cfg.cell_size}.")
        if cfg.x_limits[0] >= cfg.x_limits[1]:
            raise ValueError(f"Expected increasing x_limits. Received: {cfg.x_limits}.")
        if cfg.y_limits[0] >= cfg.y_limits[1]:
            raise ValueError(f"Expected increasing y_limits. Received: {cfg.y_limits}.")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        self.x_min, self.x_max = cfg.x_limits
        self.y_min, self.y_max = cfg.y_limits
        self.cell_size = float(cfg.cell_size)
        self.grid_h = math.ceil((self.x_max - self.x_min) / self.cell_size)
        self.grid_w = math.ceil((self.y_max - self.y_min) / self.cell_size)
        self.total_cells = self.grid_h * self.grid_w
        self.max_segment_cells = max(1, int(cfg.max_segment_cells))

        self.visited = torch.zeros((self.num_envs, self.grid_h, self.grid_w), device=self.device, dtype=torch.bool)
        self.spawn_cell = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.current_cell = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.previous_cell = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.spawn_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.current_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.previous_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.out_of_bounds = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.coverage_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

    def xy_to_grid(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert env-local x-y positions to clamped grid indices and bounds flags."""
        xy_batch, squeeze = self._as_xy_batch(xy)
        ix = torch.floor((xy_batch[:, 0] - self.x_min) / self.cell_size).long()
        iy = torch.floor((xy_batch[:, 1] - self.y_min) / self.cell_size).long()
        in_bounds = (
            (xy_batch[:, 0] >= self.x_min)
            & (xy_batch[:, 0] < self.x_max)
            & (xy_batch[:, 1] >= self.y_min)
            & (xy_batch[:, 1] < self.y_max)
        )
        cells = torch.stack((ix.clamp_(0, self.grid_h - 1), iy.clamp_(0, self.grid_w - 1)), dim=-1)
        if squeeze:
            return cells.squeeze(0), in_bounds.squeeze(0)
        return cells, in_bounds

    def reset(self, env_ids: torch.Tensor | None, spawn_xy: torch.Tensor) -> None:
        """Reset selected environments and initialize their spawn cells."""
        env_ids = self._resolve_env_ids(env_ids)
        spawn_xy_batch, _ = self._as_xy_batch(spawn_xy)
        if spawn_xy_batch.shape[0] != env_ids.shape[0]:
            raise ValueError(
                f"Expected spawn_xy batch size {env_ids.shape[0]} to match env_ids. Got: {spawn_xy_batch.shape[0]}."
            )

        spawn_cell, spawn_in_bounds = self.xy_to_grid(spawn_xy_batch)
        if not torch.all(spawn_in_bounds):
            bad_env_ids = env_ids[~spawn_in_bounds].tolist()
            raise ValueError(f"Spawn positions are outside the coverage grid for env ids: {bad_env_ids}.")

        self.visited[env_ids] = False
        self.out_of_bounds[env_ids] = False
        self.coverage_count[env_ids] = 0

        self.spawn_xy[env_ids] = spawn_xy_batch
        self.current_xy[env_ids] = spawn_xy_batch
        self.previous_xy[env_ids] = spawn_xy_batch

        self.spawn_cell[env_ids] = spawn_cell
        self.current_cell[env_ids] = spawn_cell
        self.previous_cell[env_ids] = spawn_cell

        if self.cfg.count_spawn_as_visited:
            self.visited[env_ids, spawn_cell[:, 0], spawn_cell[:, 1]] = True
            self.coverage_count[env_ids] = 1

    def update(self, env_ids: torch.Tensor | None, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update selected environments with new x-y positions.

        Returns:
            A tuple ``(new_visit_count, in_bounds)`` where both tensors have shape ``(len(env_ids),)``.
        """
        env_ids = self._resolve_env_ids(env_ids)
        xy_batch, _ = self._as_xy_batch(xy)
        if xy_batch.shape[0] != env_ids.shape[0]:
            raise ValueError(f"Expected xy batch size {env_ids.shape[0]}. Got: {xy_batch.shape[0]}.")

        previous_xy = self.current_xy[env_ids].clone()
        previous_cell = self.current_cell[env_ids].clone()

        current_cell, current_in_bounds = self.xy_to_grid(xy_batch)
        sample_cells, sample_valid = self._cells_along_segments(previous_xy, xy_batch)

        new_visit_count = self._mark_visited(env_ids, sample_cells, sample_valid)

        self.previous_xy[env_ids] = previous_xy
        self.current_xy[env_ids] = xy_batch
        self.previous_cell[env_ids] = previous_cell
        self.current_cell[env_ids] = current_cell
        self.out_of_bounds[env_ids] = ~current_in_bounds

        return new_visit_count, current_in_bounds

    def coverage_ratio(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the per-environment visited-cell ratio."""
        env_ids = self._resolve_env_ids(env_ids)
        return self.coverage_count[env_ids].to(dtype=torch.float32) / float(self.total_cells)

    def _cells_along_segments(self, start_xy: torch.Tensor, end_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample cells along line segments between batched start and end points."""
        if not self.cfg.mark_path:
            cells, in_bounds = self.xy_to_grid(end_xy)
            return cells.unsqueeze(1), in_bounds.unsqueeze(1)

        delta = end_xy - start_xy
        steps = torch.ceil(delta.abs().amax(dim=-1) / self.cell_size).long() + 1
        # Fixed sample count (no host sync); clamp so a segment never needs more.
        steps = steps.clamp(1, self.max_segment_cells)
        max_steps = self.max_segment_cells

        step_idx = torch.arange(max_steps, device=self.device, dtype=torch.long).unsqueeze(0)
        step_idx = step_idx.expand(start_xy.shape[0], -1)
        denom = (steps - 1).clamp_min(1).unsqueeze(1)
        t = torch.minimum(step_idx, steps.unsqueeze(1) - 1).to(dtype=torch.float32) / denom.to(dtype=torch.float32)

        samples = start_xy.unsqueeze(1) + t.unsqueeze(-1) * delta.unsqueeze(1)
        sample_cells, sample_in_bounds = self.xy_to_grid(samples.reshape(-1, 2))
        sample_cells = sample_cells.reshape(start_xy.shape[0], max_steps, 2)
        sample_in_bounds = sample_in_bounds.reshape(start_xy.shape[0], max_steps)
        sample_valid = step_idx < steps.unsqueeze(1)
        sample_valid &= sample_in_bounds
        return sample_cells, sample_valid

    def _mark_visited(self, env_ids: torch.Tensor, cells: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mark valid cells as visited and return the number of newly visited cells per env.

        Fully vectorized (no Python loop / host sync): invalid samples are routed
        to a throwaway sentinel column while scattering, so each batch row builds a
        deduplicated ``to_visit`` mask in one pass.
        """
        batch = env_ids.shape[0]
        # Flatten (row, col) cell indices; send invalid samples to a sentinel column.
        flat_idx = cells[..., 0] * self.grid_w + cells[..., 1]
        flat_idx = torch.where(valid, flat_idx, flat_idx.new_full((), self.total_cells))

        to_visit = torch.zeros((batch, self.total_cells + 1), device=self.device, dtype=torch.bool)
        to_visit.scatter_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.bool))
        to_visit = to_visit[:, : self.total_cells]

        visited_flat = self.visited.view(self.num_envs, -1)[env_ids]
        new_cells = to_visit & ~visited_flat
        new_visit_count = new_cells.sum(dim=-1)

        self.visited[env_ids] = (visited_flat | to_visit).view(batch, self.grid_h, self.grid_w)
        self.coverage_count[env_ids] += new_visit_count
        return new_visit_count

    def _resolve_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    @staticmethod
    def _as_xy_batch(xy: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if xy.dim() == 1:
            if xy.shape[0] != 2:
                raise ValueError(f"Expected xy shape (2,). Got: {xy.shape}.")
            return xy.unsqueeze(0), True
        if xy.dim() != 2 or xy.shape[-1] != 2:
            raise ValueError(f"Expected xy shape (N, 2). Got: {xy.shape}.")
        return xy, False


__all__ = ["CoverageGrid2D", "CoverageGrid2DCfg"]
