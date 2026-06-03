# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch


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
        # Fixed number of rejection iterations (no host sync / early break). The
        # first accepted candidate per env is kept; still-unassigned envs fall back
        # to the most recent candidate so every env always has a position.
        chosen = _sample_uniform_xy(num_envs, device, x_limits=inner_x_limits, y_limits=inner_y_limits)

        for _ in range(max_tries):
            candidate = _sample_uniform_xy(num_envs, device, x_limits=inner_x_limits, y_limits=inner_y_limits)

            valid = torch.linalg.vector_norm(candidate - spawn_xy, dim=-1) >= spawn_clearance
            if pillar_id > 0:
                previous = centers[:, :pillar_id]
                separation = torch.linalg.vector_norm(candidate.unsqueeze(1) - previous, dim=-1).amin(dim=-1)
                valid &= separation >= pillar_spacing

            accept = (~assigned) & valid
            chosen = torch.where(accept.unsqueeze(-1), candidate, chosen)
            assigned |= accept
            # Keep unassigned envs tracking the latest candidate as a fallback.
            chosen = torch.where((~assigned).unsqueeze(-1), candidate, chosen)

        centers[:, pillar_id] = chosen

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


__all__ = [
    "build_room_object_state",
    "sample_pillar_positions",
    "sample_spawn_positions",
    "sample_yaws",
]
