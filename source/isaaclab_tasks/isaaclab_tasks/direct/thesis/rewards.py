# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


FLY_HEIGHT = 1.2


def compute_thesis_exploration_reward(
    *,
    new_free_cells: torch.Tensor,
    new_occupied_cells: torch.Tensor,
    new_visit_count: torch.Tensor,
    coverage_ratio: torch.Tensor,
    frontier_count: torch.Tensor,
    revisit_ratio: torch.Tensor,
    min_depth: torch.Tensor,
    root_height: torch.Tensor,
    up_z: torch.Tensor,
    ang_vel_norm: torch.Tensor,
    action_diff_norm: torch.Tensor,
    collision: torch.Tensor,
    out_of_bounds: torch.Tensor,
    bad_height: torch.Tensor,
    revisit_penalty_scale: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute thesis exploration reward terms from batched scalar features."""
    free_info_reward = 0.035 * torch.sqrt(new_free_cells)
    occupied_info_reward = 0.055 * torch.sqrt(new_occupied_cells)
    frontier_reward = 0.8 * frontier_count
    coverage_reward = 0.25 * new_visit_count
    coverage_milestone_reward = 4.0 * torch.clamp(coverage_ratio - 0.25, min=0.0)
    sustained_gain_reward = 0.004 * torch.clamp(new_free_cells + 2.0 * new_occupied_cells, max=512.0)
    alive_reward = torch.full_like(new_free_cells, 0.03)

    upright_reward = 0.25 * torch.square(torch.clamp(up_z, min=0.0))
    height_reward = 0.35 * torch.exp(-6.0 * torch.square(root_height - FLY_HEIGHT))
    spin_penalty = -0.04 * torch.square(torch.clamp(ang_vel_norm - 1.5, min=0.0))
    action_jerk_penalty = -0.025 * torch.square(action_diff_norm)
    risk_penalty = -3.0 * torch.clamp(0.55 - min_depth, min=0.0)
    low_height_soft_penalty = -1.5 * torch.clamp(0.65 - root_height, min=0.0)
    high_height_soft_penalty = -1.0 * torch.clamp(root_height - 1.75, min=0.0)
    revisit_penalty = -float(revisit_penalty_scale) * revisit_ratio

    collision_penalty = torch.where(collision, torch.full_like(new_free_cells, -450.0), 0.0)
    bounds_penalty = torch.where(out_of_bounds, torch.full_like(new_free_cells, -350.0), 0.0)
    height_penalty = torch.where(bad_height, torch.full_like(new_free_cells, -150.0), 0.0)

    reward = (
        free_info_reward
        + occupied_info_reward
        + frontier_reward
        + coverage_reward
        + coverage_milestone_reward
        + sustained_gain_reward
        + alive_reward
        + upright_reward
        + height_reward
        + spin_penalty
        + action_jerk_penalty
        + risk_penalty
        + low_height_soft_penalty
        + high_height_soft_penalty
        + revisit_penalty
        + collision_penalty
        + bounds_penalty
        + height_penalty
    )
    reward_info = {
        "free_info_reward": free_info_reward,
        "occupied_info_reward": occupied_info_reward,
        "frontier_reward": frontier_reward,
        "coverage_reward": coverage_reward,
        "coverage_milestone_reward": coverage_milestone_reward,
        "sustained_gain_reward": sustained_gain_reward,
        "alive_reward": alive_reward,
        "upright_reward": upright_reward,
        "height_reward": height_reward,
        "spin_penalty": spin_penalty,
        "action_jerk_penalty": action_jerk_penalty,
        "risk_penalty": risk_penalty,
        "low_height_soft_penalty": low_height_soft_penalty,
        "high_height_soft_penalty": high_height_soft_penalty,
        "revisit_penalty": revisit_penalty,
        "collision_penalty": collision_penalty,
        "bounds_penalty": bounds_penalty,
        "height_penalty": height_penalty,
        "reward": reward,
    }
    return reward.to(device), {key: value.to(device) for key, value in reward_info.items()}


__all__ = ["compute_thesis_exploration_reward"]
