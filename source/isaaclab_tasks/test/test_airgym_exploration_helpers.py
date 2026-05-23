# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from isaaclab.utils.math import convert_camera_frame_orientation_convention


_HELPERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "airgym_x152b"
    / "exploration_helpers.py"
)
_SPEC = importlib.util.spec_from_file_location("airgym_x152b_exploration_helpers", _HELPERS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

build_critic_grid_from_depth = _HELPERS.build_critic_grid_from_depth
build_room_object_state = _HELPERS.build_room_object_state
build_room_visibility_grid_from_depth = _HELPERS.build_room_visibility_grid_from_depth
sample_pillar_positions = _HELPERS.sample_pillar_positions


def _log_test(name: str, detail: str):
    print(f"\n[airgym-exploration] {name}: {detail}")


def test_sample_pillar_positions_respect_clearance_and_spacing():
    _log_test(
        "pillar sampling",
        "samples 6 pillars in 4 rooms and checks wall bounds, spawn clearance, and pairwise spacing",
    )
    torch.manual_seed(7)

    spawn_xy = torch.tensor(
        [[0.0, 0.0], [1.0, -1.0], [-1.25, 1.25], [0.75, 0.5]],
        dtype=torch.float32,
    )
    pillar_xy = sample_pillar_positions(
        4,
        6,
        "cpu",
        x_limits=(-5.0, 5.0),
        y_limits=(-5.0, 5.0),
        spawn_xy=spawn_xy,
        wall_margin=1.0,
        pillar_spacing=1.0,
        spawn_clearance=1.25,
    )

    assert pillar_xy.shape == (4, 6, 2)
    assert torch.all(pillar_xy[..., 0] >= -4.0)
    assert torch.all(pillar_xy[..., 0] <= 4.0)
    assert torch.all(pillar_xy[..., 1] >= -4.0)
    assert torch.all(pillar_xy[..., 1] <= 4.0)

    spawn_dist = torch.linalg.vector_norm(pillar_xy - spawn_xy.unsqueeze(1), dim=-1)
    _log_test(
        "pillar sampling",
        f"minimum spawn clearance={spawn_dist.min().item():.3f} m, expected >= 1.250 m",
    )
    assert torch.all(spawn_dist >= 1.25)

    for env_id in range(pillar_xy.shape[0]):
        pairwise = torch.linalg.vector_norm(
            pillar_xy[env_id].unsqueeze(1) - pillar_xy[env_id].unsqueeze(0), dim=-1
        )
        pairwise.fill_diagonal_(torch.inf)
        assert torch.all(pairwise >= 1.0)


def test_build_room_object_state_places_walls_first_then_pillars():
    _log_test(
        "room object state",
        "builds batched wall/pillar object states and checks wall order, pillar positions, heights, and identity quats",
    )
    wall_xy = torch.tensor([[0.0, 5.1], [0.0, -5.1], [5.1, 0.0], [-5.1, 0.0]], dtype=torch.float32)
    pillar_xy = torch.tensor(
        [[[1.0, 1.5], [-2.0, 0.5]], [[0.5, -1.0], [1.5, 2.0]]],
        dtype=torch.float32,
    )

    pos, quat = build_room_object_state(wall_xy, pillar_xy, wall_height=2.2, pillar_height=2.2)
    _log_test("room object state", f"object state shapes pos={tuple(pos.shape)}, quat={tuple(quat.shape)}")

    assert pos.shape == (2, 6, 3)
    assert quat.shape == (2, 6, 4)
    torch.testing.assert_close(pos[:, :4, :2], wall_xy.unsqueeze(0).expand(2, -1, -1))
    torch.testing.assert_close(pos[:, 4:, :2], pillar_xy)
    torch.testing.assert_close(pos[:, :, 2], torch.full((2, 6), 1.1))
    torch.testing.assert_close(quat[..., 0], torch.ones((2, 6)))
    torch.testing.assert_close(quat[..., 1:], torch.zeros((2, 6, 3)))


def test_build_critic_grid_from_depth_does_not_mark_max_range_endpoints_occupied():
    _log_test(
        "critic depth grid",
        "checks that finite hits become occupied while max-range depth contributes free space only",
    )
    depth = torch.tensor(
        [[[[1.0, 4.5], [4.5, 4.5]]]],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    root_pos_w = torch.zeros((1, 3), dtype=torch.float32)
    world_to_local = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    quat_w_world = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    camera_quat_ros = convert_camera_frame_orientation_convention(quat_w_world, origin="world", target="ros")

    grid = build_critic_grid_from_depth(
        depth,
        intrinsics,
        camera_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        camera_quat_ros=camera_quat_ros,
        root_pos_w=root_pos_w,
        world_to_local=world_to_local,
        camera_max_distance=4.5,
        x_limits=(0.0, 5.0),
        y_limits=(-2.5, 2.5),
        z_limits=(-2.0, 2.0),
        cell_size=1.0,
        free_samples_per_ray=3,
        inflation_radius=0,
    )

    assert grid.shape == (1, 2, 5, 5)
    _log_test(
        "critic depth grid",
        f"free cells={torch.sum(grid[:, 0]).item():.0f}, occupied cells={torch.sum(grid[:, 1]).item():.0f}",
    )
    assert torch.sum(grid[:, 1]).item() == 1.0
    assert torch.sum(grid[:, 0]).item() >= 1.0


def test_build_room_visibility_grid_from_depth_marks_free_and_occupied_cells():
    _log_test(
        "room visibility grid",
        "checks camera-origin ray tracing marks the camera cell free and the depth hit occupied in env-local room cells",
    )
    depth = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    quat_w_world = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    camera_quat_ros = convert_camera_frame_orientation_convention(quat_w_world, origin="world", target="ros")

    grid = build_room_visibility_grid_from_depth(
        depth,
        intrinsics,
        camera_pos_w=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        camera_quat_ros=camera_quat_ros,
        env_origins=torch.zeros((1, 3), dtype=torch.float32),
        camera_max_distance=4.5,
        x_limits=(0.0, 4.0),
        y_limits=(0.0, 4.0),
        z_limits=(0.0, 3.0),
        cell_size=1.0,
        free_samples_per_ray=3,
    )

    assert grid.shape == (1, 2, 4, 4)
    _log_test(
        "room visibility grid",
        f"expected free cell (1, 1)={grid[0, 0, 1, 1].item():.0f}, "
        f"occupied cell (2, 1)={grid[0, 1, 2, 1].item():.0f}",
    )
    assert grid[0, 0, 1, 1].item() == 1.0
    assert grid[0, 1, 2, 1].item() == 1.0


def test_build_room_visibility_grid_from_depth_does_not_mark_max_range_endpoints_occupied():
    _log_test(
        "room visibility max range",
        "checks max-range depth produces free-space evidence but no occupied endpoint",
    )
    depth = torch.tensor(
        [[[[4.5, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    quat_w_world = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    camera_quat_ros = convert_camera_frame_orientation_convention(quat_w_world, origin="world", target="ros")

    grid = build_room_visibility_grid_from_depth(
        depth,
        intrinsics,
        camera_pos_w=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        camera_quat_ros=camera_quat_ros,
        env_origins=torch.zeros((1, 3), dtype=torch.float32),
        camera_max_distance=4.5,
        x_limits=(0.0, 6.0),
        y_limits=(0.0, 4.0),
        z_limits=(0.0, 3.0),
        cell_size=1.0,
        free_samples_per_ray=5,
    )

    _log_test(
        "room visibility max range",
        f"free cells={torch.sum(grid[:, 0]).item():.0f}, occupied cells={torch.sum(grid[:, 1]).item():.0f}",
    )
    assert torch.sum(grid[:, 1]).item() == 0.0
    assert torch.sum(grid[:, 0]).item() > 0.0


def test_build_room_visibility_grid_from_depth_uses_env_local_coordinates():
    _log_test(
        "room visibility env-local frame",
        "checks world camera pose and env origin are converted into the same env-local cells as the unshifted case",
    )
    depth = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    quat_w_world = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    camera_quat_ros = convert_camera_frame_orientation_convention(quat_w_world, origin="world", target="ros")

    grid = build_room_visibility_grid_from_depth(
        depth,
        intrinsics,
        camera_pos_w=torch.tensor([[11.0, 21.0, 1.0]], dtype=torch.float32),
        camera_quat_ros=camera_quat_ros,
        env_origins=torch.tensor([[10.0, 20.0, 0.0]], dtype=torch.float32),
        camera_max_distance=4.5,
        x_limits=(0.0, 4.0),
        y_limits=(0.0, 4.0),
        z_limits=(0.0, 3.0),
        cell_size=1.0,
        free_samples_per_ray=3,
    )

    _log_test(
        "room visibility env-local frame",
        f"shifted free cell (1, 1)={grid[0, 0, 1, 1].item():.0f}, "
        f"shifted occupied cell (2, 1)={grid[0, 1, 2, 1].item():.0f}",
    )
    assert grid[0, 0, 1, 1].item() == 1.0
    assert grid[0, 1, 2, 1].item() == 1.0
