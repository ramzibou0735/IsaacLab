# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


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

build_room_object_state = _HELPERS.build_room_object_state
sample_pillar_positions = _HELPERS.sample_pillar_positions


def _log_test(name: str, detail: str):
    print(f"\n[airgym-exploration] {name}: {detail}")


def test_sample_pillar_positions_respect_clearance_and_spacing():
    # The de-synced sampler runs a fixed number of rejection iterations (no
    # bool(torch.all(...)) early break); it must still honor wall bounds, spawn
    # clearance, and pairwise spacing when valid placements exist.
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


def test_sample_pillar_positions_is_sync_free_and_shaped():
    # A zero-pillar request returns an empty tensor; non-zero requests always
    # return a fully populated tensor even under tight constraints (fallback).
    _log_test("pillar sampling", "checks empty-pillar shortcut and fully-populated fallback output")
    torch.manual_seed(0)

    empty = sample_pillar_positions(
        3, 0, "cpu", x_limits=(-5.0, 5.0), y_limits=(-5.0, 5.0), spawn_xy=torch.zeros((3, 2)),
        wall_margin=1.0, pillar_spacing=1.0, spawn_clearance=1.0,
    )
    assert empty.shape == (3, 0, 2)

    filled = sample_pillar_positions(
        2, 5, "cpu", x_limits=(-2.0, 2.0), y_limits=(-2.0, 2.0), spawn_xy=torch.zeros((2, 2)),
        wall_margin=0.5, pillar_spacing=0.5, spawn_clearance=0.5, max_tries=8,
    )
    assert filled.shape == (2, 5, 2)
    assert torch.isfinite(filled).all()


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
