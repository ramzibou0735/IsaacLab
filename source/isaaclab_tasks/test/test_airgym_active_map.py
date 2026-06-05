# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict


_AIRGYM_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "airgym_x152b"
)


def _load_module(name: str, relative_path: str):
    path = _AIRGYM_PATH / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_centered_map_matches_synthetic_global_cells():
    active_map = _load_module("airgym_active_map", "active_map.py")
    occupancy_map = torch.zeros(2, 121, 81, 61)
    occupancy_map[0, 60, 40, 30] = 2.0
    occupancy_map[1, 61, 40, 30] = 1.0
    index_base, position_base = active_map.build_local_map_bases(2, 21, 0.1, "cpu")
    quat = torch.zeros(2, 4)
    quat[:, 0] = 1.0
    root = torch.zeros(2, 3)

    local_map = active_map.extract_centered_map(
        occupancy_map,
        21,
        root,
        quat,
        (-6.0, -4.0, -3.0),
        (6.0, 4.0, 3.0),
        index_base,
        position_base,
    )

    assert local_map.shape == (2, 21, 21, 21)
    assert local_map[0, 10, 10, 10] == 2.0
    assert local_map[1, 11, 10, 10] == 1.0


def test_project_occupancy_to_bev_does_not_clamp_outside_map_bounds():
    active_map = _load_module("airgym_active_map", "active_map.py")
    occupancy_map = torch.zeros(1, 121, 81, 61)
    occupancy_map[0, 60, 0, 30] = 2.0

    projected = active_map.project_occupancy_to_bev(
        occupancy_map,
        x_limits=(-0.25, 0.25),
        y_limits=(-5.0, -4.0),
        z_limits=(0.0, 0.1),
        cell_size=0.5,
        map_bounds_min=(-6.0, -4.0, -3.0),
        map_bounds_max=(6.0, 4.0, 3.0),
    )

    assert projected.shape == (1, 2, 1, 2)
    assert torch.count_nonzero(projected) == 0


def test_active_map_network_shapes_and_recurrent_minibatch_path():
    active_map_ppo = _load_module("airgym_active_map_ppo", "agents/active_map_ppo.py")
    from rsl_rl.storage import RolloutStorage

    obs = TensorDict(
        {
            "observations": torch.zeros(4, 85),
            "observations_map": torch.zeros(4, 1, 21, 21, 21),
        },
        batch_size=[4],
    )
    obs_groups = {"actor": ["observations", "observations_map"], "critic": ["observations", "observations_map"]}
    model = active_map_ppo.ActiveMapSharedActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )

    map_encoding = model.map_encoder(torch.zeros(4, 1, 21, 21, 21))
    assert map_encoding.shape == (4, 432)
    actions, values = model.act_and_evaluate(obs, stochastic_output=True)
    assert actions.shape == (4, 4)
    assert values.shape == (4, 1)
    assert model.get_hidden_state().shape == (1, 4, 64)

    sequence_obs = TensorDict(
        {
            "observations": torch.zeros(3, 4, 85),
            "observations_map": torch.zeros(3, 4, 1, 21, 21, 21),
        },
        batch_size=[3, 4],
    )
    masks = torch.ones(3, 4, dtype=torch.bool)
    hidden = torch.zeros(1, 4, 64)
    seq_actions, seq_values = model.act_and_evaluate(
        sequence_obs,
        masks=masks,
        hidden_state=hidden,
        stochastic_output=True,
    )
    assert seq_actions.shape == (3, 4, 4)
    assert seq_values.shape == (3, 4, 1)

    storage = RolloutStorage("rl", 4, 2, obs, [4], "cpu")
    alg = active_map_ppo.ActiveMapPPO(
        model,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        device="cpu",
    )
    for _ in range(2):
        alg.act(obs)
        next_obs = TensorDict(
            {
                "observations": torch.zeros(4, 85),
                "observations_map": torch.zeros(4, 1, 21, 21, 21),
            },
            batch_size=[4],
        )
        alg.process_env_step(next_obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {})
    alg.compute_returns(obs)
    losses = alg.update()
    assert set(losses) == {"value", "surrogate", "entropy"}

    small_obs = TensorDict(
        {
            "observations": torch.zeros(2, 85),
            "observations_map": torch.zeros(2, 1, 21, 21, 21),
        },
        batch_size=[2],
    )
    small_model = active_map_ppo.ActiveMapSharedActorCritic(
        small_obs,
        obs_groups,
        "actor",
        4,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    small_storage = RolloutStorage("rl", 2, 2, small_obs, [4], "cpu")
    small_alg = active_map_ppo.ActiveMapPPO(
        small_model,
        small_storage,
        num_learning_epochs=1,
        num_mini_batches=4,
        device="cpu",
    )
    for _ in range(2):
        small_alg.act(small_obs)
        small_alg.process_env_step(small_obs, torch.ones(2), torch.zeros(2, dtype=torch.long), {})
    small_alg.compute_returns(small_obs)
    small_losses = small_alg.update()
    assert set(small_losses) == {"value", "surrogate", "entropy"}


def test_active_map_policy_export_wrappers(tmp_path):
    active_map_ppo = _load_module("airgym_active_map_ppo_export", "agents/active_map_ppo.py")
    obs = TensorDict(
        {
            "observations": torch.zeros(1, 85),
            "observations_map": torch.zeros(1, 1, 21, 21, 21),
        },
        batch_size=[1],
    )
    obs_groups = {"actor": ["observations", "observations_map"], "critic": ["observations", "observations_map"]}
    model = active_map_ppo.ActiveMapSharedActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )

    jit_model = model.as_jit()
    scripted = torch.jit.script(jit_model)
    actions = scripted(obs["observations"], obs["observations_map"])
    assert actions.shape == (1, 4)
    scripted.reset()

    onnx_model = model.as_onnx()
    onnx_inputs = onnx_model.get_dummy_inputs()
    actions, h_out = onnx_model(*onnx_inputs)
    assert actions.shape == (1, 4)
    assert h_out.shape == (1, 1, 64)
    try:
        import onnx  # noqa: F401
    except ImportError:
        pytest.skip("onnx package is not installed")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_inputs,
        tmp_path / "policy.onnx",
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )


def test_active_perception_warp_sensor_cpu_smoke():
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/warp-cache")
    active_sensor = _load_module("airgym_active_perception_sensor", "active_perception_sensor.py")
    sensor = active_sensor.ActivePerceptionWarpSensor(
        1,
        1,
        height=8,
        width=8,
        horizontal_fov_deg=87.0,
        max_range=10.0,
        device="cpu",
    )
    sensor.set_env_origins(torch.zeros(1, 3))
    sensor.set_box_half_extents(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32))
    sensor.set_box_transforms(torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32))

    camera_quat = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float32)
    depth, occupancy_map = sensor.compute(torch.zeros(1, 3), camera_quat)

    assert depth.shape == (1, 8, 8)
    assert occupancy_map.shape == (1, 121, 81, 61)
    torch.testing.assert_close(depth[0, 4, 4], torch.tensor(1.5), atol=1.0e-5, rtol=1.0e-5)
    assert torch.count_nonzero(occupancy_map == 2.0) > 0


def test_active_perception_warp_sensor_uses_env_origins_for_map_indexing():
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/warp-cache")
    active_sensor = _load_module("airgym_active_perception_sensor_origin", "active_perception_sensor.py")
    sensor = active_sensor.ActivePerceptionWarpSensor(
        2,
        1,
        height=8,
        width=8,
        horizontal_fov_deg=87.0,
        max_range=10.0,
        device="cpu",
    )
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=torch.float32)
    sensor.set_env_origins(env_origins)
    sensor.set_box_half_extents(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32))
    sensor.set_box_transforms(
        torch.tensor([[[2.0, 0.0, 0.0]], [[22.0, 0.0, 0.0]]], dtype=torch.float32)
    )

    camera_quat = torch.tensor([[0.5, -0.5, 0.5, -0.5], [0.5, -0.5, 0.5, -0.5]], dtype=torch.float32)
    depth, occupancy_map = sensor.compute(env_origins, camera_quat)

    torch.testing.assert_close(depth[:, 4, 4], torch.full((2,), 1.5), atol=1.0e-5, rtol=1.0e-5)
    assert occupancy_map[0, 75, 40, 30] == 2.0
    assert occupancy_map[1, 75, 40, 30] == 2.0
