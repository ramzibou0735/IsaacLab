# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict


_THESIS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "thesis"
)


def _load_module(name: str, relative_path: str):
    path = _THESIS_PATH / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_thesis_obs(num_envs: int) -> TensorDict:
    return TensorDict(
        {
            "observations": torch.zeros(num_envs, 32),
            "observations_image": torch.zeros(num_envs, 2, 135, 240),
            "observations_vae": torch.zeros(num_envs, 64),
            "observations_map": torch.zeros(num_envs, 4, 31, 31, 31),
            "critic_observations": torch.zeros(num_envs, 51),
        },
        batch_size=[num_envs],
    )


def test_thesis_active_map_channels_are_consistent_and_resettable():
    active_map = _load_module("thesis_active_map", "active_map.py")
    occupancy_map = torch.zeros(2, 7, 7, 7)
    occupancy_map[0, 3, 3, 3] = 1.0
    occupancy_map[0, 4, 3, 3] = 2.0
    occupancy_map[1, 2, 2, 2] = 1.0
    known_free = torch.zeros_like(occupancy_map, dtype=torch.bool)
    known_occupied = torch.zeros_like(occupancy_map, dtype=torch.bool)
    recency = torch.zeros_like(occupancy_map)

    new_free, new_occupied, new_total = active_map.update_thesis_map_tensors(
        occupancy_map,
        known_free,
        known_occupied,
        recency,
        recency_decay=0.9,
    )

    assert new_free.tolist() == [1, 1]
    assert new_occupied.tolist() == [1, 0]
    assert new_total.tolist() == [2, 1]
    channels = active_map.compose_thesis_map_channels(known_free, known_occupied, recency)
    assert channels.shape == (2, 4, 7, 7, 7)
    assert not torch.any(channels[:, 0].bool() & channels[:, 1].bool())
    assert torch.all((channels[:, 0] + channels[:, 1] + channels[:, 2]) == 1.0)
    assert channels[0, 3, 3, 3, 3] == 1.0
    assert channels[0, 2, 3, 3, 3] == 0.0
    assert channels[0, 1, 4, 3, 3] == 1.0

    active_map.reset_thesis_map_tensors(known_free, known_occupied, recency, torch.tensor([0]))
    assert torch.count_nonzero(known_free[0]) == 0
    assert torch.count_nonzero(known_occupied[0]) == 0
    assert torch.count_nonzero(recency[0]) == 0
    assert torch.count_nonzero(known_free[1]) == 1


def test_thesis_gym_registration_metadata_is_declared():
    package_init = (_THESIS_PATH / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-Thesis-X152b-Exploration-Direct-v0" in package_init
    assert "exploration_env:ThesisX152bExplorationEnv" in package_init
    assert "ThesisX152bExplorationPPORunnerCfg" in package_init


def test_thesis_vae_mode_updates_observation_contract():
    source = (_THESIS_PATH / "exploration_env.py").read_text(encoding="utf-8")
    assert "from isaaclab.utils.vae import DepthVAEEncoder" in source
    assert "THESIS_VAE_LATENT_DIM = 64" in source
    assert '"observations_vae"' in source
    assert "def set_vae_mode" in source
    assert "self._vae.encode(self._camera_depth_metric_image())" in source
    assert 'obs["observations_image"] = self._image_observation()' in source


def test_thesis_local_crop_follows_robot_pose_and_orientation():
    active_map = _load_module("thesis_active_map_crop", "active_map.py")
    map_channels = torch.zeros(1, 4, 11, 11, 11)
    map_channels[0, 1, 6, 5, 5] = 1.0
    index_base, position_base = active_map.build_local_map_bases(1, 5, 0.2, "cpu")
    root = torch.zeros(1, 3)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    local_map = active_map.extract_centered_map_channels(
        map_channels,
        5,
        root,
        identity_quat,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        index_base,
        position_base,
    )

    assert local_map.shape == (1, 4, 5, 5, 5)
    assert local_map[0, 1, 3, 2, 2] == 1.0

    yaw_180_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    rotated = active_map.extract_centered_map_channels(
        map_channels,
        5,
        root,
        yaw_180_quat,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        index_base,
        position_base,
    )
    assert rotated[0, 1, 1, 2, 2] == 1.0


def test_thesis_network_shapes_and_recurrent_minibatch_path():
    thesis_ppo = _load_module("thesis_active_map_ppo_shapes", "agents/thesis_active_map_ppo.py")
    obs = _make_thesis_obs(4)
    obs_groups = {
        "actor": ["observations", "observations_image", "observations_map"],
        "critic": ["critic_observations"],
    }
    model = thesis_ppo.ThesisActiveMapActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        obs_normalization=False,
        critic_obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )

    actions, values = model.act_and_evaluate(obs, stochastic_output=True)
    assert obs["observations"].shape == (4, 32)
    assert obs["observations_image"].shape == (4, 2, 135, 240)
    assert obs["observations_map"].shape == (4, 4, 31, 31, 31)
    assert actions.shape == (4, 4)
    assert values.shape == (4, 1)
    assert model.get_hidden_state().shape == (1, 4, 512)

    sequence_obs = TensorDict(
        {
            "observations": torch.zeros(3, 4, 32),
            "observations_image": torch.zeros(3, 4, 2, 135, 240),
            "observations_map": torch.zeros(3, 4, 4, 31, 31, 31),
            "critic_observations": torch.zeros(3, 4, 51),
        },
        batch_size=[3, 4],
    )
    seq_actions, seq_values = model.act_and_evaluate(
        sequence_obs,
        masks=torch.ones(3, 4, dtype=torch.bool),
        hidden_state=torch.zeros(1, 4, 512),
        stochastic_output=True,
    )
    assert seq_actions.shape == (3, 4, 4)
    assert seq_values.shape == (3, 4, 1)


def test_thesis_ppo_update_smoke_without_nans():
    thesis_ppo = _load_module("thesis_active_map_ppo_update", "agents/thesis_active_map_ppo.py")
    from rsl_rl.storage import RolloutStorage

    obs = _make_thesis_obs(2)
    obs_groups = {
        "actor": ["observations", "observations_image", "observations_map"],
        "critic": ["critic_observations"],
    }
    model = thesis_ppo.ThesisActiveMapActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        obs_normalization=False,
        critic_obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    storage = RolloutStorage("rl", 2, 2, obs, [4], "cpu")
    alg = thesis_ppo.ThesisActiveMapPPO(
        model,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    for _ in range(2):
        alg.act(obs)
        alg.process_env_step(obs, torch.ones(2), torch.zeros(2, dtype=torch.long), {})
    alg.compute_returns(obs)
    losses = alg.update()

    assert set(losses) == {"value", "surrogate", "entropy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_thesis_vae_policy_uses_latent_visual_branch_without_depth_image():
    thesis_ppo = _load_module("thesis_active_map_ppo_vae", "agents/thesis_active_map_ppo.py")
    obs = TensorDict(
        {
            "observations": torch.zeros(4, 32),
            "observations_vae": torch.zeros(4, 64),
            "observations_map": torch.zeros(4, 4, 31, 31, 31),
            "critic_observations": torch.zeros(4, 51),
        },
        batch_size=[4],
    )
    obs_groups = {
        "actor": ["observations", "observations_vae", "observations_map"],
        "critic": ["critic_observations"],
    }
    model = thesis_ppo.ThesisActiveMapActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        use_vae_latent=True,
        obs_normalization=False,
        critic_obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )

    actions, values = model.act_and_evaluate(obs, stochastic_output=True)
    assert actions.shape == (4, 4)
    assert values.shape == (4, 1)
    assert model.get_hidden_state().shape == (1, 4, 512)

    sequence_obs = TensorDict(
        {
            "observations": torch.zeros(3, 4, 32),
            "observations_vae": torch.zeros(3, 4, 64),
            "observations_map": torch.zeros(3, 4, 4, 31, 31, 31),
            "critic_observations": torch.zeros(3, 4, 51),
        },
        batch_size=[3, 4],
    )
    seq_actions, seq_values = model.act_and_evaluate(
        sequence_obs,
        masks=torch.ones(3, 4, dtype=torch.bool),
        hidden_state=torch.zeros(1, 4, 512),
        stochastic_output=True,
    )
    assert seq_actions.shape == (3, 4, 4)
    assert seq_values.shape == (3, 4, 1)

    onnx_model = model.as_onnx()
    assert onnx_model.input_names == ["observations", "observations_vae", "observations_map", "h_in"]
    onnx_actions, h_out = onnx_model(*onnx_model.get_dummy_inputs())
    assert onnx_actions.shape == (1, 4)
    assert h_out.shape == (1, 1, 512)


def test_thesis_vae_ppo_update_smoke_without_nans():
    thesis_ppo = _load_module("thesis_active_map_ppo_vae_update", "agents/thesis_active_map_ppo.py")
    from rsl_rl.storage import RolloutStorage

    obs = TensorDict(
        {
            "observations": torch.zeros(2, 32),
            "observations_vae": torch.zeros(2, 64),
            "observations_map": torch.zeros(2, 4, 31, 31, 31),
            "critic_observations": torch.zeros(2, 51),
        },
        batch_size=[2],
    )
    obs_groups = {
        "actor": ["observations", "observations_vae", "observations_map"],
        "critic": ["critic_observations"],
    }
    model = thesis_ppo.ThesisActiveMapActorCritic(
        obs,
        obs_groups,
        "actor",
        4,
        use_vae_latent=True,
        obs_normalization=False,
        critic_obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    storage = RolloutStorage("rl", 2, 2, obs, [4], "cpu")
    alg = thesis_ppo.ThesisActiveMapPPO(
        model,
        storage,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    for _ in range(2):
        alg.act(obs)
        alg.process_env_step(obs, torch.ones(2), torch.zeros(2, dtype=torch.long), {})
    alg.compute_returns(obs)
    losses = alg.update()

    assert set(losses) == {"value", "surrogate", "entropy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_thesis_reward_prefers_new_information_and_terminal_penalties():
    rewards = _load_module("thesis_rewards", "rewards.py")
    base_args = {
        "new_free_cells": torch.tensor([0.0, 64.0, 0.0]),
        "new_occupied_cells": torch.tensor([0.0, 4.0, 0.0]),
        "new_visit_count": torch.tensor([0.0, 2.0, 0.0]),
        "coverage_ratio": torch.tensor([0.1, 0.35, 0.1]),
        "frontier_count": torch.tensor([0.0, 0.4, 0.0]),
        "revisit_ratio": torch.tensor([1.0, 0.0, 0.0]),
        "min_depth": torch.tensor([1.0, 1.0, 1.0]),
        "root_height": torch.tensor([1.2, 1.2, 1.2]),
        "up_z": torch.tensor([1.0, 1.0, 1.0]),
        "ang_vel_norm": torch.tensor([0.0, 0.0, 0.0]),
        "action_diff_norm": torch.tensor([0.0, 0.0, 0.0]),
        "collision": torch.tensor([False, False, True]),
        "out_of_bounds": torch.tensor([False, False, False]),
        "bad_height": torch.tensor([False, False, False]),
        "revisit_penalty_scale": 0.1,
        "device": "cpu",
    }
    reward, info = rewards.compute_thesis_exploration_reward(**base_args)

    assert reward[1] > reward[0]
    assert info["revisit_penalty"][0] < 0.0
    assert reward[2] < -400.0
    assert info["collision_penalty"][2] == -450.0
