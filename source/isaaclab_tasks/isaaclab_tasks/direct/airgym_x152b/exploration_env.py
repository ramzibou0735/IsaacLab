# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.coverage import CoverageGrid2D, CoverageGrid2DCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab.utils.vae import DepthVAEEncoder

from warp_perception import GridSpec, WarpRoomPerception

from .base_env import AirGymX152bBaseEnv
from .base_env_cfg import AirGymX152bBaseEnvCfg
from .exploration_helpers import (
    build_room_object_state,
    sample_pillar_positions,
    sample_spawn_positions,
    sample_yaws,
)
from .task_common import (
    make_contact_sensor_cfg,
    make_exploration_room_cfg,
    make_plane_cfg,
)


ROOM_X_LIMITS = (-5.0, 5.0)
ROOM_Y_LIMITS = (-5.0, 5.0)
ROOM_HALF_EXTENT = 5.0
WALL_HEIGHT = 2.2
PILLAR_HEIGHT = 2.2
FLY_HEIGHT = 1.2
SPAWN_MARGIN = 1.0
PILLAR_WALL_MARGIN = 1.0
PILLAR_SPACING = 1.2
SPAWN_CLEARANCE = 1.4
CRITIC_GRID_X_LIMITS = (0.0, 4.5)
CRITIC_GRID_Y_LIMITS = (-2.25, 2.25)
CRITIC_GRID_Z_LIMITS = (-0.5, 1.6)
CRITIC_GRID_CELL_SIZE = 0.25
CRITIC_GRID_STRIDE = 4
CRITIC_GRID_INFLATION_RADIUS = 1
COVERAGE_CELL_SIZE = 0.5
VISIBILITY_GRID_CELL_SIZE = 0.5
VISIBILITY_GRID_Z_LIMITS = (0.2, WALL_HEIGHT)
# Warp perception camera: matches the VAE training domain (HFOV ~87 deg, 16:9,
# 10 m range). 135x240 upsamples cleanly (x2 nearest) to the VAE's 270x480 input.
PERCEPTION_WIDTH = 240
PERCEPTION_HEIGHT = 135
PERCEPTION_HFOV_DEG = 87.0
PERCEPTION_MAX_RANGE = 10.0
PERCEPTION_FREE_SAMPLES = 48
# Box half-extents in collection order (walls N, S, E, W; then pillars), derived
# from the cuboid sizes in make_exploration_room_cfg().
WALL_HALF_EXTENTS = torch.tensor(
    [
        [5.3, 0.1, 0.5 * WALL_HEIGHT],  # north (10.6 x 0.2 x 2.2)
        [5.3, 0.1, 0.5 * WALL_HEIGHT],  # south
        [0.1, 5.3, 0.5 * WALL_HEIGHT],  # east (0.2 x 10.6 x 2.2)
        [0.1, 5.3, 0.5 * WALL_HEIGHT],  # west
    ],
    dtype=torch.float32,
)
PILLAR_HALF_EXTENT = (0.25, 0.25, 0.5 * PILLAR_HEIGHT)  # 0.5 x 0.5 x 2.2
PATH_COVERAGE_REWARD_SCALE = 0.5
FREE_INFO_GAIN_REWARD_SCALE = 1.0
OCCUPIED_INFO_GAIN_REWARD_SCALE = 0.5
WALL_XY = torch.tensor(
    [[0.0, 5.1], [0.0, -5.1], [5.1, 0.0], [-5.1, 0.0]],
    dtype=torch.float32,
)


@configclass
class AirGymX152bExplorationEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 20.0
    observation_space = 21
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=14.0, replicate_physics=True)
    terrain = make_plane_cfg()
    contact_sensor = make_contact_sensor_cfg()
    # No RTX camera: perception is produced by the Warp sensor, so sim.render()
    # is skipped entirely during headless training.
    onboard_camera = None
    obstacles = make_exploration_room_cfg(8)
    camera_max_distance = PERCEPTION_MAX_RANGE
    camera_additive_noise_std = 0.1
    camera_multiplicative_noise_std = 0.3

    # Warp perception sensor configuration.
    perception_width = PERCEPTION_WIDTH
    perception_height = PERCEPTION_HEIGHT
    perception_hfov_deg = PERCEPTION_HFOV_DEG
    perception_free_samples = PERCEPTION_FREE_SAMPLES
    perception_use_cuda_graph = False

    # Frozen depth-VAE encoder (visual observation backbone).
    vae_weights_path: str | None = None
    vae_latent_dims = 64
    vae_return_sampled_latent = False

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 17 + self.action_space


class AirGymX152bExplorationEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bExplorationEnvCfg

    def __init__(self, cfg: AirGymX152bExplorationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._wall_xy = WALL_XY.to(device=self.device)
        self._num_walls = int(self._wall_xy.shape[0])
        self._num_pillars = self._obstacles.num_objects - self._num_walls

        self._setup_perception()
        self._vae = DepthVAEEncoder(
            weights_path=self.cfg.vae_weights_path,
            latent_dims=self.cfg.vae_latent_dims,
            max_range=self.cfg.camera_max_distance,
            return_sampled_latent=self.cfg.vae_return_sampled_latent,
            device=self.device,
        )
        self._coverage_grid = CoverageGrid2D(
            CoverageGrid2DCfg(
                x_limits=ROOM_X_LIMITS,
                y_limits=ROOM_Y_LIMITS,
                cell_size=COVERAGE_CELL_SIZE,
                mark_path=True,
                count_spawn_as_visited=True,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._last_new_visits = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._visibility_grid_h = math.ceil((ROOM_X_LIMITS[1] - ROOM_X_LIMITS[0]) / VISIBILITY_GRID_CELL_SIZE)
        self._visibility_grid_w = math.ceil((ROOM_Y_LIMITS[1] - ROOM_Y_LIMITS[0]) / VISIBILITY_GRID_CELL_SIZE)
        self._visibility_num_cells = self._visibility_grid_h * self._visibility_grid_w
        self._known_free = torch.zeros(
            (self.num_envs, self._visibility_grid_h, self._visibility_grid_w),
            device=self.device,
            dtype=torch.bool,
        )
        self._known_occupied = torch.zeros_like(self._known_free)
        self._last_new_free_cells = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_new_occupied_cells = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._episode_metric_sums = {
            "info_gain_reward": torch.zeros((self.num_envs,), device=self.device),
            "coverage_reward": torch.zeros((self.num_envs,), device=self.device),
            "proximity_penalty": torch.zeros((self.num_envs,), device=self.device),
            "collision_penalty": torch.zeros((self.num_envs,), device=self.device),
        }
        self._last_collision_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_out_of_bounds_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_bad_height_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        x_centers = ROOM_X_LIMITS[0] + VISIBILITY_GRID_CELL_SIZE * (
            torch.arange(self._visibility_grid_h, device=self.device, dtype=torch.float32) + 0.5
        )
        y_centers = ROOM_Y_LIMITS[0] + VISIBILITY_GRID_CELL_SIZE * (
            torch.arange(self._visibility_grid_w, device=self.device, dtype=torch.float32) + 0.5
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self._visibility_cell_centers = torch.stack((grid_x, grid_y), dim=-1)
        self._room_diagonal = math.hypot(
            ROOM_X_LIMITS[1] - ROOM_X_LIMITS[0],
            ROOM_Y_LIMITS[1] - ROOM_Y_LIMITS[0],
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._log_completed_episode_metrics(env_ids)
        super()._reset_idx(env_ids)

    def _setup_perception(self) -> None:
        """Construct the fused Warp perception sensor and push static box geometry."""
        num_boxes = int(self._obstacles.num_objects)
        critic_grid = GridSpec(
            CRITIC_GRID_X_LIMITS, CRITIC_GRID_Y_LIMITS, CRITIC_GRID_Z_LIMITS, CRITIC_GRID_CELL_SIZE
        )
        vis_grid = GridSpec(ROOM_X_LIMITS, ROOM_Y_LIMITS, VISIBILITY_GRID_Z_LIMITS, VISIBILITY_GRID_CELL_SIZE)
        self._perception = WarpRoomPerception(
            self.num_envs,
            num_boxes,
            height=self.cfg.perception_height,
            width=self.cfg.perception_width,
            horizontal_fov_deg=self.cfg.perception_hfov_deg,
            max_range=self.cfg.camera_max_distance,
            critic_grid=critic_grid,
            vis_grid=vis_grid,
            grid_stride=CRITIC_GRID_STRIDE,
            free_samples=self.cfg.perception_free_samples,
            device=self.device,
            use_cuda_graph=self.cfg.perception_use_cuda_graph,
        )
        self._perception.set_env_origins(self._env_origins)
        pillar_half = torch.tensor(PILLAR_HALF_EXTENT, device=self.device, dtype=torch.float32).expand(
            self._num_pillars, 3
        )
        box_half = torch.cat((WALL_HALF_EXTENTS.to(self.device), pillar_half), dim=0)
        self._perception.set_box_half_extents(box_half)

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        spawn_pos = sample_spawn_positions(
            num_resets,
            self.device,
            x_limits=ROOM_X_LIMITS,
            y_limits=ROOM_Y_LIMITS,
            z_height=FLY_HEIGHT,
            xy_margin=SPAWN_MARGIN,
            z_jitter=0.15,
        )
        spawn_yaw = sample_yaws(num_resets, self.device)
        pillar_xy = sample_pillar_positions(
            num_resets,
            self._num_pillars,
            self.device,
            x_limits=ROOM_X_LIMITS,
            y_limits=ROOM_Y_LIMITS,
            spawn_xy=spawn_pos[:, :2],
            wall_margin=PILLAR_WALL_MARGIN,
            pillar_spacing=PILLAR_SPACING,
            spawn_clearance=SPAWN_CLEARANCE,
        )
        obstacle_pos, obstacle_quat = build_room_object_state(
            self._wall_xy,
            pillar_xy,
            wall_height=WALL_HEIGHT,
            pillar_height=PILLAR_HEIGHT,
        )
        self._write_object_collection_state_local(self._obstacles, env_ids, obstacle_pos, obstacle_quat)
        # Push the new world-frame box transforms to the Warp sensor (analytic
        # ray-casting reads these directly; no sim read-back / re-render needed).
        box_center_w = obstacle_pos + self._env_origins[env_ids].unsqueeze(1)
        self._perception.set_box_transforms(box_center_w, obstacle_quat, env_ids)

        quat = quat_from_euler_xyz(
            torch.zeros_like(spawn_yaw),
            torch.zeros_like(spawn_yaw),
            spawn_yaw,
        )
        lin_vel = torch.zeros((num_resets, 3), device=self.device)
        ang_vel = torch.zeros((num_resets, 3), device=self.device)
        self._write_robot_state_local(env_ids, spawn_pos, quat, lin_vel, ang_vel)

        self._coverage_grid.reset(env_ids, spawn_pos[:, :2])
        self._last_new_visits[env_ids] = 0
        self._known_free[env_ids] = False
        self._known_occupied[env_ids] = False
        self._last_new_free_cells[env_ids] = 0
        self._last_new_occupied_cells[env_ids] = 0

        # Robot/obstacle poses were just written, so root_pos_w already reflects the
        # new spawn. Recompute perception (all envs) and seed the reset envs' known
        # space from their initial post-reset view.
        self._update_perception(force=True)
        self._update_known_space_from_depth(env_ids)

    def _resolve_map_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _room_visibility_grid(self) -> torch.Tensor:
        """Env-local free/occupied visibility grid for this step (from Warp)."""
        self._update_perception()
        return self._perception.visibility_grid

    def _update_known_space_from_depth(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = self._resolve_map_env_ids(env_ids)
        visible_grid = self._room_visibility_grid()
        visible_free = visible_grid[:, 0] > 0.0
        visible_occupied = visible_grid[:, 1] > 0.0
        visible_free &= ~visible_occupied

        env_visible_free = visible_free[env_ids]
        env_visible_occupied = visible_occupied[env_ids]
        previously_known = self._known_free[env_ids] | self._known_occupied[env_ids]

        new_occupied = env_visible_occupied & ~previously_known
        new_free = env_visible_free & ~previously_known
        new_free &= ~env_visible_occupied

        self._known_occupied[env_ids] |= env_visible_occupied
        self._known_free[env_ids] |= env_visible_free
        self._known_free[env_ids] &= ~self._known_occupied[env_ids]

        return new_free.flatten(1).sum(dim=-1), new_occupied.flatten(1).sum(dim=-1)

    def _known_space_ratio(self) -> torch.Tensor:
        known = self._known_free | self._known_occupied
        return known.flatten(1).to(dtype=torch.float32).mean(dim=-1)

    def _map_feature_observation(self, root_pos_local: torch.Tensor, world_to_local: torch.Tensor) -> torch.Tensor:
        known = self._known_free | self._known_occupied
        unknown = ~known
        unknown_neighbor = F.max_pool2d(unknown.to(dtype=torch.float32).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        frontier = self._known_free & (unknown_neighbor > 0.0)

        cell_centers = self._visibility_cell_centers.unsqueeze(0)
        delta_xy = cell_centers - root_pos_local[:, None, None, :2]
        distance = torch.linalg.vector_norm(delta_xy, dim=-1)
        masked_distance = torch.where(frontier, distance, torch.full_like(distance, torch.inf))
        nearest_distance, nearest_idx = masked_distance.flatten(1).min(dim=-1)
        has_frontier = torch.isfinite(nearest_distance)

        nearest_xy = self._visibility_cell_centers.reshape(-1, 2)[nearest_idx]
        nearest_delta_xy = nearest_xy - root_pos_local[:, :2]
        nearest_delta = torch.cat(
            (nearest_delta_xy, torch.zeros((self.num_envs, 1), device=self.device, dtype=root_pos_local.dtype)),
            dim=-1,
        )
        nearest_delta_local = torch.einsum("bij,bj->bi", world_to_local, nearest_delta)
        nearest_bearing = torch.atan2(nearest_delta_local[:, 1], nearest_delta_local[:, 0]) / torch.pi
        nearest_distance = torch.where(
            has_frontier,
            torch.clamp(nearest_distance / self._room_diagonal, 0.0, 1.0),
            torch.ones_like(nearest_distance),
        )
        nearest_bearing = torch.where(has_frontier, nearest_bearing, torch.zeros_like(nearest_bearing))

        delta = torch.cat(
            (delta_xy, torch.zeros((*delta_xy.shape[:-1], 1), device=self.device, dtype=root_pos_local.dtype)),
            dim=-1,
        )
        local_delta = torch.einsum("bij,bhwj->bhwi", world_to_local, delta)
        ahead = (
            (local_delta[..., 0] >= CRITIC_GRID_X_LIMITS[0])
            & (local_delta[..., 0] < CRITIC_GRID_X_LIMITS[1])
            & (local_delta[..., 1] >= CRITIC_GRID_Y_LIMITS[0])
            & (local_delta[..., 1] < CRITIC_GRID_Y_LIMITS[1])
        )
        ahead_count = ahead.flatten(1).sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        unknown_ahead_ratio = (unknown & ahead).flatten(1).sum(dim=-1).to(dtype=torch.float32) / ahead_count
        known_space_ratio = self._known_space_ratio()

        return torch.stack((nearest_distance, nearest_bearing, unknown_ahead_ratio, known_space_ratio), dim=-1)

    def _state_observation(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        world_to_local, euler_local, vel_local, ang_vel_local = self._compute_yaw_local_frame()
        coverage_ratio = self._coverage_grid.coverage_ratio().unsqueeze(-1)
        map_features = self._map_feature_observation(root_pos_local, world_to_local)

        position_obs = torch.stack(
            (
                root_pos_local[:, 0] / ROOM_HALF_EXTENT,
                root_pos_local[:, 1] / ROOM_HALF_EXTENT,
                (root_pos_local[:, 2] - FLY_HEIGHT) / FLY_HEIGHT,
            ),
            dim=-1,
        )
        return torch.cat(
            (position_obs, euler_local, vel_local, ang_vel_local, self._policy_actions, coverage_ratio, map_features),
            dim=-1,
        )

    def _critic_grid_observation(self) -> torch.Tensor:
        """Ego-local two-channel critic occupancy grid from the Warp sensor.

        Mirrors the previous PyTorch helper's semantics: occupied cells are
        dilated by one cell and free cells beneath occupied ones are cleared.
        """
        self._update_perception()
        grid = self._perception.critic_grid
        occupied = grid[:, 1]
        if CRITIC_GRID_INFLATION_RADIUS > 0:
            ksize = 2 * CRITIC_GRID_INFLATION_RADIUS + 1
            occupied = F.max_pool2d(
                occupied.unsqueeze(1), ksize, stride=1, padding=CRITIC_GRID_INFLATION_RADIUS
            ).squeeze(1)
        free = torch.where(occupied > 0, torch.zeros_like(grid[:, 0]), grid[:, 0])
        return torch.stack((free, occupied), dim=1)

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        # Perception (depth/critic/visibility) is refreshed for all envs in the
        # reward step and re-run for reset envs inside _reset_task, so the cache is
        # already valid here. Known-space seeding for reset envs happens in
        # _reset_task, not here.
        self._update_perception()
        obs = self._state_observation()
        latent = self._vae.encode(self._camera_depth_metric_image())
        return {
            "policy": obs,
            "observation": obs,
            "latent": latent,
            "critic_grid": self._critic_grid_observation(),
        }

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        # Refresh perception once (all envs, post-physics) before any consumer.
        self._update_perception()
        root_pos_local = self._root_pos_local()
        _, _, _, ang_vel_local = self._compute_yaw_local_frame()
        collision = self._collision_state()
        new_visit_count, in_bounds = self._coverage_grid.update(None, root_pos_local[:, :2])
        self._last_new_visits.copy_(new_visit_count)
        coverage_ratio = self._coverage_grid.coverage_ratio()
        new_free_cells, new_occupied_cells = self._update_known_space_from_depth()
        self._last_new_free_cells.copy_(new_free_cells)
        self._last_new_occupied_cells.copy_(new_occupied_cells)
        known_space_ratio = self._known_space_ratio()

        depth_metric = self._camera_depth_metric_image()
        min_depth = depth_metric.view(self.num_envs, -1).amin(dim=-1)
        proximity_penalty = -2.0 * torch.clamp(0.45 - min_depth, min=0.0)

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        upright_reward = 0.3 * torch.square((ups[:, 2] + 1.0) / 2.0)
        height_reward = 0.3 * torch.exp(-4.0 * torch.square(root_pos_local[:, 2] - FLY_HEIGHT))

        action_diff = self._actions - self._previous_actions
        action_smoothness_reward = 0.1 * torch.exp(-torch.linalg.vector_norm(action_diff, dim=-1))
        spin_reward = 0.1 / (1.0 + torch.square(torch.linalg.vector_norm(ang_vel_local, dim=-1)))
        path_coverage_reward = PATH_COVERAGE_REWARD_SCALE * new_visit_count.to(dtype=torch.float32)
        info_gain_reward = (
            FREE_INFO_GAIN_REWARD_SCALE * new_free_cells.to(dtype=torch.float32)
            + OCCUPIED_INFO_GAIN_REWARD_SCALE * new_occupied_cells.to(dtype=torch.float32)
        )
        coverage_reward = path_coverage_reward
        alive_reward = torch.full((self.num_envs,), 0.1, device=self.device)

        out_of_bounds = ~in_bounds
        bad_height = (root_pos_local[:, 2] < 0.35) | (root_pos_local[:, 2] > 2.0)
        collision_penalty = torch.where(collision, torch.full((self.num_envs,), -50.0, device=self.device), 0.0)
        bounds_penalty = torch.where(out_of_bounds, torch.full((self.num_envs,), -25.0, device=self.device), 0.0)
        height_penalty = torch.where(bad_height, torch.full((self.num_envs,), -20.0, device=self.device), 0.0)

        reward = (
            coverage_reward
            + info_gain_reward
            + alive_reward
            + upright_reward
            + height_reward
            + action_smoothness_reward
            + spin_reward
            + proximity_penalty
            + collision_penalty
            + bounds_penalty
            + height_penalty
        )

        reward_info = {
            "coverage_reward": coverage_reward,
            "path_coverage_reward": path_coverage_reward,
            "info_gain_reward": info_gain_reward,
            "coverage_ratio": coverage_ratio,
            "known_space_ratio": known_space_ratio,
            "new_visit_count": new_visit_count.to(dtype=torch.float32),
            "new_free_cells": new_free_cells.to(dtype=torch.float32),
            "new_occupied_cells": new_occupied_cells.to(dtype=torch.float32),
            "alive_reward": alive_reward,
            "upright_reward": upright_reward,
            "height_reward": height_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "spin_reward": spin_reward,
            "proximity_penalty": proximity_penalty,
            "collision_penalty": collision_penalty,
            "bounds_penalty": bounds_penalty,
            "height_penalty": height_penalty,
            "reward": reward,
        }
        for key in self._episode_metric_sums:
            self._episode_metric_sums[key] += reward_info[key]
        return reward, reward_info

    def _log_completed_episode_metrics(self, env_ids: torch.Tensor) -> None:
        episode_lengths = self.episode_length_buf[env_ids].to(dtype=torch.float32)
        completed_episode = self.reset_terminated[env_ids] | self.reset_time_outs[env_ids]
        valid_episode = completed_episode & (episode_lengths > 0)
        if not torch.any(valid_episode):
            return

        valid_env_ids = env_ids[valid_episode]
        episode_lengths = episode_lengths[valid_episode].clamp_min(1.0)
        coverage_ratio = self._coverage_grid.coverage_ratio()[valid_env_ids]
        known_space_ratio = self._known_space_ratio()[valid_env_ids]

        log = {}
        for key, value in self._episode_metric_sums.items():
            log[f"Episode_Reward/{key}_mean"] = torch.mean(value[valid_env_ids] / episode_lengths)
            value[valid_env_ids] = 0.0

        log["Episode_Termination/collision_rate"] = self._last_collision_termination[valid_env_ids].to(
            dtype=torch.float32
        ).mean()
        log["Episode_Termination/out_of_bounds_rate"] = self._last_out_of_bounds_termination[valid_env_ids].to(
            dtype=torch.float32
        ).mean()
        log["Episode_Termination/bad_height_rate"] = self._last_bad_height_termination[valid_env_ids].to(
            dtype=torch.float32
        ).mean()
        log["Episode_Termination/time_out_rate"] = self.reset_time_outs[valid_env_ids].to(dtype=torch.float32).mean()
        log["Metrics/coverage_ratio"] = coverage_ratio.mean()
        log["Metrics/known_space_ratio"] = known_space_ratio.mean()

        self.extras["log"] = log

    def _compute_terminated(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)

        out_of_bounds = root_pos_local[:, 0] < ROOM_X_LIMITS[0]
        out_of_bounds |= root_pos_local[:, 0] > ROOM_X_LIMITS[1]
        out_of_bounds |= root_pos_local[:, 1] < ROOM_Y_LIMITS[0]
        out_of_bounds |= root_pos_local[:, 1] > ROOM_Y_LIMITS[1]
        bad_height = (root_pos_local[:, 2] < 0.35) | (root_pos_local[:, 2] > 2.0)
        collision = self._collision_state()

        self._last_out_of_bounds_termination.copy_(out_of_bounds)
        self._last_bad_height_termination.copy_(bad_height)
        self._last_collision_termination.copy_(collision)

        terminated = out_of_bounds | bad_height
        terminated |= ups[:, 2] < 0.0
        terminated |= collision
        return terminated
