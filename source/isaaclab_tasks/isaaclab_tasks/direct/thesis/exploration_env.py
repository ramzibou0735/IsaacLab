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

from isaaclab_tasks.direct.airgym_x152b.base_env import AirGymX152bBaseEnv
from isaaclab_tasks.direct.airgym_x152b.base_env_cfg import AirGymX152bBaseEnvCfg
from isaaclab_tasks.direct.airgym_x152b.exploration_helpers import (
    build_room_object_state,
    sample_pillar_positions,
    sample_spawn_positions,
    sample_yaws,
)
from isaaclab_tasks.direct.airgym_x152b.task_common import (
    make_contact_sensor_cfg,
    make_exploration_room_cfg,
    make_plane_cfg,
)

from .active_perception_sensor import ThesisActivePerceptionWarpSensor
from .active_map import (
    build_local_map_bases,
    compose_thesis_map_channels,
    extract_centered_map_channels,
    project_thesis_map_to_bev,
    reset_thesis_map_tensors,
    stamp_robot_recency,
    update_thesis_map_tensors,
)
from .rewards import compute_thesis_exploration_reward


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
COVERAGE_CELL_SIZE = 0.5
VISIBILITY_GRID_CELL_SIZE = 0.5
VISIBILITY_GRID_Z_LIMITS = (0.2, WALL_HEIGHT)
PERCEPTION_WIDTH = 240
PERCEPTION_HEIGHT = 135
PERCEPTION_HFOV_DEG = 87.0
PERCEPTION_MAX_RANGE = 10.0
OCCUPANCY_MAP_BOUNDS_MIN = (-6.0, -4.0, -3.0)
OCCUPANCY_MAP_BOUNDS_MAX = (6.0, 4.0, 3.0)
OCCUPANCY_MAP_SHAPE = (121, 81, 61)
LOCAL_MAP_SIZE = 31
LOCAL_MAP_CELL_SIZE = 0.2
THESIS_STATE_DIM = 32
THESIS_CRITIC_DIM = 51
THESIS_VAE_LATENT_DIM = 64
THESIS_IMAGE_SHAPE = (2, PERCEPTION_HEIGHT, PERCEPTION_WIDTH)
THESIS_MAP_SHAPE = (4, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE)

WALL_HALF_EXTENTS = torch.tensor(
    [
        [5.3, 0.1, 0.5 * WALL_HEIGHT],
        [5.3, 0.1, 0.5 * WALL_HEIGHT],
        [0.1, 5.3, 0.5 * WALL_HEIGHT],
        [0.1, 5.3, 0.5 * WALL_HEIGHT],
    ],
    dtype=torch.float32,
)
PILLAR_HALF_EXTENT = (0.25, 0.25, 0.5 * PILLAR_HEIGHT)
WALL_XY = torch.tensor(
    [[0.0, 5.1], [0.0, -5.1], [5.1, 0.0], [-5.1, 0.0]],
    dtype=torch.float32,
)

CURRICULUM_STAGES = (
    {
        "active_pillars": 0,
        "episode_length_s": 18.0,
        "spawn_z_jitter": 0.02,
        "yaw_range": 0.0,
        "pillar_spacing": 2.2,
        "spawn_clearance": 2.1,
        "camera_additive_noise_std": 0.0,
        "camera_multiplicative_noise_std": 0.0,
        "revisit_penalty_scale": 0.0,
    },
    {
        "active_pillars": 0,
        "episode_length_s": 20.0,
        "spawn_z_jitter": 0.05,
        "yaw_range": math.pi,
        "pillar_spacing": 2.2,
        "spawn_clearance": 2.1,
        "camera_additive_noise_std": 0.015,
        "camera_multiplicative_noise_std": 0.04,
        "revisit_penalty_scale": 0.01,
    },
    {
        "active_pillars": 3,
        "episode_length_s": 22.0,
        "spawn_z_jitter": 0.08,
        "yaw_range": math.pi,
        "pillar_spacing": 1.8,
        "spawn_clearance": 1.8,
        "camera_additive_noise_std": 0.02,
        "camera_multiplicative_noise_std": 0.06,
        "revisit_penalty_scale": 0.02,
    },
    {
        "active_pillars": 8,
        "episode_length_s": 24.0,
        "spawn_z_jitter": 0.10,
        "yaw_range": math.pi,
        "pillar_spacing": 1.2,
        "spawn_clearance": 1.4,
        "camera_additive_noise_std": 0.04,
        "camera_multiplicative_noise_std": 0.12,
        "revisit_penalty_scale": 0.05,
    },
    {
        "active_pillars": 8,
        "episode_length_s": 36.0,
        "spawn_z_jitter": 0.12,
        "yaw_range": math.pi,
        "pillar_spacing": 1.2,
        "spawn_clearance": 1.4,
        "camera_additive_noise_std": 0.05,
        "camera_multiplicative_noise_std": 0.15,
        "revisit_penalty_scale": 0.10,
    },
)


@configclass
class ThesisX152bExplorationEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 36.0
    observation_space = {
        "observations": THESIS_STATE_DIM,
        "observations_image": list(THESIS_IMAGE_SHAPE),
        "observations_map": list(THESIS_MAP_SHAPE),
        "critic_observations": THESIS_CRITIC_DIM,
    }
    state_space = THESIS_CRITIC_DIM
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=0.0, replicate_physics=True)
    terrain = make_plane_cfg()
    contact_sensor = make_contact_sensor_cfg()
    onboard_camera = None
    obstacles = make_exploration_room_cfg(8)
    camera_max_distance = PERCEPTION_MAX_RANGE
    camera_additive_noise_std = 0.05
    camera_multiplicative_noise_std = 0.15

    perception_width = PERCEPTION_WIDTH
    perception_height = PERCEPTION_HEIGHT
    perception_hfov_deg = PERCEPTION_HFOV_DEG
    perception_free_samples = 48
    perception_use_cuda_graph = False
    occupancy_map_shape = OCCUPANCY_MAP_SHAPE
    local_map_size = LOCAL_MAP_SIZE
    local_map_cell_size = LOCAL_MAP_CELL_SIZE

    curriculum_enabled = True
    curriculum_stage = 0
    curriculum_window_episodes = 512
    curriculum_min_episodes = 256
    curriculum_inactive_pillar_xy = (0.0, 6.5)

    map_recency_decay = 0.96
    map_observed_recency = 0.25
    map_new_info_recency = 1.0
    path_recency_radius = 0.3

    vae_enabled = False
    vae_weights_path: str | None = None
    vae_latent_dims = THESIS_VAE_LATENT_DIM
    vae_return_sampled_latent = False

    def __post_init__(self):
        super().__post_init__()
        self._refresh_observation_space()

    def _refresh_observation_space(self):
        self.observation_space = {
            "observations": THESIS_STATE_DIM,
            "observations_map": list(THESIS_MAP_SHAPE),
            "critic_observations": THESIS_CRITIC_DIM,
        }
        if self.vae_enabled:
            self.observation_space["observations_vae"] = self.vae_latent_dims
        else:
            self.observation_space["observations_image"] = list(THESIS_IMAGE_SHAPE)
        self.state_space = THESIS_CRITIC_DIM

    def set_vae_mode(self, enabled: bool = True):
        self.vae_enabled = bool(enabled)
        self._refresh_observation_space()


class ThesisX152bExplorationEnv(AirGymX152bBaseEnv):
    cfg: ThesisX152bExplorationEnvCfg

    def __init__(self, cfg: ThesisX152bExplorationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._wall_xy = WALL_XY.to(device=self.device)
        self._num_walls = int(self._wall_xy.shape[0])
        self._num_pillars = self._obstacles.num_objects - self._num_walls

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
        self._setup_perception()
        self._vae: DepthVAEEncoder | None = None
        if self.cfg.vae_enabled:
            self._vae = DepthVAEEncoder(
                weights_path=self.cfg.vae_weights_path,
                latent_dims=self.cfg.vae_latent_dims,
                max_range=self.cfg.camera_max_distance,
                return_sampled_latent=self.cfg.vae_return_sampled_latent,
                device=self.device,
            )
        self._known_free_3d = torch.zeros(
            (self.num_envs, *self.cfg.occupancy_map_shape),
            device=self.device,
            dtype=torch.bool,
        )
        self._known_occupied_3d = torch.zeros_like(self._known_free_3d)
        self._recency_3d = torch.zeros(
            (self.num_envs, *self.cfg.occupancy_map_shape),
            device=self.device,
            dtype=torch.float32,
        )
        self._visibility_grid_h = math.ceil((ROOM_X_LIMITS[1] - ROOM_X_LIMITS[0]) / VISIBILITY_GRID_CELL_SIZE)
        self._visibility_grid_w = math.ceil((ROOM_Y_LIMITS[1] - ROOM_Y_LIMITS[0]) / VISIBILITY_GRID_CELL_SIZE)
        self._visibility_num_cells = self._visibility_grid_h * self._visibility_grid_w
        self._last_new_visits = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_new_free_cells = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_new_occupied_cells = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_new_total_cells = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._last_frontier_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._last_revisit_ratio = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._last_min_depth = torch.full((self.num_envs,), PERCEPTION_MAX_RANGE, device=self.device)
        self._last_collision_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_out_of_bounds_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_bad_height_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_tip_termination = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._last_reward_terms: dict[str, torch.Tensor] = {}
        self._episode_metric_sums = {
            "free_info_reward": torch.zeros((self.num_envs,), device=self.device),
            "occupied_info_reward": torch.zeros((self.num_envs,), device=self.device),
            "frontier_reward": torch.zeros((self.num_envs,), device=self.device),
            "coverage_milestone_reward": torch.zeros((self.num_envs,), device=self.device),
            "sustained_gain_reward": torch.zeros((self.num_envs,), device=self.device),
            "revisit_penalty": torch.zeros((self.num_envs,), device=self.device),
            "risk_penalty": torch.zeros((self.num_envs,), device=self.device),
            "collision_penalty": torch.zeros((self.num_envs,), device=self.device),
            "bounds_penalty": torch.zeros((self.num_envs,), device=self.device),
            "height_penalty": torch.zeros((self.num_envs,), device=self.device),
            "action_jerk_penalty": torch.zeros((self.num_envs,), device=self.device),
        }
        self._curriculum_stage = self._clamp_curriculum_stage(int(self.cfg.curriculum_stage))
        self._curriculum_episode_history: list[dict[str, float]] = []
        self._curriculum_stage_advanced = False

        x_centers = ROOM_X_LIMITS[0] + VISIBILITY_GRID_CELL_SIZE * (
            torch.arange(self._visibility_grid_h, device=self.device, dtype=torch.float32) + 0.5
        )
        y_centers = ROOM_Y_LIMITS[0] + VISIBILITY_GRID_CELL_SIZE * (
            torch.arange(self._visibility_grid_w, device=self.device, dtype=torch.float32) + 0.5
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self._visibility_cell_centers = torch.stack((grid_x, grid_y), dim=-1)
        self._room_diagonal = math.hypot(ROOM_X_LIMITS[1] - ROOM_X_LIMITS[0], ROOM_Y_LIMITS[1] - ROOM_Y_LIMITS[0])
        self._local_map_index_base, self._local_map_position_base = build_local_map_bases(
            self.num_envs,
            self.cfg.local_map_size,
            self.cfg.local_map_cell_size,
            self.device,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._log_completed_episode_metrics(env_ids)
        super()._reset_idx(env_ids)

    def _setup_perception(self) -> None:
        num_boxes = int(self._obstacles.num_objects)
        self._perception = ThesisActivePerceptionWarpSensor(
            self.num_envs,
            num_boxes,
            height=self.cfg.perception_height,
            width=self.cfg.perception_width,
            horizontal_fov_deg=self.cfg.perception_hfov_deg,
            max_range=self.cfg.camera_max_distance,
            map_shape=self.cfg.occupancy_map_shape,
            device=self.device,
        )
        self._perception.set_env_origins(self._env_origins)
        pillar_half = torch.tensor(PILLAR_HALF_EXTENT, device=self.device, dtype=torch.float32).expand(
            self._num_pillars, 3
        )
        box_half = torch.cat((WALL_HALF_EXTENTS.to(self.device), pillar_half), dim=0)
        self._perception.set_box_half_extents(box_half)

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        stage = self._curriculum_settings()
        spawn_pos = sample_spawn_positions(
            num_resets,
            self.device,
            x_limits=ROOM_X_LIMITS,
            y_limits=ROOM_Y_LIMITS,
            z_height=FLY_HEIGHT,
            xy_margin=SPAWN_MARGIN,
            z_jitter=float(stage["spawn_z_jitter"]),
        )
        spawn_yaw = self._sample_curriculum_yaws(num_resets, float(stage["yaw_range"]))
        active_pillars = min(int(stage["active_pillars"]), self._num_pillars)
        if active_pillars > 0:
            active_pillar_xy = sample_pillar_positions(
                num_resets,
                active_pillars,
                self.device,
                x_limits=ROOM_X_LIMITS,
                y_limits=ROOM_Y_LIMITS,
                spawn_xy=spawn_pos[:, :2],
                wall_margin=PILLAR_WALL_MARGIN,
                pillar_spacing=float(stage["pillar_spacing"]),
                spawn_clearance=float(stage["spawn_clearance"]),
            )
        else:
            active_pillar_xy = torch.zeros((num_resets, 0, 2), device=self.device, dtype=torch.float32)
        inactive_count = self._num_pillars - active_pillars
        if inactive_count > 0:
            inactive_xy = torch.tensor(self.cfg.curriculum_inactive_pillar_xy, device=self.device).view(1, 1, 2)
            inactive_xy = inactive_xy.expand(num_resets, inactive_count, 2)
            pillar_xy = torch.cat((active_pillar_xy, inactive_xy), dim=1)
        else:
            pillar_xy = active_pillar_xy
        obstacle_pos, obstacle_quat = build_room_object_state(
            self._wall_xy,
            pillar_xy,
            wall_height=WALL_HEIGHT,
            pillar_height=PILLAR_HEIGHT,
        )
        self._write_object_collection_state_local(self._obstacles, env_ids, obstacle_pos, obstacle_quat)
        box_center_w = obstacle_pos + self._env_origins[env_ids].unsqueeze(1)
        self._perception.set_box_transforms(box_center_w, obstacle_quat, env_ids)

        quat = quat_from_euler_xyz(torch.zeros_like(spawn_yaw), torch.zeros_like(spawn_yaw), spawn_yaw)
        lin_vel = torch.zeros((num_resets, 3), device=self.device)
        ang_vel = torch.zeros((num_resets, 3), device=self.device)
        self._write_robot_state_local(env_ids, spawn_pos, quat, lin_vel, ang_vel)

        self._coverage_grid.reset(env_ids, spawn_pos[:, :2])
        self._last_new_visits[env_ids] = 0
        self._last_new_free_cells[env_ids] = 0
        self._last_new_occupied_cells[env_ids] = 0
        self._last_new_total_cells[env_ids] = 0
        self._last_frontier_count[env_ids] = 0.0
        self._last_revisit_ratio[env_ids] = 0.0
        self._last_min_depth[env_ids] = self.cfg.camera_max_distance
        self._last_collision_termination[env_ids] = False
        self._last_out_of_bounds_termination[env_ids] = False
        self._last_bad_height_termination[env_ids] = False
        self._last_tip_termination[env_ids] = False
        reset_thesis_map_tensors(self._known_free_3d, self._known_occupied_3d, self._recency_3d, env_ids)
        for value in self._episode_metric_sums.values():
            value[env_ids] = 0.0

        self._update_perception(force=True)
        self._update_active_map(env_ids)
        stamp_robot_recency(
            self._recency_3d,
            spawn_pos,
            env_ids,
            OCCUPANCY_MAP_BOUNDS_MIN,
            OCCUPANCY_MAP_BOUNDS_MAX,
            radius=float(self.cfg.path_recency_radius),
            value=1.0,
        )

    def _clamp_curriculum_stage(self, stage: int) -> int:
        return max(0, min(stage, len(CURRICULUM_STAGES) - 1))

    def _curriculum_settings(self) -> dict[str, float | int]:
        if self.cfg.curriculum_enabled:
            return CURRICULUM_STAGES[self._curriculum_stage]
        return {
            "active_pillars": self._num_pillars,
            "episode_length_s": self.cfg.episode_length_s,
            "spawn_z_jitter": 0.12,
            "yaw_range": math.pi,
            "pillar_spacing": PILLAR_SPACING,
            "spawn_clearance": SPAWN_CLEARANCE,
            "camera_additive_noise_std": self.cfg.camera_additive_noise_std,
            "camera_multiplicative_noise_std": self.cfg.camera_multiplicative_noise_std,
            "revisit_penalty_scale": 0.10,
        }

    def _sample_curriculum_yaws(self, num_resets: int, yaw_range: float) -> torch.Tensor:
        if yaw_range <= 0.0:
            return torch.zeros((num_resets,), device=self.device, dtype=torch.float32)
        if yaw_range >= math.pi:
            return sample_yaws(num_resets, self.device)
        return 2.0 * yaw_range * torch.rand((num_resets,), device=self.device, dtype=torch.float32) - yaw_range

    def _camera_additive_noise_std(self) -> float:
        return float(self._curriculum_settings()["camera_additive_noise_std"])

    def _camera_multiplicative_noise_std(self) -> float:
        return float(self._curriculum_settings()["camera_multiplicative_noise_std"])

    @property
    def max_episode_length_s(self) -> float:
        if hasattr(self, "_curriculum_stage") and self.cfg.curriculum_enabled:
            return float(self._curriculum_settings()["episode_length_s"])
        return float(self.cfg.episode_length_s)

    def _resolve_map_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _update_active_map(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._update_perception()
        new_free, new_occupied, new_total = update_thesis_map_tensors(
            self._perception.occupancy_map,
            self._known_free_3d,
            self._known_occupied_3d,
            self._recency_3d,
            self._resolve_map_env_ids(env_ids),
            recency_decay=float(self.cfg.map_recency_decay),
            observed_recency=float(self.cfg.map_observed_recency),
            new_info_recency=float(self.cfg.map_new_info_recency),
        )
        return new_free, new_occupied, new_total

    def _thesis_bev_map(self) -> torch.Tensor:
        return project_thesis_map_to_bev(
            self._known_free_3d,
            self._known_occupied_3d,
            self._recency_3d,
            x_limits=ROOM_X_LIMITS,
            y_limits=ROOM_Y_LIMITS,
            z_limits=VISIBILITY_GRID_Z_LIMITS,
            cell_size=VISIBILITY_GRID_CELL_SIZE,
            map_bounds_min=OCCUPANCY_MAP_BOUNDS_MIN,
            map_bounds_max=OCCUPANCY_MAP_BOUNDS_MAX,
        )

    def _known_space_ratio(self) -> torch.Tensor:
        known = self._known_free_3d | self._known_occupied_3d
        return known.flatten(1).to(dtype=torch.float32).mean(dim=-1)

    def _occupied_space_ratio(self) -> torch.Tensor:
        return self._known_occupied_3d.flatten(1).to(dtype=torch.float32).mean(dim=-1)

    def _frontier_features(
        self,
        root_pos_local: torch.Tensor,
        world_to_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bev = self._thesis_bev_map()
        known_free = bev[:, 0] > 0.0
        known_occupied = bev[:, 1] > 0.0
        unknown = bev[:, 2] > 0.0
        unknown_neighbor = F.max_pool2d(unknown.to(dtype=torch.float32).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        frontier = known_free & (unknown_neighbor > 0.0)

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
            (local_delta[..., 0] >= 0.0)
            & (local_delta[..., 0] < 4.5)
            & (local_delta[..., 1] >= -2.25)
            & (local_delta[..., 1] < 2.25)
        )
        ahead_count = ahead.flatten(1).sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        unknown_ahead_ratio = (unknown & ahead).flatten(1).sum(dim=-1).to(dtype=torch.float32) / ahead_count
        occupied_ahead_ratio = (known_occupied & ahead).flatten(1).sum(dim=-1).to(dtype=torch.float32) / ahead_count
        frontier_count_norm = frontier.flatten(1).sum(dim=-1).to(dtype=torch.float32) / float(self._visibility_num_cells)
        self._last_frontier_count.copy_(frontier_count_norm)
        return nearest_distance, nearest_bearing, unknown_ahead_ratio, occupied_ahead_ratio, frontier_count_norm, bev

    def _state_observation(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        world_to_local, euler_local, vel_local, ang_vel_local = self._compute_yaw_local_frame()
        coverage_ratio = self._coverage_grid.coverage_ratio()
        known_ratio = self._known_space_ratio()
        occupied_ratio = self._occupied_space_ratio()
        frontier_distance, frontier_bearing, unknown_ahead, occupied_ahead, frontier_count, _ = self._frontier_features(
            root_pos_local, world_to_local
        )
        depth_metric = self._camera_depth_metric_image()
        min_depth = depth_metric.view(self.num_envs, -1).amin(dim=-1)
        mean_depth = depth_metric.view(self.num_envs, -1).mean(dim=-1)
        self._last_min_depth.copy_(min_depth)

        position_obs = torch.stack(
            (
                root_pos_local[:, 0] / ROOM_HALF_EXTENT,
                root_pos_local[:, 1] / ROOM_HALF_EXTENT,
                (root_pos_local[:, 2] - FLY_HEIGHT) / FLY_HEIGHT,
            ),
            dim=-1,
        )
        map_progress = torch.stack(
            (
                coverage_ratio,
                known_ratio,
                occupied_ratio,
                self._last_new_free_cells.to(dtype=torch.float32) / 1024.0,
                self._last_new_occupied_cells.to(dtype=torch.float32) / 256.0,
                frontier_distance,
                frontier_bearing,
                unknown_ahead,
                occupied_ahead,
                frontier_count,
                self._last_revisit_ratio,
            ),
            dim=-1,
        )
        depth_obs = torch.stack(
            (
                min_depth / self.cfg.camera_max_distance,
                mean_depth / self.cfg.camera_max_distance,
            ),
            dim=-1,
        )
        curriculum_obs = torch.stack(
            (
                torch.full((self.num_envs,), float(self._curriculum_stage), device=self.device)
                / float(len(CURRICULUM_STAGES) - 1),
                self.episode_length_buf.to(dtype=torch.float32) / float(max(1, self.max_episode_length)),
                torch.full(
                    (self.num_envs,),
                    float(CURRICULUM_STAGES[self._curriculum_stage]["active_pillars"]),
                    device=self.device,
                )
                / float(max(1, self._num_pillars)),
            ),
            dim=-1,
        )
        observations = torch.cat(
            (
                position_obs,
                euler_local,
                vel_local,
                ang_vel_local,
                self._policy_actions,
                map_progress,
                depth_obs,
                curriculum_obs,
            ),
            dim=-1,
        )
        if observations.shape[-1] != THESIS_STATE_DIM:
            raise RuntimeError(f"Thesis observation contract expected {THESIS_STATE_DIM}, got {observations.shape[-1]}")
        return observations

    def _image_observation(self) -> torch.Tensor:
        depth = self._camera_depth_metric_image()
        depth_norm = (depth / self.cfg.camera_max_distance).clamp(0.0, 1.0)
        near_field = torch.clamp((1.5 - depth) / 1.5, 0.0, 1.0)
        return torch.cat((depth_norm, near_field), dim=1)

    def _vae_observation(self) -> torch.Tensor:
        if self._vae is None:
            raise RuntimeError("Thesis VAE observations requested but cfg.vae_enabled is False.")
        return self._vae.encode(self._camera_depth_metric_image())

    def _local_map_observation(self) -> torch.Tensor:
        map_channels = compose_thesis_map_channels(self._known_free_3d, self._known_occupied_3d, self._recency_3d)
        return extract_centered_map_channels(
            map_channels,
            self.cfg.local_map_size,
            self._root_pos_local(),
            self._robot.data.root_quat_w,
            OCCUPANCY_MAP_BOUNDS_MIN,
            OCCUPANCY_MAP_BOUNDS_MAX,
            self._local_map_index_base,
            self._local_map_position_base,
        )

    def _critic_observation(self, state_obs: torch.Tensor | None = None) -> torch.Tensor:
        if state_obs is None:
            state_obs = self._state_observation()
        root_pos_local = self._root_pos_local()
        _, _, vel_local, ang_vel_local = self._compute_yaw_local_frame()
        coverage_ratio = self._coverage_grid.coverage_ratio()
        known_ratio = self._known_space_ratio()
        occupied_ratio = self._occupied_space_ratio()
        bev = self._thesis_bev_map()
        free_ratio_bev = bev[:, 0].flatten(1).mean(dim=-1)
        occupied_ratio_bev = bev[:, 1].flatten(1).mean(dim=-1)
        unknown_ratio_bev = bev[:, 2].flatten(1).mean(dim=-1)
        recency_mean_bev = bev[:, 3].flatten(1).mean(dim=-1)
        collision = self._collision_state().to(dtype=torch.float32)
        out_of_bounds_margin = torch.stack(
            (
                root_pos_local[:, 0] - ROOM_X_LIMITS[0],
                ROOM_X_LIMITS[1] - root_pos_local[:, 0],
                root_pos_local[:, 1] - ROOM_Y_LIMITS[0],
                ROOM_Y_LIMITS[1] - root_pos_local[:, 1],
            ),
            dim=-1,
        ).amin(dim=-1)
        critic_extra = torch.stack(
            (
                coverage_ratio,
                known_ratio,
                occupied_ratio,
                free_ratio_bev,
                occupied_ratio_bev,
                unknown_ratio_bev,
                recency_mean_bev,
                self._last_new_free_cells.to(dtype=torch.float32) / 1024.0,
                self._last_new_occupied_cells.to(dtype=torch.float32) / 256.0,
                self._last_new_total_cells.to(dtype=torch.float32) / 1280.0,
                self._last_frontier_count,
                self._last_revisit_ratio,
                self._last_min_depth / self.cfg.camera_max_distance,
                torch.linalg.vector_norm(vel_local, dim=-1) / 6.0,
                torch.linalg.vector_norm(ang_vel_local, dim=-1) / 6.0,
                collision,
            ),
            dim=-1,
        )
        margins = torch.stack(
            (
                out_of_bounds_margin / ROOM_HALF_EXTENT,
                (root_pos_local[:, 2] - 0.35) / FLY_HEIGHT,
                (2.0 - root_pos_local[:, 2]) / FLY_HEIGHT,
            ),
            dim=-1,
        )
        critic_obs = torch.cat((state_obs, critic_extra, margins), dim=-1)
        if critic_obs.shape[-1] != THESIS_CRITIC_DIM:
            raise RuntimeError(f"Thesis critic contract expected {THESIS_CRITIC_DIM}, got {critic_obs.shape[-1]}")
        return critic_obs

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        self._update_perception()
        observations = self._state_observation()
        observations_map = self._local_map_observation()
        critic_observations = self._critic_observation(observations)
        obs = {
            "policy": observations,
            "observations": observations,
            "observations_map": observations_map,
            "critic_observations": critic_observations,
            "critic_grid": self._thesis_bev_map(),
        }
        if self.cfg.vae_enabled:
            obs["observations_vae"] = self._vae_observation()
        else:
            obs["observations_image"] = self._image_observation()
        return obs

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        self._update_perception()
        root_pos_local = self._root_pos_local()
        world_to_local, _, _, ang_vel_local = self._compute_yaw_local_frame()
        current_cell, current_in_bounds = self._coverage_grid.xy_to_grid(root_pos_local[:, :2])
        previously_visited = self._coverage_grid.visited[
            torch.arange(self.num_envs, device=self.device), current_cell[:, 0], current_cell[:, 1]
        ]
        new_visit_count, in_bounds = self._coverage_grid.update(None, root_pos_local[:, :2])
        self._last_new_visits.copy_(new_visit_count)
        new_free_cells, new_occupied_cells, new_total_cells = self._update_active_map()
        self._last_new_free_cells.copy_(new_free_cells)
        self._last_new_occupied_cells.copy_(new_occupied_cells)
        self._last_new_total_cells.copy_(new_total_cells)
        self._frontier_features(root_pos_local, world_to_local)
        stamp_robot_recency(
            self._recency_3d,
            root_pos_local,
            torch.arange(self.num_envs, device=self.device, dtype=torch.long),
            OCCUPANCY_MAP_BOUNDS_MIN,
            OCCUPANCY_MAP_BOUNDS_MAX,
            radius=float(self.cfg.path_recency_radius),
            value=1.0,
        )

        depth_metric = self._camera_depth_metric_image()
        min_depth = depth_metric.view(self.num_envs, -1).amin(dim=-1)
        self._last_min_depth.copy_(min_depth)
        revisit_ratio = (
            previously_visited.to(dtype=torch.float32)
            * current_in_bounds.to(dtype=torch.float32)
            * (new_visit_count == 0).to(dtype=torch.float32)
        )
        self._last_revisit_ratio.copy_(revisit_ratio)

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        collision = self._collision_state()
        out_of_bounds = ~in_bounds
        bad_height = (root_pos_local[:, 2] < 0.35) | (root_pos_local[:, 2] > 2.0)

        reward, reward_info = compute_thesis_exploration_reward(
            new_free_cells=new_free_cells.to(dtype=torch.float32),
            new_occupied_cells=new_occupied_cells.to(dtype=torch.float32),
            new_visit_count=new_visit_count.to(dtype=torch.float32),
            coverage_ratio=self._coverage_grid.coverage_ratio(),
            frontier_count=self._last_frontier_count,
            revisit_ratio=revisit_ratio,
            min_depth=min_depth,
            root_height=root_pos_local[:, 2],
            up_z=ups[:, 2],
            ang_vel_norm=torch.linalg.vector_norm(ang_vel_local, dim=-1),
            action_diff_norm=torch.linalg.vector_norm(self._actions - self._previous_actions, dim=-1),
            collision=collision,
            out_of_bounds=out_of_bounds,
            bad_height=bad_height,
            revisit_penalty_scale=float(self._curriculum_settings()["revisit_penalty_scale"]),
            device=self.device,
        )
        reward_info["known_space_ratio"] = self._known_space_ratio()
        reward_info["occupied_space_ratio"] = self._occupied_space_ratio()
        reward_info["new_total_cells"] = new_total_cells.to(dtype=torch.float32)
        self._last_reward_terms = {k: v for k, v in reward_info.items() if isinstance(v, torch.Tensor)}
        for key in self._episode_metric_sums:
            self._episode_metric_sums[key] += reward_info[key]
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)

        out_of_bounds = root_pos_local[:, 0] < ROOM_X_LIMITS[0]
        out_of_bounds |= root_pos_local[:, 0] > ROOM_X_LIMITS[1]
        out_of_bounds |= root_pos_local[:, 1] < ROOM_Y_LIMITS[0]
        out_of_bounds |= root_pos_local[:, 1] > ROOM_Y_LIMITS[1]
        bad_height = (root_pos_local[:, 2] < 0.35) | (root_pos_local[:, 2] > 2.0)
        tipped = ups[:, 2] < 0.0
        collision = self._collision_state()

        self._last_out_of_bounds_termination.copy_(out_of_bounds)
        self._last_bad_height_termination.copy_(bad_height)
        self._last_tip_termination.copy_(tipped)
        self._last_collision_termination.copy_(collision)
        return out_of_bounds | bad_height | tipped | collision

    def _log_completed_episode_metrics(self, env_ids: torch.Tensor) -> None:
        self._curriculum_stage_advanced = False
        episode_lengths = self.episode_length_buf[env_ids].to(dtype=torch.float32)
        completed_episode = self.reset_terminated[env_ids] | self.reset_time_outs[env_ids]
        valid_episode = completed_episode & (episode_lengths > 0)
        if not torch.any(valid_episode):
            self.extras["log"] = self._curriculum_log()
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
        log["Metrics/frontier_count"] = self._last_frontier_count[valid_env_ids].mean()
        self._record_curriculum_episodes(valid_env_ids, episode_lengths, coverage_ratio, known_space_ratio)
        self._maybe_advance_curriculum()
        log.update(self._curriculum_log())
        self.extras["log"] = log

    def _record_curriculum_episodes(
        self,
        env_ids: torch.Tensor,
        episode_lengths: torch.Tensor,
        coverage_ratio: torch.Tensor,
        known_space_ratio: torch.Tensor,
    ) -> None:
        if not self.cfg.curriculum_enabled:
            return
        for i, env_id in enumerate(env_ids.tolist()):
            self._curriculum_episode_history.append(
                {
                    "episode_length": float(episode_lengths[i].item()),
                    "collision": float(self._last_collision_termination[env_id].item()),
                    "out_of_bounds": float(self._last_out_of_bounds_termination[env_id].item()),
                    "bad_height": float(self._last_bad_height_termination[env_id].item()),
                    "time_out": float(self.reset_time_outs[env_id].item()),
                    "coverage_ratio": float(coverage_ratio[i].item()),
                    "known_space_ratio": float(known_space_ratio[i].item()),
                }
            )
        max_window = max(1, int(self.cfg.curriculum_window_episodes))
        if len(self._curriculum_episode_history) > max_window:
            self._curriculum_episode_history = self._curriculum_episode_history[-max_window:]

    def _curriculum_window_metrics(self) -> dict[str, float]:
        count = len(self._curriculum_episode_history)
        if count == 0:
            return {
                "episode_count": 0.0,
                "mean_episode_length": 0.0,
                "collision_rate": 0.0,
                "out_of_bounds_rate": 0.0,
                "bad_height_rate": 0.0,
                "time_out_rate": 0.0,
                "coverage_ratio": 0.0,
                "known_space_ratio": 0.0,
            }
        metrics = {"episode_count": float(count)}
        for key in (
            "episode_length",
            "collision",
            "out_of_bounds",
            "bad_height",
            "time_out",
            "coverage_ratio",
            "known_space_ratio",
        ):
            metrics[key] = sum(item[key] for item in self._curriculum_episode_history) / count
        metrics["mean_episode_length"] = metrics.pop("episode_length")
        metrics["collision_rate"] = metrics.pop("collision")
        metrics["out_of_bounds_rate"] = metrics.pop("out_of_bounds")
        metrics["bad_height_rate"] = metrics.pop("bad_height")
        metrics["time_out_rate"] = metrics.pop("time_out")
        return metrics

    def _maybe_advance_curriculum(self) -> None:
        if not self.cfg.curriculum_enabled or self._curriculum_stage >= len(CURRICULUM_STAGES) - 1:
            return
        if len(self._curriculum_episode_history) < int(self.cfg.curriculum_min_episodes):
            return

        metrics = self._curriculum_window_metrics()
        thresholds = (
            (0.15, 0.20, 0.90),
            (0.25, 0.32, 0.85),
            (0.35, 0.42, 0.80),
            (0.45, 0.55, 0.75),
        )
        coverage_target, known_target, timeout_target = thresholds[self._curriculum_stage]
        should_advance = (
            metrics["coverage_ratio"] >= coverage_target
            and metrics["known_space_ratio"] >= known_target
            and metrics["time_out_rate"] >= timeout_target
            and metrics["collision_rate"] < 0.12
            and metrics["bad_height_rate"] < 0.10
            and metrics["out_of_bounds_rate"] < 0.15
        )
        if should_advance:
            self._curriculum_stage += 1
            self._curriculum_stage_advanced = True
            self._curriculum_episode_history.clear()

    def _curriculum_log(self) -> dict[str, float | int]:
        stage = self._curriculum_settings()
        metrics = self._curriculum_window_metrics()
        return {
            "Curriculum/stage": self._curriculum_stage if self.cfg.curriculum_enabled else -1,
            "Curriculum/active_pillars": int(stage["active_pillars"]),
            "Curriculum/stage_advanced": int(self._curriculum_stage_advanced),
            "Curriculum/window_episode_count": metrics["episode_count"],
            "Curriculum/window_time_out_rate": metrics["time_out_rate"],
            "Curriculum/window_collision_rate": metrics["collision_rate"],
            "Curriculum/window_out_of_bounds_rate": metrics["out_of_bounds_rate"],
            "Curriculum/window_bad_height_rate": metrics["bad_height_rate"],
            "Curriculum/window_mean_episode_length": metrics["mean_episode_length"],
            "Curriculum/window_coverage_ratio": metrics["coverage_ratio"],
            "Curriculum/window_known_space_ratio": metrics["known_space_ratio"],
        }


__all__ = [
    "CURRICULUM_STAGES",
    "THESIS_CRITIC_DIM",
    "THESIS_IMAGE_SHAPE",
    "THESIS_MAP_SHAPE",
    "THESIS_STATE_DIM",
    "THESIS_VAE_LATENT_DIM",
    "ThesisX152bExplorationEnv",
    "ThesisX152bExplorationEnvCfg",
    "compute_thesis_exploration_reward",
]
