# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .base_env import AirGymX152bBaseEnv
from .base_env_cfg import AirGymX152bBaseEnvCfg
from .task_common import (
    make_contact_sensor_cfg,
    make_goal_cfg,
    make_onboard_camera_cfg,
    make_plane_cfg,
    make_planning_obstacles_cfg,
)


LENGTH = 8.0
WIDTH = 4.0
FLY_HEIGHT = 1.5


@configclass
class AirGymX152bPlanningEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 16.0
    observation_space = 16
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=14.0, replicate_physics=True)
    terrain = make_plane_cfg()
    contact_sensor = make_contact_sensor_cfg()
    onboard_camera = make_onboard_camera_cfg()
    goal = make_goal_cfg()
    obstacles = make_planning_obstacles_cfg(40)
    camera_additive_noise_std = 0.1
    camera_multiplicative_noise_std = 0.3


class AirGymX152bPlanningEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bPlanningEnvCfg

    def __init__(self, cfg: AirGymX152bPlanningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._previous_root_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _planning_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_global = self._goal_pos_local() - self._root_pos_local()
        world_to_local, euler_local, vel_local, ang_vel_local = self._compute_yaw_local_frame()
        pos_diff_local = torch.einsum("bij,bj->bi", world_to_local, forward_global)
        goal_dir = pos_diff_local / torch.linalg.vector_norm(pos_diff_local, dim=-1, keepdim=True).clamp_min(1.0e-6)
        related_dist = torch.linalg.vector_norm(forward_global, dim=-1)
        return pos_diff_local, euler_local, vel_local, ang_vel_local, goal_dir, related_dist

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)
        num_obstacles = self._obstacles.num_objects

        obstacle_pos = torch.zeros(num_resets, num_obstacles, 3, device=self.device)
        obstacle_pos[:, :, 0] = LENGTH * sample_uniform(-1.0, 1.0, (num_resets, num_obstacles), self.device)
        obstacle_pos[:, :, 1] = WIDTH * sample_uniform(-1.0, 1.0, (num_resets, num_obstacles), self.device)
        obstacle_pos[:, :, 2] = 1.0
        obstacle_angles = torch.zeros(num_resets, num_obstacles, 3, device=self.device)
        obstacle_angles[:, :, 2] = sample_uniform(-torch.pi, torch.pi, (num_resets, num_obstacles), self.device)
        obstacle_quat = quat_from_euler_xyz(
            obstacle_angles[:, :, 0].reshape(-1),
            obstacle_angles[:, :, 1].reshape(-1),
            obstacle_angles[:, :, 2].reshape(-1),
        ).reshape(num_resets, num_obstacles, 4)
        self._write_object_collection_state_local(self._obstacles, env_ids, obstacle_pos, obstacle_quat)

        goal_pos = torch.zeros(num_resets, 3, device=self.device)
        goal_pos[:, 0] = LENGTH + 0.5
        goal_pos[:, 1] = 1.5 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1)
        goal_pos[:, 2] = FLY_HEIGHT
        goal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        self._write_rigid_object_state_local(self._goal, env_ids, goal_pos, goal_quat)

        pos = torch.zeros(num_resets, 3, device=self.device)
        pos[:, 0] = -LENGTH - 0.5
        pos[:, 2] = FLY_HEIGHT
        init_yaw = torch.atan2(goal_pos[:, 1] - pos[:, 1], goal_pos[:, 0] - pos[:, 0])
        quat = quat_from_euler_xyz(torch.zeros_like(init_yaw), torch.zeros_like(init_yaw), init_yaw)
        lin_vel = torch.zeros(num_resets, 3, device=self.device)
        ang_vel = torch.zeros(num_resets, 3, device=self.device)
        self._write_robot_state_local(env_ids, pos, quat, lin_vel, ang_vel)

        self._previous_root_pos[env_ids] = 0.0

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        _, euler_local, vel_local, ang_vel_local, goal_dir, _ = self._planning_state()
        obs = torch.cat((goal_dir, euler_local, vel_local, ang_vel_local, self._policy_actions), dim=-1)
        return {
            "policy": obs,
            "observation": obs,
            "image": self._camera_depth_image(),
        }

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        pos_diff_local, _, vel_local, ang_vel_local, goal_dir, related_dist = self._planning_state()
        depth = self._camera_depth_image()
        esdf_dist = depth.view(self.num_envs, -1).amin(dim=-1)

        action_diff = self._actions - self._previous_actions
        continous_action_reward = 0.2 * torch.linalg.vector_norm(ang_vel_local, dim=-1)
        if self.cfg.ctl_mode in {"pos", "vel", "prop"}:
            continous_action_reward += 0.2 * torch.linalg.vector_norm(action_diff, dim=-1)
            thrust_reward = torch.zeros(self.num_envs, device=self.device)
        else:
            continous_action_reward += 0.2 * torch.linalg.vector_norm(action_diff[:, :-1], dim=-1)
            thrust_reward = 0.5 * (1.0 - torch.abs(self.cfg.hover_thrust - self._actions[:, -1]))

        forward_reward = 0.1 * (
            torch.linalg.vector_norm(self._goal_pos_local() - self._previous_root_pos, dim=-1)
            - torch.linalg.vector_norm(self._goal_pos_local() - self._root_pos_local(), dim=-1)
        )

        heading_reward = goal_dir[:, 0]
        speed_reward = -0.5 * (1.0 - torch.exp(-2.0 * torch.square(vel_local[:, 0] - 1.0)))

        z = self._root_pos_local()[:, 2]
        zeros = torch.zeros_like(z)
        z_reward = torch.minimum(torch.minimum(z - 1.8, zeros), 1.2 - z)

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        ups_reward = torch.square((ups[:, 2] + 1.0) / 2.0)

        esdf_reward = 0.5 * (1.0 - torch.exp(-0.5 * torch.square(esdf_dist)))
        alive_reward = torch.where(esdf_dist > 0.3, torch.zeros_like(esdf_dist), -torch.ones_like(esdf_dist))

        reach_goal = related_dist < 0.3
        reach_goal_reward = torch.where(reach_goal, torch.full_like(related_dist, 200.0), torch.zeros_like(related_dist))

        reward = (
            continous_action_reward
            + forward_reward
            + alive_reward
            + esdf_reward
            + ups_reward
            + z_reward
            + speed_reward
            + heading_reward
            + thrust_reward
            + reach_goal_reward
        )

        reward_info = {
            "continous_action_reward": continous_action_reward,
            "heading_reward": heading_reward,
            "speed_reward": speed_reward,
            "forward_reward": forward_reward,
            "alive_reward": alive_reward,
            "ups_reward": ups_reward,
            "z_reward": z_reward,
            "esdf_reward": esdf_reward,
            "thrust_reward": thrust_reward,
            "reach_goal_reward": reach_goal_reward,
            "reward": reward,
        }
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        _, _, _, _, goal_dir, related_dist = self._planning_state()
        root_pos = self._root_pos_local()
        terminated = root_pos[:, 2] < FLY_HEIGHT - 0.3
        terminated |= root_pos[:, 2] > FLY_HEIGHT + 0.3
        terminated |= root_pos[:, 0] < -LENGTH - 0.5
        terminated |= root_pos[:, 0] > LENGTH + 0.5
        terminated |= root_pos[:, 1] < -WIDTH
        terminated |= root_pos[:, 1] > WIDTH
        terminated |= self._collision_state()
        terminated |= related_dist < 0.3
        terminated |= goal_dir[:, 0] < 0.25
        return terminated

    def _after_rewards(self):
        self._previous_root_pos.copy_(self._root_pos_local())
