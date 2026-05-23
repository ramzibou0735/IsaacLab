# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, sample_uniform

from .base_env import AirGymX152bBaseEnv, compute_yaw_diff
from .base_env_cfg import AirGymX152bBaseEnvCfg
from .task_common import make_contact_sensor_cfg, make_moving_obstacle_cfg, make_onboard_camera_cfg, make_plane_cfg


@configclass
class AirGymX152bAvoidEnvCfg(AirGymX152bBaseEnvCfg):
    episode_length_s = 6.0
    observation_space = 16
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=6.0, replicate_physics=True)
    terrain = make_plane_cfg()
    contact_sensor = make_contact_sensor_cfg()
    onboard_camera = make_onboard_camera_cfg()
    moving_obstacle = make_moving_obstacle_cfg()
    camera_additive_noise_std = 0.1
    camera_multiplicative_noise_std = 0.3


class AirGymX152bAvoidEnv(AirGymX152bBaseEnv):
    cfg: AirGymX152bAvoidEnvCfg

    def _calculate_object_velocity(self, object_position: torch.Tensor, horizontal_speed: float) -> torch.Tensor:
        valid_positions = object_position[:, 0] != -999.0
        velocity = torch.zeros_like(object_position)
        if not torch.any(valid_positions):
            return velocity

        positions = object_position[valid_positions]
        num_valid = positions.shape[0]
        drone_position = 0.3 * sample_uniform(-1.0, 1.0, (num_valid, 3), self.device)
        drone_position += torch.tensor([0.0, 0.0, 1.0], device=self.device)

        direction = drone_position - positions
        distance_xy = torch.linalg.vector_norm(direction[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-6)
        unit_direction_xy = direction[:, :2] / distance_xy

        speed = torch.full_like(distance_xy, horizontal_speed)
        time_to_target = distance_xy / speed

        z_c = positions[:, 2:3]
        z_u = drone_position[:, 2:3]
        v_z = (z_u - z_c + 0.5 * 9.81 * time_to_target.square()) / time_to_target
        v_x = unit_direction_xy[:, 0:1] * speed
        v_y = unit_direction_xy[:, 1:2] * speed

        velocity[valid_positions] = torch.cat((v_x, v_y, v_z), dim=-1)
        return velocity

    def _reset_task(self, env_ids: torch.Tensor):
        num_resets = len(env_ids)

        obstacle_pos = torch.zeros(num_resets, 3, device=self.device)
        obstacle_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(num_resets, 1)
        obstacle_lin_vel = torch.zeros(num_resets, 3, device=self.device)

        random_values = sample_uniform(0.0, 1.0, (num_resets, 1), self.device)
        random_pos_mask = random_values[:, 0] < 0.8
        fixed_pos_mask = ~random_pos_mask

        if torch.any(random_pos_mask):
            random_env_ids = env_ids[random_pos_mask]
            random_num_resets = int(random_pos_mask.sum())
            theta = (torch.pi / 6.0) * sample_uniform(-1.0, 1.0, (random_num_resets, 1), self.device).squeeze(-1)
            obstacle_pos[random_pos_mask, 0] = 4.2 * torch.cos(theta)
            obstacle_pos[random_pos_mask, 1] = 4.2 * torch.sin(theta)
            obstacle_pos[random_pos_mask, 2] = 1.4
            obstacle_lin_vel[random_pos_mask] = self._calculate_object_velocity(obstacle_pos[random_pos_mask], 4.5)
            self._write_rigid_object_state_local(
                self._moving_obstacle,
                random_env_ids,
                obstacle_pos[random_pos_mask],
                obstacle_quat[random_pos_mask],
                obstacle_lin_vel[random_pos_mask],
            )

        if torch.any(fixed_pos_mask):
            fixed_env_ids = env_ids[fixed_pos_mask]
            obstacle_pos[fixed_pos_mask] = torch.tensor([-999.0, -999.0, 0.0], device=self.device)
            self._write_rigid_object_state_local(
                self._moving_obstacle,
                fixed_env_ids,
                obstacle_pos[fixed_pos_mask],
                obstacle_quat[fixed_pos_mask],
            )

        pos = torch.zeros(num_resets, 3, device=self.device)
        pos[:, :2] = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 2), self.device)
        pos[:, 2] = 0.2 * sample_uniform(-1.0, 1.0, (num_resets, 1), self.device).squeeze(-1) + 1.0
        angles = torch.cat(
            (
                0.01 * sample_uniform(-torch.pi, torch.pi, (num_resets, 2), self.device),
                0.05 * sample_uniform(-torch.pi, torch.pi, (num_resets, 1), self.device),
            ),
            dim=-1,
        )
        quat = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        lin_vel = torch.zeros(num_resets, 3, device=self.device)
        ang_vel = torch.zeros(num_resets, 3, device=self.device)
        self._write_robot_state_local(env_ids, pos, quat, lin_vel, ang_vel)

    def _get_task_observations(self) -> dict[str, torch.Tensor]:
        _, euler_local, vel_local, ang_vel_local = self._compute_yaw_local_frame()
        obs = torch.cat(
            (
                self._root_pos_local() - self._target_states[:, 9:12],
                euler_local,
                vel_local,
                ang_vel_local,
                self._policy_actions,
            ),
            dim=-1,
        )
        return {
            "policy": obs,
            "observation": obs,
            "image": self._camera_depth_image(),
        }

    def _compute_reward_and_metrics(self) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        target_positions = self._target_states[:, 9:12]
        relative_positions = target_positions - self._root_pos_local()
        root_euler = self._root_euler_xyz()
        relative_heading = compute_yaw_diff(torch.zeros_like(root_euler[:, 2]), root_euler[:, 2])
        distance = torch.linalg.vector_norm(torch.cat((relative_positions, relative_heading.unsqueeze(-1)), dim=-1), dim=-1)
        pose_reward = 1.0 / (1.0 + torch.square(1.6 * distance))

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        ups_reward = torch.square((ups[:, 2] + 1.0) / 2.0)

        spinnage = torch.square(self._robot.data.root_ang_vel_w[:, 2])
        spin_reward = 1.0 / (1.0 + torch.square(spinnage))

        effort_reward = 0.1 * torch.exp(-self._actions.pow(2).sum(dim=-1))
        if self.cfg.ctl_mode in {"pos", "vel", "prop"}:
            action_diff = torch.linalg.vector_norm(self._actions - self._previous_actions, dim=-1)
            thrust_reward = torch.zeros(self.num_envs, device=self.device)
        else:
            action_diff = torch.linalg.vector_norm(self._actions[:, :-1] - self._previous_actions[:, :-1], dim=-1)
            thrust_reward = 0.05 * (1.0 - torch.abs(self.cfg.hover_thrust - self._actions[:, -1]))
        action_smoothness_reward = 0.1 * torch.exp(-action_diff)

        collision = self._collision_state()
        alive_reward = torch.where(collision, torch.full_like(pose_reward, -500.0), torch.full_like(pose_reward, 0.5))

        reward = (
            pose_reward
            + pose_reward * (ups_reward + spin_reward)
            + effort_reward
            + action_smoothness_reward
            + thrust_reward
            + alive_reward
        )

        reward_info = {
            "pose_reward": pose_reward,
            "ups_reward": ups_reward,
            "spin_reward": spin_reward,
            "effort_reward": effort_reward,
            "action_smoothness_reward": action_smoothness_reward,
            "thrust_reward": thrust_reward,
            "alive_reward": alive_reward,
            "reward": reward,
        }
        return reward, reward_info

    def _compute_terminated(self) -> torch.Tensor:
        target_positions = self._target_states[:, 9:12]
        relative_positions = target_positions - self._root_pos_local()
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        ups = quat_apply(self._robot.data.root_quat_w, up_axis)
        terminated = self._root_pos_local()[:, 2] < 0.3
        terminated |= self._root_pos_local()[:, 2] > 1.7
        terminated |= torch.linalg.vector_norm(relative_positions, dim=-1) > 2.0
        terminated |= ups[:, 2] < 0.0
        terminated |= self._collision_state()
        return terminated
